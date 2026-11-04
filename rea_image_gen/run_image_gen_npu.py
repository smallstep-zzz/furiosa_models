#!/usr/bin/env python3
"""
Furiosa Renegade NPU 기반 다중 모델 이미지 생성 파이프라인
- 지원 모델:
  1. SDXL-Turbo (`sdxl`)
  2. FLUX.1-dev (`flux`)
  3. FLUX 3 Image Klein (`flux3-klein`)
  4. Qwen-Image-2.1 (`qwen`)
- NPU 컴파일 가속: 시스템 CPU 코어 수 - 2 (`RAYON_NUM_THREADS`, `OMP_NUM_THREADS`) 자동 적용
- Furiosa backend가 지원하는 그래프를 NPU에서 실행하고, 미지원 그래프는 CPU eager fallback
"""

import os
import sys
import time
import argparse
import multiprocessing
from pathlib import Path

# 1. 멀티코어 컴파일러 가속 환경변수 자동 설정 (전체 코어 수 - 2)
TOTAL_CORES = multiprocessing.cpu_count()
TARGET_THREADS = max(1, TOTAL_CORES - 2)

os.environ.setdefault("RAYON_NUM_THREADS", str(TARGET_THREADS))
os.environ.setdefault("OMP_NUM_THREADS", "8")
AUDIT_DIR = os.environ.get("FURIOSA_NPU_AUDIT_DIR")
if AUDIT_DIR:
    Path(AUDIT_DIR).mkdir(parents=True, exist_ok=True)
    os.environ["FURIOSA_GRAPH_AUDIT_PATH"] = str(Path(AUDIT_DIR) / "compiler_graphs.jsonl")

print("=" * 65)
print(f"[NPU 컴파일러 최적화] 전체 CPU 코어: {TOTAL_CORES}개")
print(f"[NPU 컴파일러 최적화] 컴파일 가속 스레드 수: {TARGET_THREADS}개 (코어 수 - 2 적용)")
print("=" * 65)

# 2. 허깅페이스 토큰 자동 로드
TOKEN_FILE = Path(__file__).parent / "hf_token.env"
HF_TOKEN = os.environ.get("HF_TOKEN")
if TOKEN_FILE.exists():
    with open(TOKEN_FILE) as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                HF_TOKEN = line.strip().split("=", 1)[1].strip("\"'")
                os.environ["HF_TOKEN"] = HF_TOKEN
if not HF_TOKEN:
    cached_token = Path("/root/.local/share/huggingface/token")
    if cached_token.exists():
        HF_TOKEN = cached_token.read_text().strip()
        os.environ["HF_TOKEN"] = HF_TOKEN

import torch
from compiler_engine.renegade_diffusion_compiler import (
    enable_npu_graphs,
    install_cpu_scheduler_execution,
    install_npu_scheduler_sigma_scale,
    keep_scheduler_latents_on_cpu,
)
from compiler_engine.execution_auditor import NPUExecutionAuditor, instrument_callable
from verify_color_diversity import analyze_image_color_diversity


def instrument_host_stages(pipe):
    instrument_callable(pipe, "encode_prompt", "host::prompt_encode")
    for index, tokenizer in enumerate((getattr(pipe, "tokenizer", None), getattr(pipe, "tokenizer_2", None)), 1):
        if tokenizer is not None:
            for method in ("_call_one", "batch_encode_plus"):
                instrument_callable(tokenizer, method, f"host::tokenizer_{index}::{method}")
    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is not None:
        for method in ("set_timesteps", "scale_model_input", "step"):
            instrument_callable(scheduler, method, f"host::scheduler::{method}")
    instrument_callable(pipe, "prepare_latents", "host::data::prepare_latents")
    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "decode"):
        instrument_callable(vae, "decode", "model_component::vae.decode")


def run_profiled_pipeline(pipe, model_key, **kwargs):
    if not AUDIT_DIR or getattr(pipe, "_initial_audit_complete", False):
        return pipe(**kwargs)
    instrument_host_stages(pipe)
    auditor = NPUExecutionAuditor(AUDIT_DIR, model_key)
    import json
    audit_start = time.perf_counter()
    with auditor:
        forward_start = time.perf_counter()
        with auditor.span(f"pipeline_inference::{model_key}"):
            result = pipe(**kwargs)
        forward_s = time.perf_counter() - forward_start
        teardown_start = time.perf_counter()
    times = {"first_profiled_forward_s": forward_s,
             "audit_finalize_s": time.perf_counter() - teardown_start,
             "audit_total_s": time.perf_counter() - audit_start,
             "warm_unprofiled_forward_s": []}
    path = Path(AUDIT_DIR) / "pipeline_timings.json"
    path.write_text(json.dumps(times, indent=2))
    for repeat in range(int(os.environ.get("FURIOSA_WARM_REPEATS", "0"))):
        generator = kwargs.get("generator")
        if generator is not None:
            generator.manual_seed(generator.initial_seed())
        started = time.perf_counter()
        if repeat == 0 and os.environ.get("FURIOSA_CONCRETE_CPU_AUDIT") == "1":
            from compiler_engine.cpu_tensor_audit import ConcreteCPUAudit
            probe = ConcreteCPUAudit(Path(AUDIT_DIR) / "warm_concrete_cpu.json")
            try:
                with probe:
                    result = pipe(**kwargs)
            finally:
                probe.save()
            times["diagnostic_warm_forward_s"] = time.perf_counter() - started
            started = time.perf_counter()
            result = pipe(**kwargs)
        else:
            result = pipe(**kwargs)
        times["warm_unprofiled_forward_s"].append(time.perf_counter() - started)
        result.images[0].save(Path(AUDIT_DIR) / f"warm_{repeat+1}.png")
        path.write_text(json.dumps(times, indent=2))
    pipe._initial_audit_complete = True
    return result


_SDXL_PIPELINE = None


def load_sdxl_pipeline():
    global _SDXL_PIPELINE
    if _SDXL_PIPELINE is None:
        model_id = "stabilityai/sdxl-turbo"
        from diffusers import AutoPipelineForText2Image
        print(f"1. SDXL-Turbo 파이프라인 로드: {model_id}")
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            variant="fp16",
            token=HF_TOKEN
        )
        pipe.to("cpu")

        # Furiosa compiler backend 연결
        enable_npu_graphs(pipe, "sdxl")

        if os.environ.get("FURIOSA_REQUIRE_NPU") == "1":
            install_npu_scheduler_sigma_scale(pipe, "sdxl")
        else:
            keep_scheduler_latents_on_cpu(pipe)
        install_cpu_scheduler_execution(pipe)

        _SDXL_PIPELINE = pipe
        from compiler_engine.memory_diagnostics import inspect_memory
        memory_path = Path(AUDIT_DIR or "reports/rngd_porting") / "model_memory.json"
        pipe._memory_diagnostics = inspect_memory(pipe, memory_path, probe=True)
        print("[메모리 진단] " + str(memory_path.resolve()), flush=True)
    return _SDXL_PIPELINE


def run_pipeline(
    model_type: str,
    prompt: str,
    output_path: str,
    steps: int,
    seed: int,
):
    from compiler_engine.work_metrics import snapshot, difference, record
    metrics_before = snapshot()
    request_started = time.perf_counter()
    print("\n" + "=" * 65)
    print(f"이미지 생성 작업 시작 (모델 타입: {model_type.upper()})")
    print("=" * 65)

    model_key = model_type.lower()

    if model_key in ["sdxl", "sdxl-turbo"]:
        load_started = time.perf_counter()
        pipeline_reused = _SDXL_PIPELINE is not None
        pipe = load_sdxl_pipeline()
        record("pipeline_lookup" if pipeline_reused else "model_load_and_adapter_setup", time.perf_counter()-load_started)

        print("\n2. SDXL 이미지 생성 추론 실행...")
        generator = torch.manual_seed(seed)
        start_t = time.time()
        result = run_profiled_pipeline(
            pipe,
            model_key,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        )
        image = result.images[0]
        gen_time = time.time() - start_t
        print(f"   추론 완료 소요 시간: {gen_time:.2f}초")

    elif model_key in ["flux", "flux.1-dev", "flux-dev"]:
        raise ValueError("FLUX.1-dev Transformer is approximately 12B and exceeds the 8B project limit. Select a separately verified <=8B FLUX variant; none is enabled yet.")
    elif model_key in ["flux3-klein", "flux3", "klein"]:
        model_id = "black-forest-labs/flux-3-image-klein"
        from diffusers import Flux2KleinPipeline
        print(f"1. FLUX 3 Image Klein 파이프라인 로드: {model_id}")
        try:
            pipe = Flux2KleinPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                token=HF_TOKEN
            )
        except Exception as e:
            if "gated repo" in str(e).lower() or "403" in str(e) or "401" in str(e):
                print("\n[오류] FLUX 3 Image Klein은 Hugging Face 게이트(Gated) 모델입니다.")
                print("라이선스 동의 페이지: https://huggingface.co/black-forest-labs/flux-3-image-klein")
                print("웹 브라우저에서 'Acknowledge license' 버튼을 누른 후 다시 시도해 주세요.\n")
            raise e

        pipe.to("cpu")

        # FLUX Klein 모델 그래프에 Furiosa compiler backend 연결
        enable_npu_graphs(pipe, model_key)

        print(f"\n2. FLUX 3 Image Klein 이미지 생성 추론 실행 (Steps: {steps})...")
        generator = torch.manual_seed(seed)
        start_t = time.time()
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=4.0,
            generator=generator,
        )
        image = result.images[0]
        gen_time = time.time() - start_t
        print(f"   추론 완료 소요 시간: {gen_time:.2f}초")

    elif model_key in ["qwen", "qwen-image", "qwen-image-2.1"]:
        model_id = "Qwen/Qwen-Image-2.1"
        from diffusers import QwenImage21Pipeline
        print(f"1. Qwen-Image-2.1 파이프라인 로드: {model_id}")
        pipe = QwenImage21Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN
        )
        pipe.to("cpu")

        # Qwen 모델 그래프에 Furiosa compiler backend 연결
        enable_npu_graphs(pipe, model_key)

        # 컴파일 그래프 입력 길이를 고정하기 위한 prompt padding
        orig_get_embeds = pipe._get_qwen_prompt_embeds
        def padded_get_embeds(prompt=None, image=None, device=None):
            orig_proc = pipe.processor
            def proc_wrapper(**kwargs):
                kwargs['padding'] = 'max_length'
                kwargs['max_length'] = 128
                return orig_proc(**kwargs)
            pipe.processor = proc_wrapper
            try:
                return orig_get_embeds(prompt=prompt, image=image, device=device)
            finally:
                pipe.processor = orig_proc

        pipe._get_qwen_prompt_embeds = padded_get_embeds

        print(f"\n2. Qwen-Image-2.1 이미지 생성 추론 실행 (Steps: {steps})...")
        generator = torch.manual_seed(seed)
        start_t = time.time()
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            true_cfg_scale=1.0,
            generator=generator,
        )
        image = result.images[0]
        gen_time = time.time() - start_t
        print(f"   추론 완료 소요 시간: {gen_time:.2f}초")

    else:
        raise ValueError(
            f"지원하지 않는 모델 타입입니다: {model_type}\n"
            f"선택 가능 모델: sdxl, flux, flux3-klein, qwen"
        )

    # 결과 이미지 저장
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    image.save(out_file)
    record("image_save", time.perf_counter()-save_started)
    print(f"\n이미지 저장 완료: {out_file.resolve()}\n")

    # 3. 색상 다양성 자동 검증
    check_started = time.perf_counter()
    analysis = analyze_image_color_diversity(str(out_file.resolve()), verbose=True)
    record("image_sanity_check", time.perf_counter()-check_started)
    metrics = difference(metrics_before)
    metrics["request_wall_s"] = round(time.perf_counter()-request_started, 6)
    metrics["inference_call_wall_s"] = round(gen_time, 6)
    metrics["pipeline_reused"] = pipeline_reused if model_key in ("sdxl", "sdxl-turbo") else False
    metrics["image_path"] = str(out_file.resolve())
    if model_key in ("sdxl", "sdxl-turbo"):
        from compiler_engine.memory_diagnostics import inspect_memory
        metrics["memory"] = inspect_memory(pipe, out_file.with_suffix(".memory.json"))
        metrics["memory"]["non_tcl_control"] = pipe._memory_diagnostics.get("non_tcl_control")
    if AUDIT_DIR and (Path(AUDIT_DIR)/"pipeline_timings.json").exists() and not metrics["pipeline_reused"]:
        import json
        metrics["initial_audit"] = json.loads((Path(AUDIT_DIR)/"pipeline_timings.json").read_text())
    ledger = Path(__file__).parent/"reports/rngd_porting/TCL_DEVELOPMENT_LEDGER.json"
    if ledger.exists():
        import json
        metrics["tcl_development_history"] = json.loads(ledger.read_text())
    import json
    metrics_path = out_file.with_suffix(".metrics.json")
    metrics["metrics_path"] = str(metrics_path.resolve())
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("[작업별 결과] " + json.dumps(metrics, ensure_ascii=False), flush=True)
    print(f"이미지: {out_file.resolve()}\n작업 보고서: {metrics_path.resolve()}", flush=True)
    analysis["work_metrics"] = metrics
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Furiosa Renegade NPU 멀티코어 컴파일 가속 다중 모델 이미지 생성기")
    parser.add_argument(
        "--model",
        choices=["sdxl", "flux", "flux3-klein", "qwen"],
        default="sdxl",
        help="사용할 모델 (sdxl, flux, flux3-klein, qwen / 기본: sdxl)"
    )
    parser.add_argument(
        "--prompt",
        default="A breathtaking cosmic dragon soaring through an aurora borealis nebula, vibrant scales, iridescent glowing light, hyper-detailed fantasy art 8k",
        help="생성할 이미지 프롬프트"
    )
    parser.add_argument(
        "-o", "--output",
        default="npu_output.png",
        help="출력 이미지 파일 경로"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="추론 스텝 (기본값: sdxl=1, flux=4, flux3-klein=16, qwen=20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드"
    )

    args = parser.parse_args()
    
    # 모델별 기본 추론 스텝 설정
    default_steps = {
        "sdxl": 1,
        "flux": 4,
        "flux3-klein": 16,
        "qwen": 20,
    }
    steps = args.steps if args.steps is not None else default_steps.get(args.model, 4)

    run_pipeline(
        model_type=args.model,
        prompt=args.prompt,
        output_path=args.output,
        steps=steps,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
