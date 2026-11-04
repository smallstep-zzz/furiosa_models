import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


@unittest.skipUnless(os.environ.get("FURIOSA_RUN_NPU_TESTS") == "1",
                     "set FURIOSA_RUN_NPU_TESTS=1 on a Furiosa server")
class SdxlSchedulerSigmaTests(unittest.TestCase):
    def test_python_scalar_mul_matches_cpu_and_emits_npu_task(self):
        from furiosa.torch.profiler import Profiler
        from compiler_engine.renegade_diffusion_compiler import install_npu_scheduler_sigma_scale

        out = Path("reports/rngd_porting/runs/sdxl_sigma_mul_repro")
        out.mkdir(parents=True, exist_ok=True)
        trace_path = out / "furiosa_trace.json"
        graph_path = out / "compiler_graphs.jsonl"
        os.environ["FURIOSA_GRAPH_AUDIT_PATH"] = str(graph_path)
        cpu_latents = torch.linspace(-1, 1, 4 * 64 * 64, dtype=torch.float32).reshape(1, 4, 64, 64).to(torch.bfloat16)
        npu_latents = cpu_latents.to("furiosa:0")
        cpu_sigma = torch.tensor(0.5, dtype=torch.bfloat16)

        with self.assertRaisesRegex(RuntimeError, "same device"):
            npu_latents * cpu_sigma

        pipe = SimpleNamespace(
            scheduler=SimpleNamespace(init_noise_sigma=cpu_sigma),
            vae_scale_factor=8,
        )
        install_npu_scheduler_sigma_scale(pipe, "sdxl")
        with Profiler(str(trace_path)):
            npu_result = pipe.prepare_latents(
                1, 4, 512, 512, torch.bfloat16, "furiosa:0", None, latents=cpu_latents,
            )
        torch.testing.assert_close(npu_result.cpu(), cpu_latents * cpu_sigma.item())
        trace = json.loads(trace_path.read_text())
        tasks = [event for event in trace.get("traceEvents", [])
                 if "npu2::Task::" in str(event.get("name", ""))]
        (out / "summary.json").write_text(json.dumps({
            "input_shape": list(cpu_latents.shape),
            "dtype": str(cpu_latents.dtype),
            "cpu_tensor_scalar": float(cpu_sigma),
            "mixed_device_tensor_scalar_rejected": True,
            "python_scalar_output_device": str(npu_result.device),
            "cpu_reference_match": True,
            "npu_task_events": len(tasks),
        }, indent=2) + "\n")
        self.assertGreater(len(tasks), 0, "scalar multiply returned without an NPU Task event")

    def test_scheduler_boundary_keeps_sample_math_on_cpu(self):
        from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
        from compiler_engine.renegade_diffusion_compiler import install_cpu_scheduler_execution

        scheduler = EulerAncestralDiscreteScheduler(num_train_timesteps=1000)
        reference = EulerAncestralDiscreteScheduler(num_train_timesteps=1000)
        install_cpu_scheduler_execution(SimpleNamespace(scheduler=scheduler))
        scheduler.set_timesteps(1, device="furiosa:0")
        reference.set_timesteps(1, device="cpu")
        self.assertEqual(str(scheduler.timesteps.device), "cpu")

        sample = torch.linspace(-1, 1, 4 * 64 * 64).reshape(1, 4, 64, 64).to(torch.bfloat16)
        model_output = (sample * 0.25).to("furiosa:0")
        timestep = scheduler.timesteps[0]
        scaled = scheduler.scale_model_input(sample.to("furiosa:0"), timestep)
        reference_scaled = reference.scale_model_input(sample.clone(), reference.timesteps[0])
        self.assertEqual(scaled.device.type, "cpu")
        torch.testing.assert_close(scaled, reference_scaled)

        result = scheduler.step(model_output, timestep, scaled,
                                generator=torch.Generator(device="cpu").manual_seed(123))
        expected = reference.step(sample * 0.25, reference.timesteps[0], reference_scaled,
                                  generator=torch.Generator(device="cpu").manual_seed(123))
        self.assertEqual(result.prev_sample.device.type, "cpu")
        torch.testing.assert_close(result.prev_sample, expected.prev_sample)


if __name__ == "__main__":
    unittest.main()
