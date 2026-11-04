# YOLO11n RNGD 최적화 3차 — 2026-11-03

최신 Forge `38034fc4965ed7ec3684fafca923446d40664bd0`를 확인하고 적용했습니다. 서버는 같은 커밋이며 직접 GitHub pull은 인증 오류로 실패해 로컬의 성공한 pull과 대조했습니다. SDK2026.4.0, torch2.10.0+cpu, FXBEX0.4.0, RNGD fusion8 기준입니다. 원본과 이전 패키지는 보존했습니다.

## 결과와 측정 범위

| 경로 | 이전 | 이번 |
|---|---:|---:|
| 모델 직접 실행, NPU resident 입력→CPU 출력 완료 | 약285ms | 중앙값28.53ms |
| 독립 FXBEX 반복 모델 추론·출력 전송 | 약435ms | 중앙값30.40ms |
| 독립 FXBEX 이미지 요청 전체 | 약522ms | 중앙값74.30ms |

모델100회 반복: 평균28.85ms, 평균 처리율34.67FPS, p95 30.19ms, 최대38.83ms. 100회 중2회는33.33ms를 넘었습니다. 평균과 p95는30FPS 프레임 예산 안이지만 모든 프레임의deadline을 보장하지 않습니다. 입력 획득/전처리, NMS/그리기/파일저장, 네트워크는 직접 모델 수치에서 제외됩니다. 영상 스트리밍30FPS를 검증한 결과는 아닙니다.

독립 패키지 첫 요청은 약24.8초였습니다(준비 및 최초 감사 포함). 이후 요청은72–76ms입니다. 웹은 이미지 파일 요청/결과 표시 데모입니다. 실제 Gradio 웹 반복 요청은 모델30–32ms, 전체83–94ms였습니다.

## 결정적 병목과 적용 변경

1. **레이아웃 재배치**: 이전 공식 추적에서 contiguous Task union 약250ms. 1×1 conv를 pack→Linear→NCHW로 실행하던 경로를 직접 공개 SDK conv로 대체했습니다. 출력80채널은128로0패딩 후crop; DFL 특수 형상은 기존 검증 경로 유지. 모델285→95ms.
2. **첫 레이어 출력 변환**: 산술을 그대로 두고 reshape/permute/contiguous만 CompileModule로 실행. 해당 레이어 약42→3.5ms, bitwise 동일. 전체95→55ms.
3. **활성화 호출 융합**: 정확한 Conv→Sigmoid→Mul 패턴76개를 합쳐152개 eager 호출 제거. 전체55→약29ms. 공식 모델 창에서 Runnable 호출400→129, NPU Task800→258.
4. **웹 입력 stride**: 같은 픽셀/출력을 가진 strided 입력433–440ms vs contiguous NCHW285–291ms를 교차 반복으로 확인. 전처리에서 NCHW 연속 배열을 생성해 불필요한 NPU 재배치 제거.
5. **PNG 저장**: lossless compression level6→1, 약69→24ms. 디코딩 픽셀은 완전히 같고 파일은504→522KB 정도로 증가.

남은 병목: 최종 공식 프로파일에서129회 실행 호출,27회cat, 장치 내부cat 복사66회/union5.20ms, 전체NPU Task union7.03ms. 이 구간들은 서로 겹칠 수 있어 합산하면 안 됩니다. 프로파일 창82.60ms에는 추적 오버헤드가 있으므로 반복28.53ms와 혼동하면 안 됩니다.

60FPS에는16.67ms,120FPS에는8.33ms가 필요합니다. 현재60/120FPS는 달성하지 못했습니다. 다음 후보는 C2f/헤드 단위의 더 큰 그래프 융합, 레이아웃 유지와 persistent buffer 재사용, 비동기 프레임 파이프라인입니다. batch throughput과batch1 latency를 별도로 측정해야 합니다. 현재 자료로 하드웨어의 최대FPS를 단정하지 않습니다.

## 정확도 및 NPU 실행

CPU 기준 좌표 rtol1e-3/atol0.1, 점수 rtol1e-2/atol0.002를 변경 없이 통과했습니다. 좌표 최대 절대 오차0.02651978, 점수0.0000591278. random reference도 통과했습니다. FP32 입출력이며 공간 합성곱은 기존과 같은 보정 BF16 세 곱+FP32 합산입니다. 원본 NPU 출력과 bitwise 동일하지 않습니다. 전체 데이터셋 mAP는 측정하지 않았습니다.

89/89 그래프 공식NPU trace 확인, unknown0, CPU 모델 그래프0, SDK CPU fallback 비활성. 긴 최초 컴파일 후 첫 audit는 invalid trace reference time으로14개 그래프가unknown이라 거절했고, fresh-process 재검증을 통과했습니다. 첫 Linear+layout 동시 융합은 bitwise 기준 실패로 제외하고 layout-only 보정만 채택했습니다. 실패 기록은 원격runs에 남아 있습니다.

## 단일 파일 배포

파일: `fxbex/yolo11n-fp32comp-opt3-640-fusion8-sdk2026.4.0-portable.fxbex`

크기: 26,581,120 bytes. SHA256: `b508dbc0eb905b0514398e37e10ed1b2aeb8a52dc5ad5ce21984f8c2804af3ac`.

원본 ONNX 전체 가중치,90개 저장 Runnable, 모델 어댑터/필수 Forge Python 코드, 클래스/정적 메타데이터/모델 카드가 포함됩니다. SDK/드라이버 자체는 포함하지 않습니다. 대상에 호환 RNGD 드라이버, SDK2026.4.0, FXBEX0.4.0, ONNX1.17, onnx2torch1.5.15, PyTorch/torchvision, numpy/Pillow가 필요합니다.

`/tmp/yolo11_delivery_opt3`로 파일을 복사하고 Linux Landlock으로 원본 모델·프로젝트·Forge 코드·HF 캐시 읽기를 차단한 상태에서4회 실제 NPU 실행에 성공했습니다. compile_calls0, replay_hits91. 독립 출력은 검증된 통합 출력과 bitwise 동일하며89/89그래프가 확인됐습니다. package manifest의 relocation 상태는 생성 당시pending이고, 이후 통과 증거는 `isolated_result.json`과 이 보고서에 기록됩니다.

```python
from fxbex import YOLO
with YOLO("yolo11n-fp32comp-opt3-640-fusion8-sdk2026.4.0-portable.fxbex") as model:
    result = model.predict("image.jpg")
```

웹: http://127.0.0.1:7863/ . 원격 작업 폴더: `/root/yolo11-rngd-opt3`.

후속 Forge 개선 후보는 `FORGE_IDEAS.md`에 증거/가설/미검증 범위를 구분해 기록했습니다. Forge 공통 런타임에 추가 변경이나push는 하지 않았습니다. 검증 자료는 `final_summary.json`, `final_timing.json`, `final_trace_breakdown.json`, `activation_audit_summary.json`, `isolated_result.json`에 있습니다.
