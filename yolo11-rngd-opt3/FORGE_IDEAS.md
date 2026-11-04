# Forge에 반영할 후보 메모 — 2026-11-03

검증된 결과와 가설을 구분하며, 이 파일은 아직 공통 런타임 변경 제안입니다.

1. **비동기 대기와 연산 원인 분리**: YOLO 반복 host-node 측정은 Concat(50ms, 31ms)을 상위로 표시했지만, 동일 공식 추적의 Task 구간은 contiguous 재배치가 약250ms였다. host 호출의 대기 위치를 실제 연산 병목으로 오인할 수 있다. optimization_packet에 동일 창의 공식 Task 종류별 union 및 enqueue/producer scope 연결을 추가할 가치가 있다. raw: /root/yolo11-rngd-opt3/runs/remaining_1793706699485804809.
2. **레이아웃을 계약 키에 포함**: shape/dtype뿐 아니라 input stride, memory format, storage offset을 baseline/package 비교 보고서에 함께 남기자. 파일 전처리와 NPZ 참조는 같은 shape여도 stride가 다를 수 있다. 이 모델에서 영향은 아직 검증 전.
3. **직접 1×1 합성곱 우선 후보**: 공간 합성곱 실패 때문에 전체 Conv2d를 pack+Linear로 바꾸면 1×1까지 불필요한 NCHW↔matrix 변환을 상속한다. geometry별 지원 레지스트리와 직접경로 우선 검증이 필요하다. 현재 실제 모델 형상 suite 진행 중; 성공을 선결론으로 기록하지 않는다.
4. **FPS는 latency와 throughput 분리**: batch1 동기 완료 지연의 역수를 파이프라인/배치 처리량과 구분한다. 30/60/120 FPS 목표는33.3/16.7/8.3ms이며, 현재284.7ms baseline으로 달성을 약속할 수 없다.
5. **재현 가능한 패키지 A/B**: 직접 모델 vs 패키지 차이를 replay 비용으로 단정하지 말고 동일 입력 stride·출력 완료 경계·스레드·감사 모드에서 A/B 측정하도록 도구화. 기존434ms vs285ms 차이는 아직 원인 미확정.

6. **NPU 잠금은 프로세스 종료까지 유지**: feature runner가 `with flock`을 빠져나온 뒤 살아 있는 device tensor/runtime 정리 사이에 다음 프로세스가 진입하면 EBUSY가 OutOfMemoryError(2048 bytes)로 표면화될 수 있다. 실제 layout_ab 첫 시도에서 관찰. 큐는 프로세스 exit 및 SMI 해제를 확인하거나 부모가 child lifetime 전체 잠금을 소유해야 한다. 메모리 부족으로 오진하지 않도록 원인 예외를 보존하자.

검증 업데이트:
- 직접 1×1 전체 모델:95ms, 기존 수치 기준+random 통과,88/88 NPU 그래프. 출력80은128채널로0패딩 후crop; DFL1채널은 수치 실패로 제외.
- 첫 합성곱의 Linear+layout 통합은 bitwise 기준에서 최대7.6e-6 차이로 제외. layout만 CompileModule로 실행하면 bitwise 동일,42ms→3.5ms. 합친 모델55ms. trace graph_id는16hex만인식하므로 자유 문자열을쓰면unknown; ID 생성 API/검증기가 필요.
- 입력 stride A/B 확정: 같은 픽셀 값, 동일 모델에서web strided433–440ms vs NCHW contiguous285–291ms. 출력bitwise 동일. 약150ms 차이는 replay 비용이 아니라 입력 레이아웃으로 재현됨. raw:/root/yolo11-rngd-opt3/runs/layout_ab_1793707273521952967.

- 55ms 경로 공식 trace(프로파일 창134.7ms): NPU Task union8.48ms, D2D union5.60ms, D2H0.55ms. 각각 겹칠 수 있으므로 합산을 실제 lower bound로 주장하지 않는다. 프로파일 자체 비용이 커서 unprofiled latency와 구분한다. 단순 FLOPS/SMI 기준 추정 대신host dispatch/작업granularity를 우선 살펴야 한다.
- 웹 PNG lossless 저장 level6 약69ms → level1 약24ms, 파일504KB→522KB, 재디코딩 pixel exact. 모델 외 비용이 kernel 최적화 후 주요 병목으로 바뀌므로 추론/전처리/후처리/파일저장/전송 단계별 측정이 필요하다.

7. **Cold load와 warm latency를 별도 지표로**: FXBEX replay는 compile_calls=0이어도 Python 모델 재구성/export/lowering 준비가 수십 초 걸린다. 이것은 매 프레임 비용이 아니다. Forge는 준비 단계별 타이머를 보고하고, FXBEX 담당 저장소에서 검증된 ExportedProgram/실행 계획의 안전한 재사용 가능성을 별도로 검토할 가치가 있다. 아직 구현/성능 검증하지 않은 아이디어이며 SDK 캐시 바이너리를 임의 분석하거나 재패키징하자는 제안은 아니다.

8. **Source 변경과 감사 ID 변경을 분리해 기록**: 처음 layout_full_v1은 커널 수치/성능은 통과했지만 자유 문자열 graph ID 때문에89개 중1개가unknown이었다. 16hex ID로 바로잡은 후 fresh profile에서89/89 확인됐다. 통합 runner는 task_count>0만으로accept하지 말고 unknown_graphs=0과expected graph count를 강제해야 한다. 이 캠페인의 integration_pointwise.py에 반영했다.

9. **실시간 모드와 감사/이미지 데모 모드 분리**: 모델이 수십ms가 되면 매 요청 전체raw tensor의.npy 저장과PNG 렌더링이 전체 요청을 지배한다. 향후 별도 streaming API에서 디버그 산출물 선택 저장, 버퍼 재사용, 프레임 큐 및 backpressure를 검증할 가치가 있다. 감사용 최초 출력/trace와 정확도 검증은 유지해야 한다. 현재 구현은 이미지 파일 요청이며 영상30/60/120FPS를 검증한 것은 아니다.

10. **FPS 목표의 다음 단계는 블록/실행 계획 단위**: 개별conv+activation 융합 이후에도 수십개의Python→SDK 호출이 남는다. 향후 C2f의split/conv/add/cat 또는 검출헤드를 하나의 공개 CompileModule로 만드는 bounded 실험, persistent buffer 재사용, 비동기 pipeline을 순서대로 검토할 수 있다. 60FPS(16.7ms)/120FPS(8.3ms)는 배치처리량과batch1 latency를 별도 명시하고 검증해야 한다. 현존SDK의 layout/DFL 예외와 실제 수치 기준을 그대로 유지해야 하며, 전체 모델을 무작정 export하거나 INT8로 변경해서 목표를 달성했다고 해서는 안 된다.

최종 검증 업데이트:
- conv+SiLU 76개 융합,152개 eager 호출 제거.100회 resident input/CPU output 완료 경계에서 median28.53ms,p95 30.19ms,mean28.85ms(34.67FPS),max38.83ms.100회 중2회33.3ms 초과. 영상 실시간성 보장은 아님.
- 공식 동일 창에서129 Runnable calls,258 model NPU Tasks,Task union7.03ms,Concat D2D5.20ms. 프로파일 창82.60ms는 오버헤드가 포함되므로 warm latency로 사용하지 않는다.
- 긴 최초 컴파일 뒤SDK invalid trace reference time으로14개그래프가unknown: acceptance거절. 캐시 준비된 새프로세스재검증에서89/89통과. 원인은SDK추적시각문제로그가근거이며 내부clock원인은확정하지않음. 원본실패trace를보존하고 재검증을명시하자.
- bundled Python 복사 시 macOS AppleDouble `._*.py`를 제외하자. 소스구문검사에서발견된OS메타데이터이며 실행코드가아니다.
