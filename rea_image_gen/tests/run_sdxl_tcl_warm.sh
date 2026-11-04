#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir=${1:-reports/rngd_porting/runs/sdxl_tcl_warm_r13}
mkdir -p "$run_dir"
env -u TORCH_DEVICE_BACKEND_AUTOLOAD \
 FURIOSA_REQUIRE_NPU=1 FURIOSA_SEGMENT_UNET=1 FURIOSA_SEGMENT_TEXT_ENCODERS=1 \
 FURIOSA_SEGMENT_UNET_OPERATORS=1 FURIOSA_SEGMENT_UNET_OPERATOR_BLOCKS=0,1,2,3 \
 FURIOSA_DYNAMO_RECOMPILE_LIMIT=512 \
 FURIOSA_SPATIAL_CONV_SIGNATURES=640x640x3x3,640x1280x3x3,1280x1280x3x3,320x640x1x1,640x640x1x1,640x1280x1x1,1280x1280x1x1 \
 FURIOSA_NPU_AUDIT_DIR="$run_dir" FURIOSA_WARM_REPEATS=3 \
 RAYON_NUM_THREADS=126 OMP_NUM_THREADS=8 \
 /usr/bin/python3.10 run_image_gen_npu.py --model sdxl --steps 1 --seed 20261102 \
 --prompt 'A small red fox sitting beside a mountain lake at sunrise, golden autumn trees, mist over the water, cinematic photography, detailed natural light' \
 --output "$run_dir/image.png" > "$run_dir/run.log" 2>&1
