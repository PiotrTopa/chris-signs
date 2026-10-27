#!/bin/sh
# Convert the two sign models (fp32 ONNX from scripts/fetch-models.sh) into the ncnn models used by
# `signd --backend ncnn`.
#   usage: tools/ncnn/make_ncnn.sh MODELS_DIR [PYTHON_VENV]
#   reads  MODELS_DIR/mtsd_signdet_yolov8n_384x640.onnx, MODELS_DIR/mtsd_signcls_yolov8n_64.onnx
#   writes MODELS_DIR/mtsd_signdet_yolov8n_384x640-ncnn.{param,bin}, MODELS_DIR/mtsd_signcls_yolov8n_64-ncnn.{param,bin}
#   and, if MODELS_DIR/bdd100k_yolo11n_320x960.onnx exists, bdd100k_yolo11n_320x960-ncnn.{param,bin} (lights)
# Steps per model: onnxsim -> pnnx (ONNX -> ncnn, fp32 weights) -> tools/ncnn/fix_ncnn.py (detector
# only: split the final Concat, see there). Needs onnx, onnxsim, pnnx (+ torch) in PYTHON_VENV
# (default: roadsense's .venv-models next to this checkout, created by roadsense scripts/fetch-models.sh;
# tested with onnx 1.23.2, onnxsim 0.8.1, pnnx 20261020). The models carry the licence caveats of
# docs/SIGN_MODELS.md: never commit or re-host the outputs.
set -eu
[ $# -ge 1 ] || { sed -n '2,10p' "$0"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
M=$(realpath "$1")
VENV=$(realpath "${2:-$HERE/../../../roadsense/.venv-models}")
W=$(mktemp -d "${TMPDIR:-/tmp}/signs-ncnn.XXXXXX")
trap 'rm -rf "$W"' EXIT
conv() { # stem inputshape fix(det|cls)
    cp "$M/$1.onnx" "$W/m.onnx"
    (cd "$W" && "$VENV/bin/python" -I -m onnxsim m.onnx m-sim.onnx >/dev/null &&
        "$VENV/bin/pnnx" m-sim.onnx inputshape="$2" fp16=0 >pnnx.log 2>&1) || { tail -20 "$W/pnnx.log"; exit 1; }
    "$VENV/bin/python" -I "$HERE/fix_ncnn.py" "$3" "$W/m_sim.ncnn.param" "$W/m_sim.ncnn.bin" "$M/$1-ncnn.param" "$M/$1-ncnn.bin"
}
conv mtsd_signdet_yolov8n_384x640 "[1,3,384,640]" det
conv mtsd_signcls_yolov8n_64 "[1,3,64,64]" cls
# traffic lights (signd --lights): same YOLO head split (boxes out0, 10 class scores out1); its two MatMuls
# (C2PSA attention) have no Vulkan kernel in ncnn and run on the CPU between GPU submits
[ ! -f "$M/bdd100k_yolo11n_320x960.onnx" ] || conv bdd100k_yolo11n_320x960 "[1,3,320,960]" det
sha256sum "$M"/*-ncnn.param "$M"/*-ncnn.bin
