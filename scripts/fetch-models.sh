#!/bin/sh
# Download the pinned MTSD sign models (.pt, Hugging Face jkot/pova-traffic-sign-recognition-models)
# and export them locally to the ONNX files and class lists signd expects, in ./models.
# Run on a dev box (glibc x86_64/aarch64 with Python 3, ~2 GB for torch in the venv), not on the
# phone (no torch/ultralytics wheels for musl). Recipe and rationale: docs/SIGN_MODELS.md.
#
# The weights carry no licence (all rights reserved) and an Ultralytics AGPL-3.0 claim, and were
# trained on MTSD (CC BY-NC-SA): private, non-commercial use only. Never commit or re-host the
# .pt or .onnx files (.gitignore keeps them out).
#
#   scripts/fetch-models.sh            # download + scan + export (skips steps already done)
#   NCNN=0 scripts/fetch-models.sh     # skip the ncnn (Vulkan) conversion, see below
#   PYTHON=python3.12 scripts/fetch-models.sh
#   ULTRALYTICS_VERSION=... TORCH_VERSION=... scripts/fetch-models.sh   # override the tested pins
set -eu
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
VENV=${VENV:-.venv-export}
ULTRALYTICS_VERSION=${ULTRALYTICS_VERSION:-8.4.183}   # tested 2026-10-21
TORCH_VERSION=${TORCH_VERSION:-2.14.1}
HF=https://huggingface.co/jkot/pova-traffic-sign-recognition-models/resolve/0c135006544630d813603f48866de2f11287c518
mkdir -p models/src

fetch() { # url sha256 dest
    if [ -f "$3" ] && [ "$(sha256sum "$3" | cut -d' ' -f1)" = "$2" ]; then
        echo "ok   $3"; return 0
    fi
    echo "get  $3"
    curl -fsSL -o "$3.part" "$1"
    got=$(sha256sum "$3.part" | cut -d' ' -f1)
    [ "$got" = "$2" ] || { echo "sha256 mismatch for $3: $got"; rm -f "$3.part"; exit 1; }
    mv "$3.part" "$3"
}
fetch "$HF/binary_detector-nano-1280.pt" cf5fe8c5c2c3cb1ff77b8232778c4f95d8321ddeb2d9bad7928e03e3373db444 \
      models/src/binary_detector-nano-1280.pt
fetch "$HF/yolo_classifier_nano.pt" 9124dd54de24d4b0b22eb7d6524544b74cbda24c4258f75441c53a67a7a6fef4 \
      models/src/yolo_classifier_nano.pt
# Traffic lights (signd --lights, docs/LIGHT_MODELS.md): dronefreak/bdd100k-yolo11n, YOLO11n fine-tuned on
# BDD100K (10 classes, 8 = traffic light). Card: AGPL-3.0; trained on BDD100K (research / non-commercial terms):
# private use only, never commit or re-host. Skip with LIGHTS=0.
HF_TL=https://huggingface.co/dronefreak/bdd100k-yolo11n/resolve/2b5554133522f96f50ce87a4df0a2de556e4b7ad
[ "${LIGHTS:-1}" = 0 ] || fetch "$HF_TL/best.pt" c120d075bcc6aaf94523f9a2c9d681819ebd692f36eb5ea64656d1c5f1278a70 \
      models/src/bdd100k_yolo11n.pt

# The .pt files are pickles: list their globals without unpickling before anything loads them.
"$PYTHON" -I tools/scan_pt.py models/src/*.pt

DET=models/mtsd_signdet_yolov8n_384x640.onnx
CLS=models/mtsd_signcls_yolov8n_64.onnx
TL=models/bdd100k_yolo11n_320x960.onnx
NEED_TL=0
[ "${LIGHTS:-1}" = 0 ] || [ -f "$TL" ] || NEED_TL=1
if [ ! -f "$DET" ] || [ ! -f "$CLS" ] || [ ! -f models/mtsd_signdet_classes.txt ] || [ ! -f models/mtsd_signcls_classes.txt ] || [ $NEED_TL = 1 ]; then
    if [ ! -x "$VENV/bin/python" ]; then
        "$PYTHON" -m venv "$VENV"
        "$VENV/bin/pip" -q install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
        "$VENV/bin/pip" -q install "ultralytics==$ULTRALYTICS_VERSION" onnx onnxslim dill
    fi
    # Ultralytics writes settings/caches under $HOME; keep them inside the venv.
    export YOLO_CONFIG_DIR="$PWD/$VENV/ultralytics-config"
    "$VENV/bin/python" -I tools/export_onnx.py models/src/binary_detector-nano-1280.pt "$DET" 384 640 models/mtsd_signdet_classes.txt
    "$VENV/bin/python" -I tools/export_onnx.py models/src/yolo_classifier_nano.pt "$CLS" 64 64 models/mtsd_signcls_classes.txt
    # lights: 960x320 input = the upper 62 % of a 1280x720 frame at 0.72x (signd --lights-in/--lights-roi defaults)
    [ $NEED_TL = 0 ] || "$VENV/bin/python" -I tools/export_onnx.py models/src/bdd100k_yolo11n.pt "$TL" 320 960 models/bdd100k_classes.txt
fi

# Re-exports are not byte-identical across library versions: check I/O shapes, not sha256.
[ -x "$VENV/bin/python" ] && "$VENV/bin/python" -I - "$DET" "$CLS" <<'PY'
import sys, onnx
want = {sys.argv[1]: ([1, 3, 384, 640], [1, 5, 5040]), sys.argv[2]: ([1, 3, 64, 64], [1, 400])}
for path, (i, o) in want.items():
    g = onnx.load(path).graph
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]
    got = (dims(g.input[0]), dims(g.output[0]))
    print(("ok   " if got == (i, o) else "BAD  ") + path, got)
    if got != (i, o):
        sys.exit(1)
PY
[ "$(wc -l < models/mtsd_signcls_classes.txt)" = 400 ] || { echo "classifier class list is not 400 lines"; exit 1; }

# ncnn (Vulkan) models for `signd --backend ncnn`: onnxsim + pnnx + tools/ncnn/fix_ncnn.py, see
# tools/ncnn/make_ncnn.sh. pnnx links against torch; the conversion venv is roadsense's
# .venv-models (NCNN_VENV, default ../roadsense/.venv-models, created by roadsense scripts/fetch-models.sh;
# a fresh one needs: pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu,
# then onnx==1.23.2 onnxsim==0.8.1 pnnx==20261020 numpy). Skip with NCNN=0.
if [ "${NCNN:-1}" != 0 ] && { [ ! -f models/mtsd_signdet_yolov8n_384x640-ncnn.param ] || [ ! -f models/mtsd_signcls_yolov8n_64-ncnn.param ] || { [ -f "$TL" ] && [ ! -f models/bdd100k_yolo11n_320x960-ncnn.param ]; }; }; then
    NCNN_VENV=${NCNN_VENV:-../roadsense/.venv-models}
    if [ ! -x "$NCNN_VENV/bin/pnnx" ]; then
        "$PYTHON" -m venv "$NCNN_VENV"
        "$NCNN_VENV/bin/pip" -q install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
        "$NCNN_VENV/bin/pip" -q install onnx==1.23.2 onnxsim==0.8.1 pnnx==20261020 numpy
    fi
    tools/ncnn/make_ncnn.sh models "$NCNN_VENV"
fi
sha256sum models/*.onnx models/*.txt models/*-ncnn.param models/*-ncnn.bin 2>/dev/null
