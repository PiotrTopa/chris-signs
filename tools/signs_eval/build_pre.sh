#!/bin/sh
# Builds libsigns_pre.so (tools/signs_eval/pre.cc + signd's image code) for the Python evaluation scripts.
# usage: build_pre.sh OUT_DIR [ROADSENSE_DIR]   (needs libjpeg-dev and the ONNX Runtime dev files)
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?out dir}; RS=${2:-$R/../roadsense}
mkdir -p "$OUT"
c++ -std=c++17 -O3 -fPIC -shared -o "$OUT/libsigns_pre.so" "$R/tools/signs_eval/pre.cc" "$R/signd/yolo.cc" \
  "$R/signd/verify.cc" "$R/signd/ort_runner.cc" -I"$R/signd" -I"$RS/camerad" -I/usr/include/onnxruntime \
  -lonnxruntime -ljpeg
echo "$OUT/libsigns_pre.so"
