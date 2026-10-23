#!/bin/sh
# Reference deployment (Poco F1 / postmarketOS, systemd): build and install signd next to an
# installed roadsense. Run ON THE PHONE as root from a checkout (e.g. ~/roadsense-signs), with a
# roadsense checkout at $ROADSENSE_DIR (default ../roadsense) for the IPC headers.
# Models: ./models from scripts/fetch-models.sh (run on a dev box, then copy models/ over).
# GPU backend: NCNN_DIR=/path/to/ncnn (source tree built in NCNN_DIR/build, roadsense
# scripts/build-ncnn.sh) builds `--backend ncnn` in and installs the *-ncnn.param/.bin models too.
set -eu
cd "$(dirname "$0")/.."
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)"; exit 1; }
ROADSENSE_DIR=${ROADSENSE_DIR:-$(cd .. && pwd)/roadsense}
BUILD=${BUILD:-build}
for m in mtsd_signdet_yolov8n_384x640.onnx mtsd_signcls_yolov8n_64.onnx mtsd_signdet_classes.txt mtsd_signcls_classes.txt; do
    [ -f "models/$m" ] || { echo "models/$m missing (scripts/fetch-models.sh on a dev box)"; exit 1; }
done
NCNN_ARGS=""
if [ -n "${NCNN_DIR:-}" ]; then
    for m in mtsd_signdet_yolov8n_384x640-ncnn.param mtsd_signdet_yolov8n_384x640-ncnn.bin mtsd_signcls_yolov8n_64-ncnn.param mtsd_signcls_yolov8n_64-ncnn.bin; do
        [ -f "models/$m" ] || { echo "models/$m missing (scripts/fetch-models.sh or tools/ncnn/make_ncnn.sh on a dev box)"; exit 1; }
    done
    NCNN_ARGS="-DROADSENSE_NCNN=ON -DNCNN_DIR=$NCNN_DIR"
fi
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DROADSENSE_DIR="$ROADSENSE_DIR" $NCNN_ARGS >/dev/null
cmake --build "$BUILD" -j"$(nproc)"
install -m 755 "$BUILD/signd" "$BUILD/signd_replay" /usr/local/bin/
mkdir -p /var/lib/roadsense/models /etc/roadsense
install -m 644 models/mtsd_*.onnx models/mtsd_*_classes.txt /var/lib/roadsense/models/
[ -z "${NCNN_DIR:-}" ] || install -m 644 models/mtsd_*-ncnn.param models/mtsd_*-ncnn.bin /var/lib/roadsense/models/
install -m 644 systemd/roadsense-signd.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable roadsense-signd.service >/dev/null
echo "installed. start:  systemctl start roadsense-signd (or roadsense.target)   status: systemctl status roadsense-signd"
