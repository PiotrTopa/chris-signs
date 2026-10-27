#!/bin/sh
# fetch_gtsrb.sh DIR: GTSRB (Stallkamp et al., IJCNN 2011; Zenodo record 13741936, record licence CC BY 4.0;
# the original site asks for a citation of the paper) into DIR/data. Positives for the verifier.
set -eu
d=${1:?usage: fetch_gtsrb.sh DIR}
mkdir -p "$d"
curl -fL -o "$d/data.zip" "https://zenodo.org/records/13741936/files/data.zip?download=1"
echo "7faafd0d78d831c7eb4d555641a83a62  $d/data.zip" | md5sum -c -
unzip -q -o "$d/data.zip" -d "$d/data"
