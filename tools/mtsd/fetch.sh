#!/bin/bash
# Resumable sequential fetch of MTSD parts by line number of a signed-URL file.
# Usage: fetch.sh URLFILE OUTDIR LINE [LINE...]   (URLs never echoed)
set -u
uf=$1; out=$2; shift 2
for i in "$@"; do
  u=$(sed -n "${i}p" "$uf")
  f="$out/part_$i.zip"
  for try in 1 2 3 4 5 6 7 8; do
    curl -sS -L --retry 5 --retry-delay 10 -C - -o "$f" "$u" 2>&1 | sed 's#https\?://[^ ]*#<url>#g'
    st=${PIPESTATUS[0]}
    [ "$st" = 0 ] && break
    echo "part_$i try $try exit $st"; sleep 20
  done
  echo "$(date -Is) part_$i done size=$(stat -c %s "$f")"
done
