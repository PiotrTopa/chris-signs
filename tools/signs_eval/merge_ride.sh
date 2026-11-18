#!/bin/bash
# Merge the cache_dets.py parts of a ride (run_cache layout) into CACHE/<slug>.<view>.{npz,dets}
# usage: merge_ride.sh SLUG [PARTS_DIR] [CACHE_DIR]
S=$1; P=${2:-/home/piotr/datasets/l24s/parts}; C=${3:-/home/piotr/datasets/l24s/cache}
mkdir -p $C
for v in A T3 B1 LW LZ; do
  parts=$(ls -d $P/$S.p*.d | sort | while read d; do [ -s $d/$S.$v.npz ] && echo $d/$S.$v; done)
  /home/piotr/l12/.venv/bin/python -I $(dirname $0)/merge_parts.py $C/$S.$v $parts
done
echo "$S merged"
