#!/bin/bash
# One sign variant on the six L11 rides: cached detections -> signd's tracker (variant.py) -> L11 recording ->
# roadsense rules engine (tools/rules_replay.py) -> L11 score.py; and the reference scorer (score_ref.py).
# usage: eval_variant.sh NAME PHASES "VARIANT.PY ARGS" [RIDES]     e.g. eval_variant.sh r5 "0 2" "--hz 5"
set -u
NAME=$1; PHASES=$2; VARGS=$3; RIDES=${4:-"krakow-2024 rural-zloczew forest-vistula-spit s1-roadworks warsaw-city katowice-rain"}
PY=/home/piotr/l12/.venv/bin/python
RPY=/home/piotr/projects/roadsense/.venv-models/bin/python
T=${T:-$(cd "$(dirname "$0")" && pwd)}
RS=${RS:-/home/piotr/projects/wt/roadsense-v2}
L11=$RS/docs/lanes/L11-signs-in-effect
O=${EVAL_DIR:-/home/piotr/datasets/l24s/eval}/$NAME; mkdir -p $O
for P in $PHASES; do
  for s in $RIDES; do
    $PY -I $T/variant.py $s $O/${s}_p$P.jsonl --phase $P $VARGS || exit 1
    $RPY -I $T/ride_rules.py $s /home/piotr/l11/rec/$s.jsonl $O/${s}_p$P.jsonl $O/rec_${s}_p$P.jsonl 2>> $O/rules.log
    $RPY $RS/tools/rules_replay.py $O/rec_${s}_p$P.jsonl --quiet --out $O/tl_${s}_p$P.json > /dev/null 2>> $O/rules.log
    rm -f $O/rec_${s}_p$P.jsonl
  done
  $RPY - $P $O $RIDES <<'PYEOF'
import json, sys
p, o, slugs = sys.argv[1], sys.argv[2], sys.argv[3:]
runs = []
for s in slugs: runs += json.load(open(f"{o}/tl_{s}_p{p}.json"))
json.dump(runs, open(f"{o}/tl_p{p}.json", "w"))
PYEOF
  $RPY $L11/score.py $O/tl_p$P.json $L11/labels --rides $(echo $RIDES | tr ' ' ',') --out $O/score_p$P.json > $O/score_p$P.txt 2>&1
done
echo "$NAME done $(date -Is)"
