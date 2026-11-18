#!/usr/bin/env python3
"""Table of the human-label results (L11 score.py, pooled over the rides) of sign variants evaluated by
eval_variant.sh. usage: compare_rules.py EVAL_DIR NAME[:PHASES] ...   e.g. r5:0,2 b1e2:0,2 r10:0
Totals are summed over the listed phases and divided by their number (per pass of the six rides)."""
import json
import sys

d = sys.argv[1]
cols = []
for spec in sys.argv[2:]:
    name, _, ph = spec.partition(":")
    phases = (ph or "0").split(",")
    S = [json.load(open(f"{d}/{name}/score_p{p}.json"))["pooled"] for p in phases]
    n = len(S)
    r = {}
    r["false rules"] = sum(sum(x["false"] for x in s["rules"].values()) for s in S) / n
    r["missed intervals"] = sum(sum(x["misses"] for x in s["rules"].values()) for s in S) / n
    r["matched intervals"] = sum(sum(x["matched"] for x in s["rules"].values()) for s in S) / n
    for k in ("momentary", "warnings"):
        r[f"{k} tp / engine / human"] = "%.1f / %.1f / %d" % (
            sum(sum(x["tp"] for x in s[k].values()) for s in S) / n,
            sum(sum(x["engine"] for x in s[k].values()) for s in S) / n,
            sum(x["human"] for x in S[0][k].values()))
    r["interval start delay median, s"] = "/".join(str(s["delays"]["start_median"]) for s in S)
    rules = sorted(S[0]["rules"])
    for k in rules:
        a = sum(s["rules"][k]["agreement_mean"] for s in S) / n
        f = sum(s["rules"][k]["false"] for s in S) / n
        m = sum(s["rules"][k]["misses"] for s in S) / n
        r[f"agreement {k} (F, M)"] = f"{a:.3f} (F{f:g} M{m:g})"
    cols.append((spec, r))
keys = list(cols[0][1])
print("| metric (per pass of the rides) | " + " | ".join(c for c, _ in cols) + " |")
print("|---" * (len(cols) + 1) + "|")
for k in keys:
    print(f"| {k} | " + " | ".join((f"{r[k]:g}" if isinstance(r.get(k), float) else str(r.get(k, "-"))) for _, r in cols) + " |")
