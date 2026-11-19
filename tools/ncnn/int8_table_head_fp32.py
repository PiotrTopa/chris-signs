#!/usr/bin/env python3
"""int8_table_head_fp32.py PARAM TABLE_IN TABLE_OUT : drop the int8 entries of the first conv and of the
detection head's 1x1 convs with 10, 64 or 1 output channels (among the last 16 convs), so ncnn2int8 leaves them
in fp32. Recipe of roadsense lane L26 for the light detectors (docs/lanes/L26-lights-yolov8n.md):

  ncnn2table M-ncnn.param M-ncnn.bin calib_list.txt M.table mean=[0,0,0] norm=[0.003922,0.003922,0.003922] \
      shape=[960,320,3] pixel=RGB thread=4 method=aciq
  int8_table_head_fp32.py M-ncnn.param M.table M.head.table
  ncnn2int8 M-ncnn.param M-ncnn.bin M-int8-ncnn.param M-int8-ncnn.bin M.head.table

BDD100K val (held-out split): kl on every conv loses 14-15 AP50 points, aciq 4-5, aciq with this fp32 head 0.5-1."""
import sys
param, tin, tout = sys.argv[1:4]
convs = []
for l in open(param).read().splitlines()[2:]:
    f = l.split()
    if f and f[0] == "Convolution":
        kv = dict(x.split("=", 1) for x in f[6:] if "=" in x)
        convs.append((f[1], kv.get("0"), kv.get("1")))
keep_fp = {convs[0][0]} | {n for n, o, k in convs[-16:] if k == "1" and o in ("10", "64", "1")}
print("fp32:", sorted(keep_fp))
out = [l for l in open(tin) if l.split()[0].rsplit("_param_0", 1)[0] not in keep_fp]
open(tout, "w").writelines(out)
