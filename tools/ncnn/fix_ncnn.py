#!/usr/bin/env python3
# Called by tools/ncnn/make_ncnn.sh: fix_ncnn.py det|cls IN.param IN.bin OUT.param OUT.bin
#
# det: pnnx converts the YOLOv8 head faithfully, but ncnn's output of its final layer
#      Concat(boxes [4,5040], scores [1,5040], axis 0) -> out0 [5,5040] is wrong (CPU and Vulkan,
#      ncnn 20261020: integer-looking values, scores of 1.0 at wrong anchors) while both inputs
#      are right (boxes within 7e-4 px of ORT in fp32; a stand-alone Concat of these shapes works,
#      so it is some interaction inside this graph). We drop that Concat and expose the two inputs
#      as out0 (boxes, cx cy w h in input pixels) and out1 (scores after sigmoid); the ncnn runner
#      concatenates them into the ONNX [5,5040] layout.
# cls: no changes (softmax over 400 classes comes out as out0).
# Both: fp16 range checked 2026-10-24 (largest activation 638 in the detector, 12 in the classifier),
#       so no rescaling is needed.
import sys, shutil
which, pin, bin_, pout, bout = sys.argv[1:6]
L = open(pin).read().splitlines()
if which == 'det':
    outs = None
    for l in L:
        f = l.split()
        if f and f[0] == 'Concat' and f[-2:-1] == ['out0'] and f[2:4] == ['2', '1']:
            outs = f[4:6]
    assert outs, 'final Concat producing out0 not found'
    boxes, scores = outs
    o = []
    for l in L:
        f = l.split()
        if f and f[0] == 'Concat' and f[-2:-1] == ['out0'] and f[2:4] == ['2', '1']:
            continue
        if len(f) >= 4 and f[2].isdigit() and f[3].isdigit():  # a layer line: rename its output blobs
            ni, no = int(f[2]), int(f[3])
            outs_ = f[4 + ni:4 + ni + no]
            if boxes in outs_ or scores in outs_:
                new = ['out0' if b == boxes else 'out1' if b == scores else b for b in outs_]
                l = '%-24s %-24s ' % (f[0], f[1]) + ' '.join(f[2:4 + ni] + new + f[4 + ni + no:])
        o.append(l)
    nl, nb = map(int, o[1].split())
    o[1] = '%d %d' % (nl - 1, nb - 1)
    open(pout, 'w').write('\n'.join(o) + '\n')
else:
    shutil.copy(pin, pout)
shutil.copy(bin_, bout)
print('ok', which, pout, bout)
