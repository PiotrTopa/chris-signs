#!/usr/bin/env python3
"""Export an Ultralytics .pt to a static-shape ONNX (opset 17, no NMS, fp32, batch 1) and write its
class list (line i = class id i) next to it.

usage: export_onnx.py SRC.pt DST.onnx H W CLASSES.txt
Needs ultralytics, torch, onnx, onnxslim (scripts/fetch-models.sh installs them in a venv).
"""
import os
import shutil
import sys
import tempfile

from ultralytics import YOLO

src, dst, h, w, names_out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(dst)))
try:
    p = os.path.join(tmp, 'm.pt')
    shutil.copy(src, p)  # Ultralytics writes the export next to the .pt
    m = YOLO(p)
    imgsz = (h, w) if h != w else h
    out = m.export(format='onnx', imgsz=imgsz, opset=17, simplify=True, dynamic=False, nms=False,
                   half=False, batch=1)
    shutil.move(out, dst)
    names = m.names
    with open(names_out, 'w') as f:
        f.write(''.join(names[i] + '\n' for i in sorted(names)))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print('OK', dst, names_out, len(names), 'classes')
