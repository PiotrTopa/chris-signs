#!/usr/bin/env python3
"""make_fixture.py OUT.onnx: a stand-in for a pose head with the "pose1" contract (signd/pose.h), for the tests.
No training and no data: the graph is the mean of each input plane followed by a fixed linear map written out
below, so every output is a known function of what signd feeds (the crop's mean colour, the box numbers).
tools/pose_test.cc holds the same map."""
import sys
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

W = np.zeros((12, 38), np.float32); B = np.zeros(38, np.float32)
W[0, 0] = 100.0                       # yaw = 100 x mean red
B[1], B[2], B[3] = 5.0, 0.9, 0.05     # yaw sigma, p_main, p_flip
for k in range(4): W[4 + k, 4 + k] = 1.0   # view probabilities = the four cut flags (planes 4-7)
W[3, 8] = 1.0                         # gc x = ln(w / h)
W[9, 9] = 1.0                         # gc y = (cx - cu) / f
W[10, 10] = 1.0                       # gn y = (bottom - cv) / f
W[11, 11] = 1.0                       # gl x = ln(h / f) + 3
W[8, 12] = 1.0                        # gl y = camera given
W[1, 13] = 1.0                        # gr x = mean green
W[2, 14] = 1.0                        # gr y = mean blue
B[15:22] = 0.1; B[22:26] = 0.5
B[26:29] = (4.0, 2.0, 1.5); B[29:32] = 0.1; B[32:35] = 0.2; B[35] = 1.0
g = helper.make_graph(
    [helper.make_node('ReduceMean', ['in'], ['m'], axes=[2, 3], keepdims=0),
     helper.make_node('Gemm', ['m', 'W', 'B'], ['pose'])],
    'pose_fixture',
    [helper.make_tensor_value_info('in', TensorProto.FLOAT, [1, 12, 64, 64])],
    [helper.make_tensor_value_info('pose', TensorProto.FLOAT, [1, 38])],
    [numpy_helper.from_array(W, 'W'), numpy_helper.from_array(B, 'B')])
m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)], producer_name='roadsense-signs pose fixture')
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, sys.argv[1])
