"""The re-identification network of tools/reid/train_reid.py and its ONNX export.
  reid_net.py STEM   STEM.pt -> STEM.onnx (input 1x3x32x32 RGB / 255: what signd --lights-ru-emb loads) and
                     STEM-batch.onnx (any batch: tools/reid/reid_eval.py --onnx)"""
import sys
import torch, torch.nn as nn, torch.nn.functional as F


class Net(nn.Module):
    def __init__(s, d=32):
        super().__init__()
        def blk(a, b): return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                            nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        s.b1, s.b2, s.b3, s.b4 = blk(3, 16), blk(16, 32), blk(32, 64), blk(64, 96)
        s.fc = nn.Linear(96, d)
    def forward(s, x):
        x = (x - 0.45) / 0.25
        x = F.max_pool2d(s.b1(x), 2); x = F.max_pool2d(s.b2(x), 2); x = F.max_pool2d(s.b3(x), 2); x = s.b4(x)
        return F.normalize(s.fc(x.mean((2, 3))), dim=1)


def export(net, stem):
    net.eval()
    torch.onnx.export(net, torch.zeros(1, 3, 32, 32), stem + '.onnx', input_names=['in'], output_names=['emb'],
                      opset_version=17, dynamo=False)
    torch.onnx.export(net, torch.zeros(2, 3, 32, 32), stem + '-batch.onnx', input_names=['in'], output_names=['emb'],
                      dynamic_axes={'in': {0: 'n'}, 'emb': {0: 'n'}}, opset_version=17, dynamo=False)


if __name__ == '__main__':
    n = Net()
    n.load_state_dict(torch.load(sys.argv[1] + '.pt', map_location='cpu'))
    export(n, sys.argv[1])
    print('parameters', sum(p.numel() for p in n.parameters()))
