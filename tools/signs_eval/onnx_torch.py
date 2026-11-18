"""Minimal ONNX graph interpreter on torch (GPU), for the offline evaluation of lane L24-S only.

Runs the static YOLO exports of this repository (sign detector, sign classifier, BDD100K light detector, the
verifier) node by node with torch ops, so a laptop GPU can produce the per-frame detections of whole rides in
minutes instead of hours on the CPU. Checked against ONNX Runtime on real inputs (check_against_ort below);
it is not a general converter: only the op types those graphs use are implemented.
"""
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper


def _attrs(node):
    out = {}
    for a in node.attribute:
        v = onnx.helper.get_attribute_value(a)
        if isinstance(v, bytes):
            v = v.decode()
        if isinstance(v, onnx.TensorProto):
            v = numpy_helper.to_array(v)
        out[a.name] = v
    return out


class OnnxTorch:
    def __init__(self, path, device="cuda", half=False):
        m = onnx.load(path)
        self.dev = torch.device(device)
        self.half = half
        self.init = {}
        for t in m.graph.initializer:
            a = numpy_helper.to_array(t)
            # integer tensors are shapes / indices: keep them on the CPU (no GPU sync when read)
            self.init[t.name] = torch.from_numpy(a.copy()).to(self.dev if a.dtype.kind == "f" else "cpu")
        self.nodes = [(n.op_type, list(n.input), list(n.output), _attrs(n)) for n in m.graph.node]
        self.inputs = [i.name for i in m.graph.input if i.name not in self.init]
        self.outputs = [o.name for o in m.graph.output]
        # free each intermediate tensor after its last use (keeps GPU memory at the live set)
        last = {}
        for i, (_, ins, _, _) in enumerate(self.nodes):
            for n in ins:
                last[n] = i
        self.free_after = [[n for n in ins if n and last.get(n) == i and n not in self.init and n not in self.outputs]
                           for i, (_, ins, _, _) in enumerate(self.nodes)]

    @torch.no_grad()
    def __call__(self, *xs):
        env = dict(self.init)
        for name, x in zip(self.inputs, xs):
            env[name] = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x)).to(self.dev)
        for k, (op, ins, outs, at) in enumerate(self.nodes):
            a = [env[i] if i else None for i in ins]
            r = getattr(self, "op_" + op)(a, at)
            if not isinstance(r, (list, tuple)):
                r = [r]
            for o, v in zip(outs, r):
                env[o] = v
            for n in self.free_after[k]:
                env.pop(n, None)
        return [env[o] for o in self.outputs]

    # ---- ops
    def op_Conv(self, a, at):
        x, w = a[0], a[1]
        b = a[2] if len(a) > 2 else None
        p = at.get("pads", [0, 0, 0, 0])
        if p[0] != p[2] or p[1] != p[3]:
            x = F.pad(x, (p[1], p[3], p[0], p[2]))
            pad = 0
        else:
            pad = (p[0], p[1])
        return F.conv2d(x, w, b, stride=tuple(at.get("strides", [1, 1])), padding=pad,
                        dilation=tuple(at.get("dilations", [1, 1])), groups=at.get("group", 1))

    def op_Relu(self, a, at): return F.relu(a[0])
    def op_Sigmoid(self, a, at): return torch.sigmoid(a[0])
    def _same(self, x, y):
        if x.device != y.device:
            if x.device.type == "cpu":
                x = x.to(y.device)
            else:
                y = y.to(x.device)
        return x, y

    def op_Mul(self, a, at): x, y = self._same(a[0], a[1]); return x * y
    def op_Add(self, a, at): x, y = self._same(a[0], a[1]); return x + y
    def op_Sub(self, a, at): x, y = self._same(a[0], a[1]); return x - y

    def op_Div(self, a, at):
        x, y = self._same(a[0], a[1])
        if not x.is_floating_point() and not y.is_floating_point():
            return torch.div(x, y, rounding_mode="trunc")
        return x / y

    def op_MatMul(self, a, at): return torch.matmul(a[0], a[1])
    def op_Concat(self, a, at):
        dev = next((t.device for t in a if t.device.type != "cpu"), a[0].device)
        return torch.cat([t.to(dev) for t in a], dim=at["axis"])

    def op_Split(self, a, at):
        ax = at.get("axis", 0)
        if len(a) > 1 and a[1] is not None:
            sizes = [int(v) for v in a[1].tolist()]
            if sum(sizes) != a[0].shape[ax]:
                raise ValueError("split sizes")
        elif "split" in at:
            sizes = list(at["split"])
        else:
            n = at.get("num_outputs")
            sizes = a[0].shape[ax] // n
        return list(torch.split(a[0], sizes, dim=ax))

    def op_MaxPool(self, a, at):
        p = at.get("pads", [0, 0, 0, 0])
        return F.max_pool2d(a[0], tuple(at["kernel_shape"]), stride=tuple(at.get("strides", [1, 1])),
                            padding=(p[0], p[1]))

    def op_Resize(self, a, at):
        assert at.get("mode", "nearest") == "nearest"
        scales = a[2] if len(a) > 2 and a[2] is not None and a[2].numel() else None
        if scales is not None:
            s = scales.cpu().tolist()
            return F.interpolate(a[0], scale_factor=(s[2], s[3]), mode="nearest")
        sz = [int(v) for v in a[3].tolist()]
        return F.interpolate(a[0], size=(sz[2], sz[3]), mode="nearest")

    def op_Reshape(self, a, at):
        shp = [int(v) for v in a[1].tolist()]
        shp = [a[0].shape[i] if s == 0 else s for i, s in enumerate(shp)]
        if shp and shp[0] == 1 and a[0].dim() >= 3 and a[0].shape[0] > 1:  # static batch-1 exports: batch through
            shp[0] = a[0].shape[0]
        return a[0].reshape(shp)

    def op_Transpose(self, a, at): return a[0].permute(*at["perm"])
    def op_Softmax(self, a, at): return torch.softmax(a[0], dim=at.get("axis", -1))

    def op_Slice(self, a, at):
        x = a[0]
        starts, ends = a[1].tolist(), a[2].tolist()
        axes = a[3].tolist() if len(a) > 3 and a[3] is not None else list(range(len(starts)))
        steps = a[4].tolist() if len(a) > 4 and a[4] is not None else [1] * len(starts)
        idx = [slice(None)] * x.dim()
        for s, e, ax, st in zip(starts, ends, axes, steps):
            n = x.shape[ax]
            s = max(-n, min(n, s)); e = max(-n - 1, min(n, e))
            idx[ax] = slice(s, e, st)
        return x[tuple(idx)]

    def op_Constant(self, a, at):
        v = np.array(at["value"])
        return torch.from_numpy(v).to(self.dev if v.dtype.kind == "f" else "cpu")

    def op_Shape(self, a, at): return torch.tensor(list(a[0].shape), dtype=torch.int64)

    def op_Gather(self, a, at):
        ax = at.get("axis", 0)
        idx = a[1]
        if idx.dim() == 0:
            return a[0].select(ax, int(idx))
        return torch.index_select(a[0], ax, idx.reshape(-1).long()).reshape(
            list(a[0].shape[:ax]) + list(idx.shape) + list(a[0].shape[ax + 1:]))

    def op_Unsqueeze(self, a, at):
        axes = a[1].tolist() if len(a) > 1 else at["axes"]
        x = a[0]
        for ax in sorted(axes):
            x = x.unsqueeze(ax)
        return x

    def op_Cast(self, a, at):
        to = at["to"]
        dt = {1: torch.float32, 6: torch.int32, 7: torch.int64, 9: torch.bool, 10: torch.float16, 11: torch.float64}[to]
        return a[0].to(dt)

    def op_Range(self, a, at):
        return torch.arange(a[0].item(), a[1].item(), a[2].item(), dtype=a[0].dtype)

    def op_Expand(self, a, at):
        shp = [int(v) for v in a[1].tolist()]
        x = a[0]
        nd = max(len(shp), x.dim())
        x = x.reshape([1] * (nd - x.dim()) + list(x.shape))
        shp = [1] * (nd - len(shp)) + shp
        tgt = [max(s, xs) if s != -1 else xs for s, xs in zip(shp, x.shape)]
        return x.expand(tgt)

    def op_ConstantOfShape(self, a, at):
        v = at.get("value", np.zeros(1, np.float32))
        v = np.array(v).reshape(-1)
        shp = [int(t) for t in a[0].tolist()]
        return torch.full(shp, float(v[0]) if v.dtype.kind == "f" else int(v[0]),
                          dtype=torch.from_numpy(v).dtype, device=self.dev if v.dtype.kind == "f" else "cpu")

    def op_GlobalAveragePool(self, a, at): return a[0].mean(dim=(2, 3), keepdim=True)
    def op_ReduceMean(self, a, at):
        axes = at.get("axes") if "axes" in at else (a[1].tolist() if len(a) > 1 else None)
        return a[0].mean(dim=tuple(axes), keepdim=bool(at.get("keepdims", 1)))
    def op_Flatten(self, a, at): return torch.flatten(a[0], start_dim=at.get("axis", 1))

    def op_Gemm(self, a, at):
        A, B = a[0], a[1]
        if at.get("transA", 0): A = A.t()
        if at.get("transB", 0): B = B.t()
        y = at.get("alpha", 1.0) * (A @ B)
        if len(a) > 2 and a[2] is not None:
            y = y + at.get("beta", 1.0) * a[2]
        return y


def check_against_ort(path, x):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref = s.run(None, {s.get_inputs()[0].name: x})
    got = OnnxTorch(path)(x)
    return [float(np.abs(r - g.float().cpu().numpy()).max()) for r, g in zip(ref, got)]


if __name__ == "__main__":
    import sys
    path, shape = sys.argv[1], [int(v) for v in sys.argv[2].split("x")]
    x = np.random.default_rng(0).random(shape, dtype=np.float32)
    print(path, "max abs diff vs ORT:", check_against_ort(path, x))
