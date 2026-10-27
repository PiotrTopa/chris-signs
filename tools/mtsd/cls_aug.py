"""Crop transforms for the 64x64 sign classifier, mirroring signd's classifier input on the phone.

A stored sample is a 96x96 RGB square region with the box centred, its long side 64 px (extract_crops.py). signd
crops the detector box from the frame with no margin, scales the short side to 64 and centre-crops 64x64, from an
NV12 frame (chroma at half resolution) in which the box is only `s` px long (s = box long side at 1280 px frame
width). `render()` reproduces that: optional box jitter (detector noise), downscale of the region so the box is
`s` px, optional JPEG, NV12 chroma subsampling, then the signd crop resampled to 64x64.
"""
import numpy as np
import cv2

S, BOX, OUT = 96, 64, 64


def phone_size(m):
    """long side of the box in px on a 1280-px-wide frame (the phone's ride frames are 1280x720)"""
    return max(m['w'], m['h']) * 1280.0 / m['W']


def _nv12(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    h, w = yuv.shape[:2]
    if h >= 2 and w >= 2:
        c = cv2.resize(yuv[..., 1:], (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        yuv[..., 1:] = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)


def render(x, sw, sh, s, rng=None, jitter=0.0, jpeg=None, photo=False, rot=0.0):
    """x: uint8 [96,96,3]; (sw, sh): box size in stored px; s: target box long side in frame px (None/>=64: none)."""
    R = x.shape[0]                    # stored region size (96, or smaller when train_cls.py shrinks it)
    cx = cy = R / 2.0
    bw, bh = float(sw), float(sh)
    if rng is not None and jitter > 0:
        cx += rng.normal(0, jitter) * bw; cy += rng.normal(0, jitter) * bh
        bw *= np.exp(rng.normal(0, jitter * 1.5)); bh *= np.exp(rng.normal(0, jitter * 1.5))
    img = x
    if rot:
        M = cv2.getRotationMatrix2D((R / 2, R / 2), rot, 1.0)
        img = cv2.warpAffine(img, M, (R, R), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    f = 1.0
    if s is not None and s < max(bw, bh):
        f = s / max(bw, bh)
        n = max(2, int(round(R * f)))
        img = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
        f = n / R
    if jpeg:
        ok, enc = cv2.imencode('.jpg', img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    img = _nv12(np.ascontiguousarray(img))
    # signd: box (no margin) -> short side to 64 -> centre crop 64x64
    side = min(bw, bh) * f
    x0, y0 = cx * f - side / 2, cy * f - side / 2
    sc = OUT / side
    M = np.array([[sc, 0, -x0 * sc + 0.5 * sc - 0.5], [0, sc, -y0 * sc + 0.5 * sc - 0.5]], np.float32)
    out = cv2.warpAffine(img, M, (OUT, OUT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if photo and rng is not None:
        o = out.astype(np.float32)
        o = o * rng.uniform(0.6, 1.35) + rng.uniform(-20, 20)                       # gain / offset
        g = o.mean(2, keepdims=True); o = g + (o - g) * rng.uniform(0.6, 1.3)       # saturation
        o = 255.0 * (np.clip(o, 0, 255) / 255.0) ** rng.uniform(0.75, 1.35)         # gamma
        if rng.random() < 0.3: o += rng.normal(0, rng.uniform(2, 8), o.shape)       # sensor noise
        out = np.clip(o, 0, 255).astype(np.uint8)
        if rng.random() < 0.15:                                                     # motion blur
            k = int(rng.integers(3, 6)); ker = np.zeros((k, k), np.float32)
            if rng.random() < 0.5: ker[k // 2, :] = 1.0 / k
            else: ker[:, k // 2] = 1.0 / k
            out = cv2.filter2D(out, -1, ker)
    return out


def train_sample(x, m, rng):
    s0 = phone_size(m) if 'W' in m else 40.0
    r = rng.random()
    if r < 0.15: s = None                                   # full resolution
    elif r < 0.55: s = float(np.clip(s0 * np.exp(rng.normal(0, 0.35)), 9, 200))
    else: s = float(np.exp(rng.uniform(np.log(9), np.log(56))))
    jpeg = int(rng.integers(35, 95)) if rng.random() < 0.6 else None
    rot = float(rng.normal(0, 4)) if rng.random() < 0.5 else 0.0
    return render(x, m['sw'], m['sh'], s, rng, jitter=0.06, jpeg=jpeg, photo=True, rot=rot)


def eval_sample(x, m, cond):
    """cond: 'full' (no downscale), 'phone' (box at its phone size), or an int px cap (min(phone size, cap))"""
    if cond == 'full': s = None
    elif cond == 'phone': s = phone_size(m) if 'W' in m else None
    else: s = min(phone_size(m), float(cond)) if 'W' in m else float(cond)
    return render(x, m['sw'], m['sh'], s, None, jpeg=80 if cond != 'full' else None)
