"""Python mirror of signd verify.cc glow_score on a 32x32 BILINEAR resample of the box (no margin)."""
import numpy as np
from PIL import Image
S = 32
_yy, _xx = np.mgrid[0:S, 0:S]
ELL = ((_xx + 0.5 - S / 2) / (S / 2)) ** 2 + ((_yy + 0.5 - S / 2) / (S / 2)) ** 2 <= 1.0
def box_from_margin_crop(im, m=0.15):
    W, H = im.size; mx, my = W * m / (1 + 2 * m), H * m / (1 + 2 * m)
    return im.crop((round(mx), round(my), round(W - mx), round(H - my)))
def arr32(im_box):
    return np.asarray(im_box.convert('RGB').resize((S, S), Image.BILINEAR), dtype=np.float32)
def glow(a):
    R, G, B = a[..., 0][ELL], a[..., 1][ELL], a[..., 2][ELL]
    mn = np.minimum(np.minimum(R, G), B)
    return float(np.mean((mn >= 235) | ((R >= 220) & (mn >= 80) & (R - mn >= 40))))
