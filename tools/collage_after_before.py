#!/usr/bin/env python3
"""ПОСЛЕ (top) / ДО (bottom) collages, neck to fingertips, same body scale in both halves, background removed (black).

usage: python3 tools/collage_after_before.py <dir> → <dir>/col_<angle>.png
  <dir>/{po,do}_{front,34,side}.jpg and m_*.png (rembg masks); landmarks (neck y, fingertip y) below.
"""
import sys, os, numpy as np, cv2
D = sys.argv[1]; W, H = 1080, 1920; PH = H // 2
LM = {"po_front": (155, 1870), "po_34": (115, 1905), "po_side": (65, 1900),
      "do_front": (-10, 1990), "do_34": (70, 2275), "do_side": (30, 2265)}
NAVY = np.float32([0, 12, 40]) / 255
yy, xx = np.mgrid[0:PH, 0:W].astype(np.float32)
BG = np.zeros((PH, W, 3), np.float32)                                          # plain black backdrop (the client's call)
def mask(n):
    im = cv2.imread(os.path.join(D, n + ".jpg")); m = cv2.imread(os.path.join(D, "m_" + n + ".png"), 0) > 128
    g = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 2]; thr = g > 72                   # the backdrop is near-black: the bright body/leggings come out by brightness too
    k = (m | thr).astype(np.uint8); k = cv2.morphologyEx(k, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_, lab, st, _ = cv2.connectedComponentsWithStats(k, 8); big = 1 + np.argmax(st[1:, 4]); k = (lab == big).astype(np.uint8)
    n2, l2, s2, _ = cv2.connectedComponentsWithStats(1 - k, 4)                     # fill only small holes (stickers, gaps), never the backdrop
    for j in range(1, n2):
        if s2[j, 4] < 0.004 * k.size: k[l2 == j] = 1
    a = cv2.GaussianBlur(k.astype(np.float32), (0, 0), 1.6); return im[..., ::-1].astype(np.float32) / 255, a
def panel(n, span_px=880, top=40):
    im, a = mask(n); y0, y1 = LM[n]; s = span_px / (y1 - y0)
    ys = slice(max(0, y0), y1); cols = np.where(a[ys].max(0) > 0.5)[0]; cx = (cols.min() + cols.max()) / 2 if n.endswith("front") else np.average(np.arange(a.shape[1]), weights=a[ys].sum(0) + 1e-6)
    M = np.float32([[s, 0, W / 2 - cx * s], [0, s, top - y0 * s]])
    fg = cv2.warpAffine(im, M, (W, PH), flags=cv2.INTER_AREA, borderValue=(0, 0, 0)); al = cv2.warpAffine(a, M, (W, PH), flags=cv2.INTER_AREA)[..., None]
    return fg * al + BG * (1 - al)
if __name__ == "__main__":
    for ang in ("front", "34", "side"):
        c = np.vstack([panel("po_" + ang), panel("do_" + ang)]); c[PH - 1:PH + 1] = np.float32([233, 197, 193]) / 255
        cv2.imwrite(os.path.join(D, f"col_{ang}.png"), (np.clip(c, 0, 1)[..., ::-1] * 255).astype(np.uint8))
