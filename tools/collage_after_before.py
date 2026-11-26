#!/usr/bin/env python3
"""ПОСЛЕ (top) / ДО (bottom) collages, neck to fingertips, same body scale in both halves, background removed (black).

usage: python3 tools/collage_after_before.py <dir> → <dir>/col_<angle>.png
  <dir>/{po,do}_{front,34,side}.jpg and m_*.png (rembg masks); landmarks (neck y, fingertip y) below.
"""
import sys, os, numpy as np, cv2
D = os.environ.get("COLLAGE_DIR") or sys.argv[1]; W, H = 1080, 1920; PH = H // 2
LM = {"po_front": (155, 1870), "po_34": (115, 1905), "po_side": (65, 1900),
      "do_front": (-10, 1990), "do_34": (70, 2275), "do_side": (30, 2265)}
import json
if os.path.exists(os.path.join(D, "lm.json")): LM = {k: tuple(v) for k, v in json.load(open(os.path.join(D, "lm.json"))).items() if k != "anchor"}   # per-patient landmarks
NAVY = np.float32([0, 12, 40]) / 255
yy, xx = np.mgrid[0:PH, 0:W].astype(np.float32)
BG = np.zeros((PH, W, 3), np.float32)                                          # plain black backdrop (the client's call)
def mask(n):
    im = cv2.imread(os.path.join(D, n + ".jpg")); m = cv2.imread(os.path.join(D, "m_" + n + ".png"), 0) > 128
    g = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 2]; thr = g > 72                   # the backdrop is near-black: the bright body/leggings come out by brightness too
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); backdrop = (hsv[..., 2] < 62) & (hsv[..., 1] < 80)   # grey-black backdrop pixels the matte let through
    k = ((m | thr) & ~backdrop).astype(np.uint8); k = cv2.morphologyEx(k, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_, lab, st, _ = cv2.connectedComponentsWithStats(k, 8); big = 1 + np.argmax(st[1:, 4]); k = (lab == big).astype(np.uint8)
    n2, l2, s2, _ = cv2.connectedComponentsWithStats(1 - k, 4)                     # fill only small holes (stickers, gaps), never the backdrop
    for j in range(1, n2):
        if s2[j, 4] < 0.004 * k.size and g[l2 == j].mean() > 60: k[l2 == j] = 1     # a dark hole is the backdrop showing through (between the legs)
    a = cv2.GaussianBlur(k.astype(np.float32), (0, 0), 1.6); return im[..., ::-1].astype(np.float32) / 255, a
def axis(a, y0, y1):
    """the body's lean: a line through the silhouette's mid-points (outer edges) from the neck to the fingertips"""
    sp = y1 - y0; ys = np.arange(int(y0 + 0.1 * sp), int(y0 + 0.9 * sp), 10); xs = []
    for y in ys:
        r = np.where(a[y] > 0.5)[0]; xs.append((r[0] + r[-1]) / 2 if len(r) else np.nan)
    xs = np.array(xs); ok = ~np.isnan(xs); k, b = np.polyfit(ys[ok], xs[ok], 1); return k, b
ANCH = {"front": (12, 942), "34": (12, 942), "side": (12, 942)}             # panel y of the two landmarks; the same for ДО and ПОСЛЕ, so the bodies match
if os.path.exists(os.path.join(D, "lm.json")):
    _j = json.load(open(os.path.join(D, "lm.json"))); ANCH.update({k: tuple(v) for k, v in _j.get("anchor", {}).items()})
def panel(n, span_px=None, top=None):
    """two landmarks (e.g. shoulder top and crotch) land on the same panel heights in ДО and ПОСЛЕ; the body turned upright and centred"""
    im, a = mask(n); y0, y1 = LM[n]; y0c = max(0, y0); k, b = axis(a, y0c, y1); ya, yb = ANCH[n.split("_", 1)[1]]
    ang = np.degrees(np.arctan(k)); ym = (y0 + y1) / 2; xm = k * ym + b                 # pivot: the body's mid-point
    R = cv2.getRotationMatrix2D((float(xm), float(ym)), -ang, 1.0)                        # stand the axis upright
    s = (yb - ya) / (y1 - y0); S = np.float32([[s, 0, W / 2 - xm * s], [0, s, (ya + yb) / 2 - ym * s]])
    M = (np.vstack([S, [0, 0, 1]]) @ np.vstack([R, [0, 0, 1]]))[:2].astype(np.float32)
    fg = cv2.warpAffine(im, M, (W, PH), flags=cv2.INTER_AREA, borderValue=(0, 0, 0)); al = cv2.warpAffine(a, M, (W, PH), flags=cv2.INTER_AREA)[..., None]
    return fg * al + BG * (1 - al)
if __name__ == "__main__":
    for ang in ("front", "34", "side"):
        c = np.vstack([panel("po_" + ang), panel("do_" + ang)]); c[PH - 1:PH + 1] = np.float32([233, 197, 193]) / 255
        cv2.imwrite(os.path.join(D, f"col_{ang}.png"), (np.clip(c, 0, 1)[..., ::-1] * 255).astype(np.uint8))
