import cv2, numpy as np
import tat3
from tat3 import ink_mask, remove_ink
from tat import nconv
K = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
def stickers(bgr):
    B = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 2].astype(int) - 128
    bl = cv2.morphologyEx((B < -10).astype(np.uint8), cv2.MORPH_OPEN, K(5))
    n, l, st, c = cv2.connectedComponentsWithStats(bl, 8)
    return [(c[i][0], c[i][1], np.sqrt(st[i, 4] / np.pi)) for i in range(1, n) if st[i, 4] > 120]
def script_zone(bgr):
    z = np.zeros(bgr.shape[:2], np.uint8); s = [x for x in stickers(bgr) if x[2] > 15]
    if len(s) >= 2:
        s.sort(key=lambda x: x[0]); cx, cy, r = s[-1]                 # patient's left breast = image right
        cv2.ellipse(z, (int(cx), int(cy + 4.25 * r)), (int(3.3 * r), int(1.4 * r)), 0, 0, 360, 1, -1)
        z[:int(cy + 2.85 * r)] = 0
    return z
def dense(m):
    d = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 14)
    return (m > 0) & (d > 0.07)
def fill_zone(im, m):
    f = im.astype(np.float32); L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0]
    R = cv2.dilate(m, K(7)) & cv2.erode(cv2.dilate(raw_skin(im), K(25)), K(23)); Wt = ((raw_skin(im) > 0) & (R == 0)).astype(np.float32)
    low, _ = nconv(f, Wt, 8); hi = np.clip(f - cv2.GaussianBlur(f, (0, 0), 2), -1, 4)
    a = cv2.GaussianBlur(R.astype(np.float32), (0, 0), 1.6)[..., None]
    return np.clip(f * (1 - a) + (low + hi) * a, 0, 255).astype(np.uint8)
from tat import raw_skin
def arm_zone(bgr, P):
    z = np.zeros(bgr.shape[:2], np.uint8)
    if P is None or len([x for x in stickers(bgr) if x[2] > 15]) < 2: return z     # frontal views only
    sw = np.linalg.norm(P[11] - P[12])
    for a, b, k0, k, wd in ((11, 13, 0.25, 1.0, 0.22), (13, 15, 0.0, 0.93, 0.34), (12, 14, 0.25, 1.0, 0.22), (14, 16, 0.0, 0.93, 0.34)):
        s_ = P[a] + (P[b] - P[a]) * k0; e = P[a] + (P[b] - P[a]) * k
        cv2.line(z, tuple(int(v) for v in s_), tuple(int(v) for v in e), 1, max(3, int(wd * sw)))
    return z
def clean4(bgr, pose=None):
    r = bgr; sz = script_zone(bgr)
    for _ in range(4):
        m = ink_mask(r, pose); m = dense(m).astype(np.uint8)
        r = remove_ink(r, m)
    az = arm_zone(bgr, pose)
    rs0 = cv2.morphologyEx(raw_skin(bgr), cv2.MORPH_OPEN, K(5))
    cs, _ = cv2.findContours(cv2.morphologyEx(rs0, cv2.MORPH_CLOSE, K(9)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(rs0); cv2.drawContours(filled, cs, -1, 1, -1)
    skin = cv2.erode(filled, K(11))
    near = cv2.dilate(dense(ink_mask(bgr, pose)).astype(np.uint8), K(41))
    zones = [(sz & skin, 5, 31), (az & skin & near, 6, 61)]
    for z, th, kk in zones:
        if not z.any(): continue
        for _ in range(3):
            L = cv2.cvtColor(r, cv2.COLOR_BGR2LAB)[..., 0]
            m = ((L.astype(int) < cv2.medianBlur(L, kk).astype(int) - th) & (z > 0)).astype(np.uint8)
            r = fill_zone(r, m)
    return r
