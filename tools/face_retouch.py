import cv2, numpy as np
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
EYE_R = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]; EYE_L = [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249]
LIPS = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
BROW_R = [70,63,105,66,107,55,65,52,53,46]; BROW_L = [300,293,334,296,336,285,295,282,283,276]
def poly(m, P, idx, v, grow=0):
    pts = P[idx].astype(np.float32); c = pts.mean(0); pts = c + (pts - c) * (1 + grow)
    cv2.fillPoly(m, [pts.astype(np.int32)], v)
def face_mask(shape, P):
    m = np.zeros(shape[:2], np.float32); poly(m, P, FACE_OVAL, 1, 0.04)
    for idx, g in ((EYE_R, 0.6), (EYE_L, 0.6), (LIPS, 0.15), (BROW_R, 0.3), (BROW_L, 0.3)): poly(m, P, idx, 0, g)
    return cv2.GaussianBlur(m, (0, 0), 6)
def retouch_face(bgr, P, strength=1.0):
    m = face_mask(bgr.shape, P)[..., None] * strength
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    fm = m[..., 0] > 0.5
    ref_s = np.median(s[fm]); ref_v = np.median(v[fm])
    yel = np.clip((h - 12) / 6, 0, 1) * (h < 35) * np.clip((s - ref_s * 0.9) / 40, 0, 1)        # yellow iodine / bruise
    pur = ((h > 140) | (h < 4)).astype(np.float32) * np.clip((ref_v - v) / 40, 0, 1) * 0.8      # purple/dark bruises
    w = cv2.GaussianBlur(np.maximum(yel, pur), (0, 0), 4)
    hsv[..., 0] = h - (h - 9) * 0.55 * w
    hsv[..., 1] = s - (s - ref_s) * 0.6 * w
    hsv[..., 2] = v + np.clip(ref_v - v, 0, None) * 0.35 * w
    shine = np.clip((v - ref_v - 25) / 40, 0, 1) * np.clip((ref_s - s) / 40, 0, 1)                # greasy highlights
    hsv[..., 2] -= cv2.GaussianBlur(shine, (0, 0), 3) * 18
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    sm = cv2.bilateralFilter(out, 9, 28, 7)
    out = out.astype(np.float32) * (1 - 0.5 * m) + sm.astype(np.float32) * 0.5 * m
    return np.clip(bgr.astype(np.float32) * (1 - m) + out * m, 0, 255).astype(np.uint8)
def nconv(f, w, s):
    num = cv2.GaussianBlur(f * w[..., None], (0, 0), s); den = cv2.GaussianBlur(w, (0, 0), s)[..., None]
    return num / np.maximum(den, 1e-4)
def retouch2(bgr, P, k_col=0.35, k_dark=0.5, k_shine=0.7, yshift=4):
    m = face_mask(bgr.shape, P)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    base = nconv(lab, (m > 0.3).astype(np.float32), 22)
    fine = lab[..., 0] - cv2.GaussianBlur(lab[..., 0], (0, 0), 1.5)
    d = lab - base
    out = lab.copy()
    out[..., 1] = base[..., 1] + d[..., 1] * k_col
    out[..., 2] = base[..., 2] + d[..., 2] * k_col - yshift
    dL = d[..., 0] - fine
    out[..., 0] = base[..., 0] + np.where(dL < 0, dL * k_dark, dL * k_shine) + fine
    res = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    mm = m[..., None]
    return np.clip(bgr.astype(np.float32) * (1 - mm) + res * mm, 0, 255).astype(np.uint8)
