import numpy as np, cv2
def skin_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    m = (((h <= 25) | (h >= 165)) & (s > 18) & (s < 150) & (v > 90)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.erode(m, np.ones((11, 11), np.uint8))
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 4)[..., None]
def retouch(rgb, k_red=0.75, k_band=0.7, k_tex=0.35):
    """light body retouch: fade redness and elastic/underwear marks, keep pores/texture"""
    m = skin_mask(rgb); lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    def nblur(x, s):                                   # skin-only normalized blur
        w = m[..., 0]; return cv2.GaussianBlur(x * w, (0, 0), s) / (cv2.GaussianBlur(w, (0, 0), s) + 1e-4)
    big_a = nblur(a, 30); da = a - big_a
    a2 = big_a + np.minimum(da, 0) + (1 - k_red) * np.maximum(da, 0)
    big_L = nblur(L, 22); low_L = cv2.GaussianBlur(L, (0, 0), 2.2); hi = L - low_L
    L2 = big_L + (low_L - big_L) * (1 - k_band) + hi * (1 - k_tex)
    big_b = nblur(b, 30); b2 = big_b + (b - big_b) * 0.8
    out = np.stack([L2, a2, b2], -1)
    out = lab * (1 - m) + out * m
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
