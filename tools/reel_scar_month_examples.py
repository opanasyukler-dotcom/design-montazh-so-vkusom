# patient examples for the scar reel: background removed, cropped neck -> fingertips
EXS = {"a_do": (560, 1880, 745), "a_po": (30, 1802, 745), "b_do": (40, 1020, 515), "b_po": (129, 2048, 721)}
def ex_crop(name, out_w, out_h):
    img = np.array(Image.open(f"ex/{name}.jpg").convert("RGB")).astype(np.float32)
    m = np.load(f"ex/m_{name}.npy").astype(np.float32); m = cv2.GaussianBlur(np.clip((m - 0.5) / 0.3, 0, 1), (0, 0), 1.5)[..., None]
    img = img * m + 8 * (1 - m)
    top, bot, cx = EXS[name]; s = out_h / (bot - top)
    M = np.float32([[s, 0, out_w / 2 - s * cx], [0, s, -s * top]])
    out = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_AREA, borderValue=(8, 8, 8)) / 255
    y = np.arange(out_h, dtype=np.float32)[:, None, None]; f = np.clip(y / 30, 0, 1) * np.clip((out_h - 1 - y) / 30, 0, 1)
    return (out * f + 8 / 255 * (1 - f)).astype(np.float32)
SL_W, SL_H = 720, 1240
SL_DO, SL_PO = ex_crop("a_do", SL_W, SL_H), ex_crop("a_po", SL_W, SL_H)
CB = [ex_crop("b_do", 470, 980), ex_crop("b_po", 470, 980)]
T_EX = timg("пример моей пациентки", "S", 0)
def dim_bg(fr, k):
    bg = cv2.GaussianBlur(fr, (0, 0), 14) * 0.35; fr[:] = fr * (1 - k) + bg * k
def rounded(img, r=28):
    h, w = img.shape[:2]; m = np.zeros((h, w), np.uint8); cv2.rectangle(m, (r, 0), (w - r, h), 255, -1); cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for x, y in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)): cv2.circle(m, (x, y), r, 255, -1, cv2.LINE_AA)
    return np.concatenate([img * (m[..., None] / 255.0), m[..., None] / 255.0], -1).astype(np.float32)
def slider_overlay(fr, t, a, b, s0, s1):
    """ДО -> ПОСЛЕ with a sideways slider (divider + round handle)"""
    if not a <= t <= b + 0.35: return False
    k = min(1, (t - a) / 0.35) * (1 - min(1, max(0, (t - b) / 0.35))); dim_bg(fr, k)
    p = ease((t - s0) / (s1 - s0)); xs = int(SL_W * p)                  # left of xs shows ПОСЛЕ
    img = SL_DO.copy(); img[:, :xs] = SL_PO[:, :xs]
    rgba = rounded(img); cx, cy = 540, 925; x0, y0 = cx - SL_W / 2, cy - SL_H / 2 + 40 * (1 - ease((t - a) / 0.45))
    blend(fr, shadow_of(rgba, 24, 0.4), x0, y0 + 14, k); blend(fr, rgba, x0, y0, k)
    if 0 < p < 1 or t > s0 - 0.2:
        X = int(x0 + max(xs, 3)); m = np.zeros((H, W), np.uint8); cv2.line(m, (X, int(y0)), (X, int(y0 + SL_H)), 255, 5)
        cv2.circle(m, (X, int(cy)), 38, 255, -1, cv2.LINE_AA)
        al = m.astype(np.float32)[..., None] / 255 * k; fr[:] = fr * (1 - al) + al
        ar = np.zeros((H, W), np.uint8)
        for d in (-1, 1): cv2.polylines(ar, [np.int32([[X + d * 8, cy - 12], [X + d * 20, cy], [X + d * 8, cy + 12]])], False, 255, 4, cv2.LINE_AA)
        al = ar.astype(np.float32)[..., None] / 255 * k; fr[:] = fr * (1 - al) + np.float32(BLUE) / 255 * al
    chip(fr, CH_DO, x0 + SL_W - 150, y0 + 24, t, a + 0.2, s0 + 0.4); chip(fr, CH_PO, x0 + 24, y0 + 24, t, s1 - 0.3)
    ablend(fr, T_EX, 540 - T_EX.shape[1] / 2, y0 - 70, k, WHITE, BLUE)
    return True
def collage_overlay(fr, t, a, b):
    if not a <= t <= b + 0.35: return False
    k = min(1, (t - a) / 0.35) * (1 - min(1, max(0, (t - b) / 0.35))); dim_bg(fr, k)
    gap = 14; w = 470; x0 = 540 - w - gap / 2; y0 = 330 + 40 * (1 - ease((t - a) / 0.45))
    for i, img in enumerate(CB):
        rgba = rounded(img, 24); x = x0 + i * (w + gap); al = k * min(1, max(0, (t - a - 0.15 * i) / 0.3))
        blend(fr, shadow_of(rgba, 22, 0.4), x, y0 + 12, al); blend(fr, rgba, x, y0, al)
    chip(fr, CH_DO, x0 + 20, y0 + 20, t, a + 0.25); chip(fr, CH_PO, x0 + w + gap + 20, y0 + 20, t, a + 0.45)
    ablend(fr, T_EX, 540 - T_EX.shape[1] / 2, y0 - 70, k, WHITE, BLUE)
    return True
