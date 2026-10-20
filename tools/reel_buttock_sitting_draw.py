SLOW = 1.5                                                               # schematics draw in slowly enough to be read
# line-art schematics that illustrate what the surgeon says (drawn in, palette colours, no glow)
def bez(p0, p1, p2, p3, n=60):
    t = np.linspace(0, 1, n)[:, None]; P = [np.float32(p) for p in (p0, p1, p2, p3)]
    return (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]
def part(pts, p):                                                        # first p (0..1) of a polyline, by length
    if p <= 0: return pts[:1]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]; L = d[-1] * min(p, 1)
    k = np.searchsorted(d, L); out = pts[:max(k, 1)]
    if 0 < k < len(pts): a = (L - d[k - 1]) / max(d[k] - d[k - 1], 1e-6); out = np.vstack([out, pts[k - 1] + (pts[k] - pts[k - 1]) * a])
    return out
def stroke(fr, pts, p, col, th=6, al=1.0):
    if p <= 0: return
    m = np.zeros((H, W), np.uint8); cv2.polylines(m, [np.int32(part(pts, p))], False, 255, th, cv2.LINE_AA)
    a = m.astype(np.float32)[..., None] / 255 * al; fr *= 1 - a; fr += np.float32(col) / 255 * a
def fill(fr, pts, al, col):
    if al <= 0: return
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.int32(pts)], 255, cv2.LINE_AA)
    a = m.astype(np.float32)[..., None] / 255 * al; fr *= 1 - a; fr += np.float32(col) / 255 * a
def dashed(fr, x0, x1, y, p, col, th=4, seg=22):
    xe = x0 + (x1 - x0) * min(max(p, 0), 1); x = x0
    while x < xe: stroke(fr, np.float32([[x, y], [min(x + seg, xe), y]]), 1, col, th); x += 2 * seg
def arrow(fr, x, y0, y1, p, col, th=6):
    if p <= 0: return
    ye = y0 + (y1 - y0) * min(p, 1); stroke(fr, np.float32([[x, y0], [x, ye]]), 1, col, th)
    if p > 0.8: d = 22 if y1 < y0 else -22; stroke(fr, np.float32([[x - 18, ye + d], [x, ye], [x + 18, ye + d]]), 1, col, th)
LBL = {}
def label_at(fr, txt, x, y, al, col=WHITE, size=40, anchor="lm"):
    if al <= 0: return
    key = (txt, size)
    if key not in LBL:
        f = mont(600, size); asc, desc = f.getmetrics(); im = Image.new("RGBA", (int(f.getlength(txt)) + 8, asc + desc + 4), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((4, 2 + asc), txt, font=f, fill=(255, 255, 255, 255), anchor="ls"); LBL[key] = to_np(im)
    im = LBL[key]; xx = x if anchor == "lm" else (x - im.shape[1] / 2 if anchor == "mm" else x - im.shape[1])
    ablend(fr, im, xx, y - im.shape[0] / 2, al, col, BLUE)
def dark_bg(t):
    g = np.clip(1 - (((xx_ - 540) / 900) ** 2 + ((yy_ - 860) / 1100) ** 2), 0, 1)[..., None]
    return np.float32([10, 10, 14]) / 255 * (1 - g) + np.float32([24, 30, 42]) / 255 * g
SKIN, MUS, IMP = (228, 210, 210), (184, 128, 146), (228, 196, 200)
