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
TI1 = Title([("ИМПЛАНТ ПОД МЫШЦЕЙ", "P")], 92); TI2 = Title([("ПОДТЯЖКА НЕ НУЖНА", "P")], 92)
def illus_muscle(fr, t, t0):
    """profile: chest wall, implant under the muscle, the breast keeps its shape"""
    P = lambda a, b: ease((t - t0 - a * SLOW) / (b * SLOW))
    wall = np.float32([[300, 430], [300, 1260]])
    skin = np.vstack([bez((300, 560), (520, 580), (730, 800), (720, 900)), bez((720, 900), (712, 1020), (520, 1080), (300, 1080))])
    imp = np.vstack([bez((305, 680), (420, 680), (610, 860), (600, 940)), bez((600, 940), (590, 1040), (420, 1050), (305, 1040)), [[305, 680]]])
    mus = bez((300, 540), (470, 600), (570, 740), (610, 860))
    stroke(fr, wall, P(0.1, 0.5), SKIN, 5, 0.6)
    fill(fr, imp, 0.85 * P(0.9, 0.5), IMP); stroke(fr, imp, P(0.8, 0.8), MUS, 5)
    stroke(fr, mus, P(1.3, 0.7), MUS, 16)
    stroke(fr, skin, P(0.3, 1.0), SKIN, 7)
    if t > t0 + 1.1: cv2.circle(fr, (720, 900), 11, tuple(float(c) / 255 for c in SKIN), -1, cv2.LINE_AA)
    arrow(fr, 850, 990, 800, P(1.8, 0.5), MUS, 7)
    label_at(fr, "мышца", 640, 690, P(1.6, 0.4)); stroke(fr, np.float32([[632, 690], [540, 700]]), P(1.6, 0.4), WHITE, 3)
    label_at(fr, "имплант", 640, 1080, P(1.2, 0.4)); stroke(fr, np.float32([[632, 1080], [520, 1010]]), P(1.2, 0.4), WHITE, 3)
    label_at(fr, "форма держится", 850, 1170, P(2.0, 0.4), PINK, 38, "mm")
    TI1.draw(fr, 250, t, t0 + 0.1, 1e9)
def profile(cx, y0, s, ptotic):
    w = cx - 120 * s
    if ptotic:
        up = bez((w, y0), (w + 120 * s, y0 + 40 * s), (w + 240 * s, y0 + 300 * s), (w + 210 * s, y0 + 470 * s))
        lo = bez((w + 210 * s, y0 + 470 * s), (w + 180 * s, y0 + 560 * s), (w + 60 * s, y0 + 520 * s), (w, y0 + 430 * s))
        nip = (w + 205 * s, y0 + 480 * s)
    else:
        up = bez((w, y0), (w + 140 * s, y0 + 30 * s), (w + 250 * s, y0 + 170 * s), (w + 240 * s, y0 + 260 * s))
        lo = bez((w + 240 * s, y0 + 260 * s), (w + 232 * s, y0 + 360 * s), (w + 120 * s, y0 + 420 * s), (w, y0 + 430 * s))
        nip = (w + 240 * s, y0 + 262 * s)
    return np.vstack([up, lo]), w, nip
def illus_ptosis(fr, t, t0):
    """ptosis (nipple below the fold) vs no ptosis"""
    P = lambda a, b: ease((t - t0 - a * SLOW) / (b * SLOW))
    for k, (cx, pt) in enumerate([(300, True), (800, False)]):
        sh, w, nip = profile(cx, 560, 1.0, pt); d = 0.25 + 0.5 * k
        stroke(fr, np.float32([[w, 470], [w, 1150]]), P(d, 0.4), SKIN, 5, 0.6)
        stroke(fr, sh, P(d + 0.1, 0.9), SKIN, 7)
        dashed(fr, w - 10, w + 290, 560 + 430, P(d + 0.6, 0.5), MUS)
        if t > t0 + d + 0.9: cv2.circle(fr, (int(nip[0]), int(nip[1])), 11, tuple(float(c) / 255 for c in SKIN), -1, cv2.LINE_AA)
    label_at(fr, "складка под грудью", 540, 1095, P(1.3, 0.4), MUS, 30, "mm")
    label_at(fr, "ПТОЗ", 260, 1200, P(1.5, 0.4), WHITE, 48, "mm")
    label_at(fr, "БЕЗ ПТОЗА", 760, 1200, P(1.7, 0.4), PINK, 48, "mm")
    stroke(fr, np.float32([[140, 1180], [180, 1220]]), P(1.9, 0.2), MUS, 7); stroke(fr, np.float32([[180, 1180], [140, 1220]]), P(2.0, 0.2), MUS, 7)   # small ✕ by ПТОЗ
    stroke(fr, np.float32([[536, 1200], [558, 1222], [600, 1176]]), P(2.1, 0.3), PINK, 7)                                                    # small ✓ by БЕЗ ПТОЗА
    TI2.draw(fr, 250, t, t0 + 0.1, 1e9)
def implant_icon(fr, t, t0, cx=930, cy=330, s=0.9):
    """anatomical (teardrop) implant, drawn in next to the 295 ml"""
    P = lambda a, b: ease((t - t0 - a * SLOW) / (b * SLOW))
    sh = np.vstack([bez((cx, cy - 90 * s), (cx + 45 * s, cy - 90 * s), (cx + 80 * s, cy + 10 * s), (cx + 70 * s, cy + 60 * s)),
                    bez((cx + 70 * s, cy + 60 * s), (cx + 55 * s, cy + 100 * s), (cx - 55 * s, cy + 100 * s), (cx - 70 * s, cy + 60 * s)),
                    bez((cx - 70 * s, cy + 60 * s), (cx - 80 * s, cy + 10 * s), (cx - 45 * s, cy - 90 * s), (cx, cy - 90 * s))])
    fill(fr, sh, 0.85 * P(0.6, 0.5), IMP); stroke(fr, sh, P(0, 0.9), MUS, 5)
    label_at(fr, "анатомический", cx, cy + 135 * s, P(0.8, 0.4), WHITE, 30, "mm")
