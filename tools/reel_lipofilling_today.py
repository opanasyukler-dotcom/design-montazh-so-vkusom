"""Lipofilling reel in the FUROR «креативный монтаж» style.

Pale-blue / white / navy palette, wave transitions, text sliding along curves,
the surgeon as a white-outlined sticker, navy line art (fat trap, body map, drops).

Run from a work dir that holds fr/NNNN.jpg + mask/NNNN.png (30 fps frames of the
source and their u2net_human_seg masks) and words.json:
    python3 reel_lipofilling_today.py            -> video.mp4
    python3 reel_lipofilling_today.py 3.5 9.2    -> f/o_<t>.jpg stills
"""
import os, sys, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
OFF = 1.2                       # reel t = source t - OFF (cuts the silence before «Сегодня»)
DUR = 25.9
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Montserrat-VariableFont_wght.ttf")
BLUE, NAVY, WHITE, PALE = (169, 200, 232), (34, 62, 146), (255, 255, 255), (214, 229, 245)

ease = lambda x: (lambda c: 1 - (1 - c) ** 3)(min(1.0, max(0.0, x)))
inout = lambda x: (lambda c: c * c * (3 - 2 * c))(min(1.0, max(0.0, x)))
def back(x):
    c = min(1.0, max(0.0, x)); k = 1.70158; return 1 + (k + 1) * (c - 1) ** 3 + k * (c - 1) ** 2

# ---------------- text ----------------
_F, _T = {}, {}
def font(size, w="ExtraBold"):
    if (size, w) not in _F:
        f = ImageFont.truetype(FONT, size); f.set_variation_by_name(w); _F[size, w] = f
    return _F[size, w]
def timg(s, size, col=NAVY, w="ExtraBold"):
    """RGBA uint8 image of a single line (cached)"""
    k = (s, size, col, w)
    if k not in _T:
        f = font(size, w); asc, desc = f.getmetrics(); pad = size // 6
        im = Image.new("RGBA", (int(f.getlength(s)) + 2 * pad, asc + desc + 2 * pad), col + (0,))
        ImageDraw.Draw(im).text((pad, pad + asc), s, font=f, fill=col + (255,), anchor="ls"); _T[k] = np.array(im)
    return _T[k]
def paste(fr, im, x, y, a=1.0):
    """alpha-blend RGBA image with its top-left at (x, y)"""
    if a <= 0.003: return
    x, y = int(round(x)), int(round(y)); h, w = im.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1: return
    s = im[y0 - y:y1 - y, x0 - x:x1 - x]; al = s[..., 3:4].astype(np.float32) * (a / 255)
    d = fr[y0:y1, x0:x1]; d[:] = (d * (1 - al) + s[..., :3] * al).astype(np.uint8)
def typed(fr, s, size, cx, y, t, t0, col=NAVY, w="ExtraBold", cps=0.032, t1=1e9):
    """FUROR-style typing: letters appear one by one, line centred on its final width"""
    if t < t0: return
    im = timg(s, size, col, w); n = len(s); k = min(n, int((t - t0) / cps) + 1)
    pad = size // 6; cut = pad + int(font(size, w).getlength(s[:k])) + (pad if k == n else 0)
    a = 1 - ease((t - t1) / 0.25) if t > t1 else 1.0
    paste(fr, im[:, :cut], cx - im.shape[1] / 2, y, a)
_C = {}
def chimg(ch, size, col, ang):
    k = (ch, size, col, int(round(ang)))
    if k not in _C:
        f = font(size); asc, desc = f.getmetrics(); wd = max(2, int(f.getlength(ch)))
        im = Image.new("RGBA", (wd + 8, asc + desc + 8), col + (0,))
        ImageDraw.Draw(im).text((4, 4 + asc), ch, font=f, fill=col + (255,), anchor="ls")
        _C[k] = np.array(im.rotate(-k[3], resample=Image.BICUBIC, expand=True))
    return _C[k]
class Path:
    def __init__(s, pts):
        s.p = np.asarray(pts, np.float32); d = np.r_[0, np.cumsum(np.hypot(*np.diff(s.p, axis=0).T))]; s.d = d; s.L = d[-1]
    def at(s, u):
        x = np.interp(u, s.d, s.p[:, 0]); y = np.interp(u, s.d, s.p[:, 1])
        x2 = np.interp(u + 3, s.d, s.p[:, 0]); y2 = np.interp(u + 3, s.d, s.p[:, 1]); x1 = np.interp(u - 3, s.d, s.p[:, 0]); y1 = np.interp(u - 3, s.d, s.p[:, 1])
        return x, y, np.degrees(np.arctan2(y2 - y1, x2 - x1))
def curved(fr, s, size, path, t, t0, col=NAVY, center=0.5, dur=0.8, a=1.0):
    """text sliding in along a curve (from the right end of the path to its resting place)"""
    if t < t0: return
    f = font(size); adv = [f.getlength(c) for c in s]; tot = sum(adv) + 0.06 * size * len(s)
    rest = path.L * center - tot / 2; u0 = rest + (path.L - rest + 40) * (1 - ease((t - t0) / dur))
    asc = f.getmetrics()[0]
    for c, w in zip(s, adv):
        u = u0 + w / 2
        if 0 <= u <= path.L and c != " ":
            x, y, ang = path.at(u); im = chimg(c, size, col, ang)
            # glyph centre sits a bit above the path (baseline on the curve)
            r = np.radians(ang); ox, oy = np.sin(r) * asc * 0.42, -np.cos(r) * asc * 0.42
            paste(fr, im, x + ox - im.shape[1] / 2, y + oy - im.shape[0] / 2, a)
        u0 += w + 0.06 * size

# ---------------- drawing ----------------
def stroke(fr, pts, p, col=NAVY, w=7):
    """polyline drawn up to fraction p of its length"""
    if p <= 0: return
    P = Path(pts); L = P.L * min(1, p); m = P.d <= L
    q = np.vstack([P.p[m], [P.at(L)[:2]]]) if L < P.L else P.p
    cv2.polylines(fr, [np.round(q * 4).astype(np.int32)], False, col, w, cv2.LINE_AA, shift=2)
def fill(fr, pts, col, a=1.0):
    if a <= 0: return
    ov = fr.copy(); cv2.fillPoly(ov, [np.round(np.asarray(pts) * 4).astype(np.int32)], col, cv2.LINE_AA, shift=2)
    fr[:] = cv2.addWeighted(ov, a, fr, 1 - a, 0) if a < 1 else ov
def bez(*P, n=40):
    t = np.linspace(0, 1, n)[:, None]; P = [np.float32(p) for p in P]
    if len(P) == 3: return (1 - t) ** 2 * P[0] + 2 * (1 - t) * t * P[1] + t ** 2 * P[2]
    return (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]
def spline(pts, n=14):
    """Catmull-Rom through points"""
    P = np.float32(pts); P = np.vstack([P[0], P, P[-1]]); out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]; t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out + [P[-2][None]])
def star(fr, cx, cy, r, a=1.0, col=WHITE):
    """FUROR four-point sparkle"""
    if r < 1 or a <= 0: return
    k = np.linspace(0, 2 * np.pi, 9)[:-1]; rr = np.where(np.arange(8) % 2 == 0, r, r * 0.16)
    pts = np.c_[cx + rr * np.sin(k), cy - rr * np.cos(k)]
    sm = np.vstack([bez(pts[i], (cx, cy), pts[(i + 1) % 8], n=10) if i % 2 else bez(pts[i], (cx + (pts[i][0] - cx) * .05, cy + (pts[i][1] - cy) * .05), pts[(i + 1) % 8], n=10) for i in range(8)])
    fill(fr, sm, col, a)
def sparkles(fr, t, t0, spots, a=1.0, col=WHITE):
    for k, (x, y, r) in enumerate(spots):
        lt = t - t0 - 0.12 * k
        if lt > 0: star(fr, x, y, r * back(lt / 0.5) * (1 + 0.08 * np.sin(lt * 3 + k)), a, col)
def drop(fr, cx, cy, s, a=1.0, fillc=WHITE, line=NAVY, w=5):
    tt = np.linspace(0, 2 * np.pi, 60); pts = np.c_[cx + s * 0.8 * np.sin(tt) * np.sin(tt / 2), cy - s * np.cos(tt)]
    fill(fr, pts, fillc, a); stroke(fr, np.vstack([pts, pts[:1]]), 1, line, w)
def bg(col): return np.full((H, W, 3), col, np.uint8)
def wave_edge(base, amp=70, lam=1.7, ph=0.0):
    x = np.arange(0, W + 8, 8, dtype=np.float32); return np.c_[x, base + amp * np.sin(2 * np.pi * x / (lam * W) + ph)]
def wave_fill(fr, edge, col, a=1.0):
    fill(fr, np.vstack([edge, [[W + 10, H + 10], [-10, H + 10]]]), col, a)

# ---------------- sticker (surgeon + patient cut out) ----------------
NF = len(os.listdir("fr")) if os.path.isdir("fr") else 0
_S = {}
def sticker_src(t):
    i = min(max(int(round((t + OFF) * FPS)) + 1, 1), NF)
    if i not in _S:
        _S.clear(); img = cv2.cvtColor(cv2.imread(f"fr/{i:04d}.jpg"), cv2.COLOR_BGR2RGB); m = cv2.imread(f"mask/{i:04d}.png", 0)
        m = cv2.GaussianBlur(m, (0, 0), 1.0); m[1075:] = 0          # nothing below the patient (sheet / table)
        hard = (m > 110).astype(np.uint8)
        out = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), 1.2)
        _S[i] = (img, m.astype(np.float32) / 255, out)
    return _S[i]
def sticker(fr, t, sc, sx, sy, cx, cy, shadow=True):
    """source point (sx, sy) lands on canvas (cx, cy) at scale sc"""
    img, m, out = sticker_src(t)
    M = np.float32([[sc, 0, cx - sx * sc], [0, sc, cy - sy * sc]])
    wi = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR); wm = cv2.warpAffine(m, M, (W, H))[..., None]
    wo = cv2.warpAffine(out, M, (W, H))[..., None]
    f = fr.astype(np.float32)
    if shadow:
        sh = cv2.GaussianBlur(cv2.warpAffine(out, np.float32([[sc, 0, cx - sx * sc + 10], [0, sc, cy - sy * sc + 16]]), (W, H)), (0, 0), 14)[..., None]
        f = f * (1 - 0.18 * sh) + np.float32(NAVY) * 0.18 * sh
    f = f * (1 - wo) + 255 * wo; f = f * (1 - wm) + wi * wm
    fr[:] = f.astype(np.uint8)
    return lambda x, y: (cx + (x - sx) * sc, cy + (y - sy) * sc)

# ---------------- scenes ----------------
CUTS = [("intro", 0.0), ("butt", 3.15), ("trap", 7.75), ("map", 13.85), ("show", 19.55)]
def cam(t):
    """sticker camera: wide on the surgeon -> push into the buttocks -> back"""
    wide = (1.75, 360, 1075, 540, 1900)            # sc, src x, src y, canvas x, canvas y (patient's edge at the bottom)
    close = (2.4, 290, 960, 540, 1330)
    p = inout((t - 3.0) / 0.9)
    return tuple(a + (b - a) * p for a, b in zip(wide, close))

def s_intro(t):
    fr = bg(BLUE); lt = t
    up = 1 - back((t - 0.25) / 0.75)               # sticker slides up from below
    sc, sx, sy, cx, cy = cam(t); sticker(fr, t, sc, sx, sy, cx, cy + 1800 * up)
    sparkles(fr, t, -0.4, [(150, 330, 46), (930, 520, 30), (880, 260, 18)], 1 - ease((t - 0.6) / 0.4))
    typed(fr, "СЕГОДНЯ У НАС", 50, 540, 205, t, 0.3, NAVY, "Bold")
    typed(fr, "ЛИПОФИЛИНГ", 104, 540, 268, t, 1.3)
    typed(fr, "ГРУДИ", 104, 540, 384, t, 2.15)
    return fr

def s_butt(t):
    fr = bg(BLUE); sc, sx, sy, cx, cy = cam(t); to = sticker(fr, t, sc, sx, sy, cx, cy)
    # hand-drawn loop around the buttocks
    down = ease((t - 3.1) / 0.7)                   # white wave drops from the top and hides the face: text gets a clean field
    e = wave_edge(-120 + 600 * down, 60, 1.5, 0.4 + 0.3 * (t - 3.1)); fill(fr, np.vstack([e, [[W + 10, -10], [-10, -10]]]), WHITE)
    c = np.float32(to(280, 975)); k = np.linspace(0.35, 0.35 + 2 * np.pi * 1.08, 120)
    ring = np.c_[c[0] + 330 * np.cos(k) * (1 + 0.04 * np.sin(3 * k)), c[1] + 225 * np.sin(k) * (1 + 0.05 * np.cos(2 * k))]
    stroke(fr, ring, ease((t - 4.05) / 0.7), NAVY, 9)
    low = Path(np.c_[c[0] + 445 * np.cos(np.linspace(np.pi * 0.9, np.pi * 0.1, 60)), c[1] + 345 * np.sin(np.linspace(np.pi * 0.92, np.pi * 0.08, 60))])
    curved(fr, "В КАДРЕ — ЯГОДИЦЫ", 66, low, t, 4.05, NAVY)
    typed(fr, "НЕСМОТРЯ НА ТО,", 50, 540, 205, t, 3.3, NAVY, "Bold", t1=5.5)
    typed(fr, "ЧТО", 50, 540, 270, t, 3.85, NAVY, "Bold", t1=5.5)
    # «мы просто забираем отсюда жир»: drops lift off and float up
    typed(fr, "ЗАБИРАЕМ", 50, 540, 205, t, 5.75, NAVY, "Bold")
    typed(fr, "ОТСЮДА ЖИР", 104, 540, 268, t, 6.3)
    for k, (dx, ph) in enumerate([(-150, 0), (40, .25), (170, .5), (-40, .7), (110, .95)]):
        lt = t - 6.45 - 0.13 * k
        if lt > 0:
            y = c[1] - 60 - 620 * ease(lt / 1.3); x = c[0] + dx + 18 * np.sin(lt * 4 + ph * 6)
            drop(fr, x, y, 42 * back(lt / 0.4), 1.0, WHITE, NAVY, 6)
    return fr

def trap_art(fr, t, t0, cx=540, cy=1000):
    """line-art mousetrap seen from above at a slant, a fat drop as the bait; the bar snaps on «стройная»"""
    p = lambda a, b: ease((t - t0 - a) / b)
    def sp(x, y, z=0.0): return (cx + x, cy + y * 0.5 - z * 0.9)      # plank plane -> screen
    top = np.float32([sp(-340, -150), sp(340, -150), sp(340, 150), sp(-340, 150), sp(-340, -150)])
    stroke(fr, top, p(0, .7), NAVY, 8)
    stroke(fr, np.float32([sp(-340, 150), (cx - 340, cy + 75 + 46), (cx + 340, cy + 75 + 46), sp(340, 150)]), p(.35, .45), NAVY, 8)
    for x in (-200, 0, 200): stroke(fr, np.float32([sp(x + 60, -110), sp(x + 120, -60)]), p(.6, .3), NAVY, 4)   # wood grain
    th = np.pi * ease((t - 13.05) / 0.14)                              # bar rotates about the hinge (x = 0)
    def bar(x, y): return sp(x * np.cos(th), y, -x * np.sin(th))
    U = [(0, -112), (-250, -112), (-250, 112), (0, 112)]
    stroke(fr, np.float32([bar(x, y) for x, y in np.vstack([np.linspace(U[i], U[i + 1], 12) for i in range(3)])]), p(.7, .5), NAVY, 9)
    for y in (-112, 112):                                              # springs on the hinge
        k = np.linspace(0, 6 * np.pi, 60); stroke(fr, np.c_[cx + 14 * np.cos(k), cy + y * 0.5 - 14 * np.sin(k) + np.linspace(-6, 6, 60)], p(.8, .4), NAVY, 5)
    if th < 0.1:                                                       # the catch arm holding the bar open
        stroke(fr, np.float32([sp(-250, 0, 4), sp(120, 0, 4)]), p(1.0, .3), NAVY, 6)
    plate = np.float32([sp(110, -45), sp(220, -45), sp(220, 45), sp(110, 45), sp(110, -45)]); stroke(fr, plate, p(.9, .3), NAVY, 6)
    if th > 0.05:
        for k in range(6):                                             # snap lines
            al = 1 - ease((t - 13.25) / 0.3); a2 = np.radians(-30 - 24 * k); r1 = 230 + 70 * ease((t - 13.1) / 0.25)
            if al > 0.4: stroke(fr, np.float32([[cx + 120 + 180 * np.cos(a2), cy - 30 + 120 * np.sin(a2)], [cx + 120 + r1 * np.cos(a2), cy - 30 + r1 * 0.66 * np.sin(a2)]]), 1, NAVY, 6)
    lt = t - 9.25                                                      # the bait falls onto the plate
    if lt > 0 and th < 2.6:
        y = cy - 520 + 520 * min(1, (lt / 0.45) ** 2); drop(fr, cx + 165, y - 50, 58, 1.0, BLUE, NAVY, 7)

def s_trap(t):
    fr = bg(WHITE)
    trap_art(fr, t, 7.85, 540, 1020)
    typed(fr, "ЗАБИРАЕМ ЖИР", 76, 540, 260, t, 7.95, NAVY, "ExtraBold", t1=9.0)
    typed(fr, "СО ВСЕХ", 50, 540, 230, t, 9.2, NAVY, "Bold")
    typed(fr, "ЖИРОВЫХ", 104, 540, 293, t, 10.2)
    typed(fr, "ЛОВУШЕК", 104, 540, 409, t, 10.75)
    typed(fr, "КОТОРЫЕ ЕСТЬ У ДЕВУШКИ", 40, 540, 600, t, 11.3, NAVY, "Bold")
    # «потому что она очень стройная» on a rising wave
    rise = ease((t - 12.2) / 0.7)
    if rise > 0:
        e = wave_edge(1920 + 60 - 560 * rise, 80, 1.6, 0.6 + 0.25 * (t - 12.2)); wave_fill(fr, e, BLUE)
        curved(fr, "ПОТОМУ ЧТО ОНА", 46, Path(e + [0, 95]), t, 12.35, WHITE, 0.5, 0.7)
        curved(fr, "ОЧЕНЬ СТРОЙНАЯ", 76, Path(e + [0, 185]), t, 12.85, NAVY, 0.5, 0.7)
    return fr

# body map: front mannequin, navy line art; offsets from the vertical axis X0
X0 = 540
R_OUT = [(34, 545), (40, 590), (130, 618), (152, 665), (132, 725), (118, 790), (110, 860), (92, 945), (108, 1030),
         (146, 1110), (160, 1190), (150, 1290), (120, 1395), (98, 1465), (102, 1545), (90, 1655), (58, 1790), (64, 1838), (22, 1846)]
R_IN = [(0, 1226), (24, 1290), (34, 1400), (40, 1465), (36, 1560), (30, 1700), (22, 1800), (20, 1846)]
def side(pts, s): return spline([(X0 + s * x, y) for x, y in pts])
BODY = [side(R_OUT, 1), side(R_OUT, -1), side(R_IN, 1), side(R_IN, -1)]
BREAST = [bez((X0 + s * 18, 775), (X0 + s * 22, 862), (X0 + s * 108, 868), (X0 + s * 114, 792)) for s in (1, -1)]
ZONES = {"gal": [(X0 + 150, 1190, 30, 62), (X0 - 150, 1190, 30, 62)], "in": [(X0 + 36, 1300, 22, 60), (X0 - 36, 1300, 22, 60)],
         "knee": [(X0 + 52, 1450, 34, 28), (X0 - 52, 1450, 34, 28)]}
ZT = {"gal": 14.3, "in": 14.9, "knee": 15.95}
def blob(fr, x, y, rx, ry, a):
    k = np.linspace(0, 2 * np.pi, 50); fill(fr, np.c_[x + rx * np.cos(k), y + ry * np.sin(k)], WHITE, a)
def s_map(t):
    fr = bg(BLUE)
    for k, b in enumerate(BODY): stroke(fr, b, ease((t - 13.95 - 0.05 * k) / 0.75), NAVY, 6)
    head = np.c_[X0 + 58 * np.sin(np.linspace(0, 2 * np.pi, 60)), 478 - 60 * np.cos(np.linspace(0, 2 * np.pi, 60))]
    stroke(fr, head, ease((t - 13.95) / 0.6), NAVY, 6)
    for b in BREAST: stroke(fr, b, ease((t - 14.3) / 0.5), NAVY, 6)
    # fat zones light up as they are named
    for z, t0 in ZT.items():
        if t > t0:
            for x, y, rx, ry in ZONES[z]:
                pulse = 1 + 0.12 * np.sin((t - t0) * 6) * np.exp(-(t - t0) * 1.5)
                gone = 1 - ease((t - 17.5) / 0.9)                      # the fat leaves the zones
                blob(fr, x, y, rx * back((t - t0) / 0.4) * pulse, ry * back((t - t0) / 0.4) * pulse, 0.9 * gone)
    def label(s, x, y, t0, anchor, lx, ly):
        if t < t0: return
        im = timg(s, 46); w = im.shape[1]; ey = y + im.shape[0] / 2
        stroke(fr, np.float32([[lx, ly], [x + w + 6 if anchor == "l" else x - w - 6, ey]]), ease((t - t0) / 0.25), NAVY, 4)
        k = min(len(s), int((t - t0 - 0.1) / 0.03) + 1) if t > t0 + 0.1 else 0
        if k: pad = 46 // 6; cut = pad + int(font(46).getlength(s[:k])) + (pad if k == len(s) else 0); paste(fr, im[:, :cut], x if anchor == "l" else x - w, y)
    label("ГАЛИФЕ", 70, 1158, 14.3, "l", X0 - 182, 1190)
    label("ВНУТРЕННИЕ", 1040, 1250, 14.9, "r", X0 + 60, 1290)
    label("БЁДРА", 1040, 1308, 15.25, "r", X0 + 60, 1320)
    label("КОЛЕНИ", 70, 1420, 15.95, "l", X0 - 88, 1452)
    typed(fr, "ЖИРОВЫЕ ЛОВУШКИ", 62, 540, 215, t, 14.0, NAVY, "ExtraBold", t1=16.75)
    typed(fr, "БУДЕМ ВВОДИТЬ", 50, 540, 190, t, 16.85, NAVY, "Bold")
    typed(fr, "В ОБЛАСТЬ ГРУДИ", 76, 540, 252, t, 17.75)
    # drops fly from the zones into the breasts, the breasts fill up
    srcs = [z for zz in ZONES.values() for z in zz]
    for k, (x, y, _, _) in enumerate(srcs * 2):
        lt = t - 17.2 - 0.17 * k
        if 0 < lt < 1.0:
            s = 1 if x > X0 else -1; tx, ty = X0 + s * 66, 818
            q = inout(lt / 1.0); pt = bez((x, y), (x + s * 230, (y + ty) / 2), (tx, ty), n=2)
            P = (1 - q) ** 2 * np.float32([x, y]) + 2 * (1 - q) * q * np.float32([x + s * 260, (y + ty) / 2]) + q * q * np.float32([tx, ty])
            drop(fr, P[0], P[1], 28 * (1 - 0.4 * q), 1.0, WHITE, NAVY, 4)
    full = ease((t - 17.9) / 1.4)
    for b in BREAST:
        c0 = np.float32([(b[0, 0] + b[-1, 0]) / 2, 780]); fill(fr, c0 + (b - c0) * (1 + 0.12 * full), WHITE, 0.9 * full)
    for b in BREAST: stroke(fr, b, 1 if t > 14.8 else 0, NAVY, 6)
    sparkles(fr, t, 18.9, [(X0 + 210, 720, 34), (X0 - 220, 760, 24), (X0 + 160, 900, 16)])
    return fr

def s_show(t):
    fr = bg(WHITE)
    up = 1 - back((t - 19.6) / 0.8)
    sticker(fr, t, 1.75, 380, 700, 540, 1230 + 1500 * up)
    rise = ease((t - 19.6) / 0.8)
    e = wave_edge(1920 + 60 - 560 * rise, 45, 2.0, 2.6 + 0.15 * (t - 19.6)); wave_fill(fr, e, BLUE)
    typed(fr, "КАК ОБРАБАТЫВАЕМ ЖИР,", 64, 540, 205, t, 19.75)
    typed(fr, "КАК ВВОДИМ —", 64, 540, 295, t, 21.3)
    curved(fr, "ОБЯЗАТЕЛЬНО ПОКАЖЕМ", 60, Path(e + [0, 110]), t, 23.35, WHITE, 0.5, 0.8)
    curved(fr, "ВО ВРЕМЯ ОПЕРАЦИИ", 60, Path(e + [0, 200]), t, 23.8, NAVY, 0.5, 0.8)
    sparkles(fr, t, 24.7, [(930, 560, 40), (150, 640, 28), (880, 760, 16)], 1.0, BLUE)
    return fr

SCN = {"intro": s_intro, "butt": s_butt, "trap": s_trap, "map": s_map, "show": s_show}
def scene_at(t):
    cur = CUTS[0]
    for c in CUTS:
        if t >= c[1]: cur = c
    return cur
def render(t):
    name, c = scene_at(t); i = [n for n, _ in CUTS].index(name)
    fr = SCN[name](t)
    # wave transition: the next scene rises from the bottom behind a wavy edge
    for j in range(len(CUTS) - 1):
        nm, cc = CUTS[j + 1]
        if cc - 0.45 <= t < cc:
            p = inout((t - (cc - 0.45)) / 0.45); prev = SCN[CUTS[j][0]](t); nxt = SCN[nm](cc + 0.01 * (t - cc))
            e = wave_edge(H + 120 - (H + 360) * p, 90, 1.4, 1.3 * j); mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [np.round(np.vstack([e, [[W + 10, H + 10], [-10, H + 10]]]) * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
            m = (mask.astype(np.float32) / 255)[..., None]; fr = (prev * (1 - m) + nxt * m).astype(np.uint8)
    # loop: end on the opening sparkles over blue
    if t > DUR - 0.5:
        p = inout((t - (DUR - 0.5)) / 0.5); e = wave_edge(H + 120 - (H + 360) * p, 90, 1.4, 4.0); end = bg(BLUE)
        sparkles(end, 1, 0, [(150, 330, 46), (930, 520, 30), (880, 260, 18)])
        mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [np.round(np.vstack([e, [[W + 10, H + 10], [-10, H + 10]]]) * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
        m = (mask.astype(np.float32) / 255)[..., None]; fr = (fr * (1 - m) + end * m).astype(np.uint8)
    return fr

if __name__ == "__main__":
    if len(sys.argv) > 1:
        os.makedirs("f", exist_ok=True)
        for ts in sys.argv[1:]: cv2.imwrite(f"f/o_{ts}.jpg", cv2.cvtColor(render(float(ts)), cv2.COLOR_RGB2BGR))
        sys.exit()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "video.mp4"], stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)): enc.stdin.write(render(i / FPS).tobytes())
    enc.stdin.close(); enc.wait()
