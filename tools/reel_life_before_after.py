import subprocess, sys, json
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

W, H, FPS = 1080, 1920, 30
SP = "/tmp/claude-0/-home-user-design-montazh-so-vkusom/b26b20a4-6ac5-50f8-8404-dcbad88c88d3/scratchpad/"
BLUE = (16, 46, 70); WHITE = (255, 255, 255); LIME = (214, 236, 46); PINK = (228, 210, 210); SKY = (228, 210, 210); GRAPH = (103, 96, 104)   # surgeon palette: powder / navy / graphite
from retouch import retouch
def osw(s): return ImageFont.truetype(SP + "v14/fonts/BN-Bold.ttf", s)          # Bebas Neue (Cyrillic)
def mont(w, s): return ImageFont.truetype(SP + f"fonts2/M{w}.ttf", s)
SUBS = []; DUR = 44.0
exec(open("helpers_src.py").read())

# ---------------- voice edit: (src_in, src_out, reel_start) ----------------
SEGS = [(6.50, 8.48, 0.35), (10.38, 13.96, 2.75), (17.55, 22.70, 6.6), (23.12, 25.72, 11.95), (25.80, 32.40, 14.65),
        (42.85, 45.05, 21.6), (71.48, 75.08, 24.1), (130.38, 131.98, 28.0), (132.88, 133.47, 29.75), (134.17, 137.88, 30.34),
        (36.45, 37.19, 34.35)]
def src_of(t):
    best = SEGS[0][0]
    for si, so, t0 in SEGS:
        if t0 <= t <= t0 + so - si: return si + t - t0
        if t < t0: return best
        best = so
    return best

BL = [("hook", 0, 3.3), ("talk1", 3.3, 6.45), ("talk2", 6.45, 8.85), ("talkq", 8.85, 11.85),
      ("before", 11.85, 14.6), ("talkj", 14.6, 19.6), ("talk3", 19.6, 23.9), ("surgeon", 23.9, 26.2), ("morph", 26.2, 30.2),
      ("aft1", 30.2, 32.2), ("aft2", 32.2, 34.3), ("happy", 34.3, 35.4),
      ("c1", 35.4, 38.3), ("c2", 38.3, 41.2), ("c3", 41.2, DUR)]
def block(t):
    for n, a, b in BL:
        if t < b: return n, a, b
    return BL[-1]
CUTS = [b[1] for b in BL[1:]]

# ---------------- sources ----------------
def load(p): return np.array(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
def blur_ell(img, c, ax):
    c = np.array(c, float); ax = np.array(ax, float)
    x0, y0 = np.maximum(0, (c - ax * 1.5).astype(int)); x1, y1 = np.minimum([img.shape[1], img.shape[0]], (c + ax * 1.5).astype(int))
    crop = img[y0:y1, x0:x1].astype(np.float32); bl = cv2.GaussianBlur(cv2.resize(crop, None, fx=0.15, fy=0.15), (0, 0), 5)
    bl = cv2.resize(bl, (crop.shape[1], crop.shape[0]))
    m = np.zeros(crop.shape[:2], np.float32); cv2.ellipse(m, (int(c[0] - x0), int(c[1] - y0)), (int(ax[0]), int(ax[1])), 0, 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(ax) / 6)[..., None]
    img = img.copy(); img[y0:y1, x0:x1] = (crop * (1 - m) + bl * m).astype(np.uint8); return img
B80, B81, B82 = load("dl/IMG_7580_edit.jpg"), load("dl/IMG_7581_edit.jpg"), load("dl/IMG_7582_edit.jpg")
A84 = retouch(blur_ell(load("dl/IMG_7584.PNG"), (525, 160), (95, 125)))          # patient face blurred
A85 = retouch(blur_ell(load("dl/IMG_7585.PNG"), (520, 150), (95, 120)))
SURG = load("dl/DSC01189.JPG")
C1, C2, C3 = load("dl/IMG_7595.JPG"), load("dl/collage_34.jpg"), load("dl/collage_side.jpg")
for _c in (C1, C2, C3): _c[:958] = retouch(np.ascontiguousarray(_c[:958]))      # retouch the ПОСЛЕ halves

class Src:
    def __init__(s, p): s.cap = cv2.VideoCapture(p); s.pos = -1; s.c = {}
    def get(s, i):
        if i in s.c: return s.c[i]
        if i != s.pos + 1: s.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = s.cap.read(); s.pos = i
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if len(s.c) > 40: s.c.pop(next(iter(s.c)))
        s.c[i] = f; return f
IV = Src("iv.mov")

def warp(img, z, cx, cy, ox=540, oy=960, size=(W, H)):
    M = np.float32([[z, 0, ox - z * cx], [0, z, oy - z * cy]])
    return cv2.warpAffine(img, M, size, flags=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(16, 16, 18)).astype(np.float32) / 255
def fillp(img, z=1.0, fx=0.5, fy=0.5):
    s = max(W / img.shape[1], H / img.shape[0]) * z
    return warp(img, s, img.shape[1] * fx, img.shape[0] * fy)
def fitp(img, z=1.0, cy=0.5):
    bg = fillp(img, 1.0) ; bg = cv2.GaussianBlur(bg, (0, 0), 30) * 0.35
    s = W / img.shape[1] * z * 0.98
    fg = warp(img, s, img.shape[1] / 2, img.shape[0] * cy, 540, 900)
    m = warp(np.full(img.shape[:2], 255, np.uint8), s, img.shape[1] / 2, img.shape[0] * cy, 540, 900)[..., :1] if False else None
    msk = cv2.warpAffine(cv2.GaussianBlur(cv2.copyMakeBorder(np.full((img.shape[0] - 40, img.shape[1] - 40), 255, np.uint8), 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0), (0, 0), 10),
                         np.float32([[s, 0, 540 - s * img.shape[1] / 2], [0, s, 900 - s * img.shape[0] * cy]]), (W, H)).astype(np.float32)[..., None] / 255
    return fg * msk + bg * (1 - msk)
def iframe(t, z=1.0, cy=900):
    f = IV.get(int(round(src_of(t) * 60))); return warp(f, z, 540, cy, 540, 960 if z <= 1.0 else 900)

# ---------------- look ----------------
yy_, xx_ = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.clip((np.arange(H) - 820) / 700, 0, 1); HAZE = (0.78 * _r ** 1.3)[:, None, None].astype(np.float32)
EDGE = np.clip((((xx_ - 540) / 560) ** 2 + ((yy_ - 900) / 1050) ** 2) ** 0.9, 0, 1)[..., None]
VIG = 1 - 0.28 * EDGE
def look(fr, t):
    g = fr @ np.float32([0.299, 0.587, 0.114]); fr = fr * 0.9 + g[..., None] * 0.1
    fr = np.clip((fr - 0.5) * 1.06 + 0.49, 0, 1) * VIG
    k = 0.0                                                     # dark light pulse on every cut
    for c in CUTS:
        if c <= t < c + 0.6: k = max(k, 0.45 * np.exp(-(t - c) / 0.22))
    return fr * (1 - k * EDGE)
def trans(t, t0, s=0.07): return 1 + s * (1 - ease((t - t0) / 0.3))
def mblur(fr, lt):
    if lt > 0.1: return fr
    k = int(40 * (1 - lt / 0.1)) | 1
    return cv2.blur(fr, (k, 1)) if k > 2 else fr

# ---------------- text ----------------
def label(txt, col):
    f = osw(54); asc, desc = f.getmetrics(); im = Image.new("RGBA", (int(f.getlength(txt)) + 10, asc + desc + 6), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((5, 3 + asc), txt, font=f, fill=col + (255,), anchor="ls"); a = to_np(im)
    return a
CH_DO, CH_PO = label("ДО", WHITE), label("ПОСЛЕ", PINK)
def chip(fr, im, x, y, t, t0, t1=1e9):
    if t < t0 or t > t1 + 0.25: return
    p = ease((t - t0) / 0.35); al = min(1, (t - t0) / 0.1) * (1 - min(1, max(0, (t - t1) / 0.25)))
    ablend(fr, im, x - (1 - p) * 140, y, al, *((PINK, GRAPH) if im is CH_PO else (WHITE, BLUE)))
BRC = {}
def bracket(frame, t, t0, lines, yc, size=78, t1=1e9):
    """two bars open from the centre and reveal the text, lime underline under the last line"""
    if t < t0 or t > t1 + 0.3: return
    key = (tuple(lines), size)
    if key not in BRC:
        f = osw(size); asc, desc = f.getmetrics(); lh = int((asc + desc) * 0.95); bw = int(max(f.getlength(l) for l in lines))
        im = Image.new("RGBA", (bw + 40, lh * len(lines) + 30), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        for k, l in enumerate(lines): d.text(((bw + 40) / 2, 8 + k * lh + asc), l, font=f, fill=WHITE + (255,), anchor="ms")
        lw = f.getlength(lines[-1]); BRC[key] = (to_np(im), ((bw + 40 - lw) / 2, (bw + 40 + lw) / 2, 8 + (len(lines) - 1) * lh + asc + 10))
    im, (u0, u1, uy) = BRC[key]; h, w = im.shape[:2]; x0 = 540 - w / 2; y0 = yc - h / 2
    fade = 1 - min(1, max(0, (t - t1) / 0.3)); lt = t - t0
    grow = ease(lt / 0.18); op = ease((lt - 0.15) / 0.45); half = 6 + op * (w / 2 + 10)
    rev = im * (np.abs(np.arange(w)[None, :, None] + x0 - 540) < half - 4).astype(np.float32)
    blend(frame, shadow_of(rev, 10, 0.65), x0, y0 + 4, fade); blend(frame, rev, x0, y0, fade)
    bh = h * grow
    for sx in (-1, 1):
        r = np.ones((int(max(bh, 2)), 6, 4), np.float32); blend(frame, r, 540 + sx * half - 3, yc - bh / 2, fade * (1 - 0.35 * op))
    up = ease((lt - 0.6) / 0.35)
    if up > 0:
        r = np.zeros((7, int(max((u1 - u0) * up, 1)), 4), np.float32); r[..., :3] = np.float32(LIME) / 255; r[..., 3] = 1
        blend(frame, r, x0 + u0, y0 + uy, fade)
TC = {}
def bullet(fr, t, t0, txt, x, y, size=38, t1=1e9):
    if t < t0 or t > t1 + 0.25: return
    key = (txt, size)
    if key not in TC:
        f = mont(600, size); asc, desc = f.getmetrics(); tw = int(f.getlength(txt)); hh = asc + desc
        im = Image.new("RGBA", (tw + 80, hh + 16), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        c = hh // 2
        d.polygon([(c - 10, c - 1), (c - 10, c + 17), (c + 12, c + 8)], fill=PINK + (255,))
        d.text((hh + 18, 8 + asc), txt, font=f, fill=WHITE + (255,), anchor="ls"); TC[key] = to_np(im)
    im = TC[key]; p = ease((t - t0) / 0.3); al = min(1, (t - t0) / 0.1) * (1 - min(1, max(0, (t - t1) / 0.25)))
    ablend(fr, im, x - 70 * (1 - p), y, al, WHITE, BLUE)

DEFER = [False]; PENDING = []
LIGHT = [False]; ROSE = (184, 128, 146)
def ablend(fr, img, x, y, al, c_dark, c_light):
    """text layer: deferred so it is drawn on top of the grade and the dark haze"""
    if DEFER[0]: PENDING.append((img, x, y, al, c_dark, c_light)); return
    if LIGHT[0]: c_dark = {WHITE: BLUE, PINK: ROSE}.get(tuple(c_dark), c_dark)
    if al <= 0.003: return
    h, w = img.shape[:2]; x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    a = img[y0 - y:y1 - y, x0 - x:x1 - x, 3:4] * al; d = fr[y0:y1, x0:x1]
    lum = d @ np.float32([0.299, 0.587, 0.114]); wa = a[..., 0]
    light = (lum * wa).sum() / max(wa.sum(), 1e-3) > 0.5                 # one colour per line, chosen by what is mostly behind it
    col = np.float32(c_dark) / 255                                        # always white / powder
    d *= 1 - a; d += col * a
class Title:
    """lines rise out of a mask with a blur-in; the key line sits on a pink plate that wipes in (navy text)"""
    def __init__(s, lines, size=78):
        s.items = []
        for txt, kind in lines:
            f = osw(size) if kind in ("H", "P", "B") else mont(500, 34); asc, desc = f.getmetrics(); tw = int(f.getlength(txt))
            hh = asc + desc + 4; ims = {}
            for nm, col in (("w", WHITE), ("powder", PINK), ("navy", BLUE), ("graph", GRAPH)):
                im = Image.new("RGBA", (tw + 8, hh), (0, 0, 0, 0)); ImageDraw.Draw(im).text((4, 2 + asc), txt, font=f, fill=col + (255,), anchor="ls"); ims[nm] = to_np(im)
            s.items.append(dict(kind=kind, ims=ims, w=tw + 8, h=hh))
    def draw(s, fr, y, t, t0, t1=1e9, cps=None):
        if t < t0 or t > t1 + 0.3: return
        out = 1 - ease((t - t1) / 0.3) if t > t1 else 1.0; yy = y
        for k, it in enumerate(s.items):
            ti = t0 + k * 0.16; p = ease((t - ti) / 0.45)
            if p <= 0: yy += it["h"] + 6; continue
            x = 540 - it["w"] / 2
            im = it["ims"]["w"]; cols = (PINK, GRAPH) if it["kind"] in ("P", "B") else (WHITE, BLUE)
            sig = (1 - p) * 7
            imb = cv2.GaussianBlur(im, (0, 0), sig) if sig > 0.4 else im
            dy = int((1 - p) * it["h"] * 0.7); vis = imb[:it["h"] - dy] if dy > 0 else imb      # rise from the line's own mask
            ablend(fr, vis, x, yy + dy, out * p, *cols)
            yy += it["h"] + 6

T_HOOK = Title([("ЖИЗНЬ РАЗДЕЛИЛАСЬ", "H"), ("НА ДО И ПОСЛЕ", "P")], 92)
T_CONF = Title([("УВЕРЕННОСТЬ В СЕБЕ", "B"), ("это дорогого стоит", "S")], 76)
T_COVER = Title([("ВСЁ ВРЕМЯ", "H"), ("ПРИКРЫВАЛАСЬ ОДЕЖДОЙ", "B")], 76)
T_LIKE = Title([("НРАВИШЬСЯ СЕБЕ РАЗДЕТОЙ", "H"), ("БОЛЬШЕ, ЧЕМ ОДЕТОЙ", "P")], 72)
T_JOY = Title([("УДОВОЛЬСТВИЕ", "H"), ("ОТ СВОЕГО ТЕЛА", "P")], 86)
T_TRIP = Title([("ПРИЕХАЛА ИЗДАЛЕКА", "B"), ("из Саратовской области", "S")], 76)
T_OP = Title([("ОЧЕНЬ СЛОЖНАЯ", "H"), ("И ДЛИННАЯ ОПЕРАЦИЯ", "B")], 78)
T_RES = Title([("РЕЗУЛЬТАТ", "H"), ("ПОТРЯСАЮЩИЙ", "P")], 92)
T_HAPPY = Title([("ОЧЕНЬ ДОВОЛЬНА", "P")], 84)

# ---------------- word-by-word subtitles ----------------
WR = json.load(open("words.json")); toks = []
for a, b, w in WR:
    if w.startswith("-") and toks: toks[-1][2] += w; continue
    toks.append([a, b, w])
FIX = {"далека": "издалека", "саратской": "саратовской", "все": "всё"}
SUB = []
for si, so, t0 in SEGS:
    for k, (a, b, w) in enumerate(toks):
        if b > si + 0.1 and a > si - 0.3 and a < so - 0.08:
            w = w.lower().strip(".,!?«»—…").strip()
            if not w: continue
            if w == "из" and k + 1 < len(toks) and toks[k + 1][2].lower().startswith("далека"): continue
            SUB.append([t0 + max(0, a - si), t0 + min(b, so) - si, FIX.get(w, w)])
for i in range(len(SUB)):
    nxt = SUB[i + 1][0] if i + 1 < len(SUB) else DUR
    SUB[i][1] = min(nxt, max(SUB[i][1], SUB[i][0] + 0.3) + 0.25)
SC = {}
def subs(frame, t, y=1530):
    for t0, t1, w in SUB:
        if t0 <= t < t1:
            if w not in SC:
                f = mont(500, 46); asc, desc = f.getmetrics(); tw = int(f.getlength(w))
                im = Image.new("RGBA", (tw + 20, asc + desc + 10), (0, 0, 0, 0)); ImageDraw.Draw(im).text((10, 5 + asc), w, font=f, fill=WHITE + (255,), anchor="ls")
                a = to_np(im); b = a.copy(); b[..., :3] = a[..., 3:4] * (np.float32(BLUE) / 255); SC[w] = (a, b)
            im, imn = SC[w]
            p = ease((t - t0) / 0.12); al = min(1, (t - t0) / 0.06)
            ablend(frame, im, 540 - im.shape[1] / 2, y - im.shape[0] / 2 + 12 * (1 - p), al, WHITE, BLUE)

# front collage halves aligned on the stickers (ПОСЛЕ top half, ДО bottom half)
AFT_H, BEF_H = C1[:955], B80                    # ДО: full original photo (more of the body than the collage half)
def half_view(img, sl, sr, z):
    mx, my = (sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2; s = SEP / np.hypot(sr[0] - sl[0], sr[1] - sl[1]) * z
    out = warp(img, s, mx, my, 540, 760)
    msk = warp(np.full(img.shape[:2], 255, np.uint8), s, mx, my, 540, 760)[..., None] if False else None
    return out
def feather_mask(shape_hw, sl, sr, z):
    m = np.full(shape_hw, 255, np.uint8); m = cv2.GaussianBlur(cv2.copyMakeBorder(m[12:-12, 12:-12], 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0), (0, 0), 7)
    mx, my = (sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2; s = SEP / np.hypot(sr[0] - sl[0], sr[1] - sl[1]) * z
    M = np.float32([[s, 0, 540 - s * mx], [0, s, 760 - s * my]])
    return cv2.warpAffine(m, M, (W, H)).astype(np.float32)[..., None] / 255
SEP = 487                       # sticker distance on screen: ПОСЛЕ half shown at full width, nothing cropped
SA, SB = ((281, 288), (768, 300)), ((267, 528), (852, 479))

def morph_frame(t, tw, z):
    a_ = warp(BEF_H, SEP / np.hypot(SB[1][0] - SB[0][0], SB[1][1] - SB[0][1]) * z, (SB[0][0] + SB[1][0]) / 2, (SB[0][1] + SB[1][1]) / 2, 540, 760)
    b_ = warp(AFT_H, SEP / np.hypot(SA[1][0] - SA[0][0], SA[1][1] - SA[0][1]) * z, (SA[0][0] + SA[1][0]) / 2, (SA[0][1] + SA[1][1]) / 2, 540, 760)
    ma, mb = feather_mask(BEF_H.shape[:2], *SB, z), feather_mask(AFT_H.shape[:2], *SA, z)
    bg = np.float32([16, 16, 18]) / 255
    a_ = a_ * ma + bg * (1 - ma); b_ = b_ * mb + bg * (1 - mb)
    p = ease((t - tw) / 1.0); f = 0.35; edge = (xx_ / W * 0.6 + yy_ / H * 0.4)[..., None]
    m = np.clip((p * (1 + f) - edge) / f, 0, 1); return a_ * (1 - m) + b_ * m
def collage(img, t, t0):
    lt = t - t0; fr = warp(img, (1 + 0.02 * lt) * trans(t, t0, 0.06), 540, 960); fr = mblur(fr, lt)
    fr[957:963] = np.maximum(fr[957:963], min(1, lt / 0.4))
    chip(fr, CH_PO, 40, 228, t, t0 + 0.25); chip(fr, CH_DO, 40, 990, t, t0 + 0.4); return fr

LOGO_D = LOGO.copy(); LOGO_D[..., :3] = LOGO[..., 3:4] * (np.float32(BLUE) / 255)
# ---------------- light (white) layouts ----------------
LIGHT = [False]
ROSE = (184, 128, 146)                                         # powder deepened so it reads on white
_g = (yy_ / H)[..., None]
WHITE_BG = (np.float32([250, 247, 246]) / 255 * (1 - _g) + np.float32([236, 222, 222]) / 255 * _g).astype(np.float32)
def white_bg(t):
    fr = WHITE_BG.copy()
    for k, (cx, cy, r) in enumerate([(160, 380, 260), (930, 1500, 330)]):        # soft powder blobs drifting
        x = cx + 40 * np.sin(t * 0.7 + k); y = cy + 30 * np.cos(t * 0.6 + k)
        g = np.exp(-(((xx_ - x) / r) ** 2 + ((yy_ - y) / r) ** 2))[..., None]
        fr = fr * (1 - 0.35 * g) + np.float32(PINK) / 255 * 0.35 * g
    return fr
def ring(fr, cx, cy, r, prog, col=BLUE, th=7, start=-90):
    if prog <= 0: return
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(r), int(r)), 0, start, start + 360 * min(prog, 1), 255, th, cv2.LINE_AA)
    a = m.astype(np.float32)[..., None] / 255; fr *= 1 - a; fr += np.float32(col) / 255 * a
def face_disc(t, r, zoom=1.0):
    f = IV.get(int(round(src_of(t) * 60))); half = 430 / zoom
    crop = f[int(720 - half):int(720 + half), int(540 - half):int(540 + half)]
    return cv2.resize(crop, (2 * int(r), 2 * int(r)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
def put_disc(fr, img, cx, cy, al=1.0, shadow=True):
    r = img.shape[0] // 2; m = np.zeros((2 * r, 2 * r), np.uint8); cv2.circle(m, (r, r), r - 1, 255, -1, cv2.LINE_AA)
    rgba = np.concatenate([img * (m[..., None] / 255.0), m[..., None] / 255.0], -1).astype(np.float32)
    if shadow: blend(fr, shadow_of(rgba, 22, 0.35), cx - r, cy - r + 14, al)
    blend(fr, rgba, cx - r, cy - r, al)
def circle_scene(fr, t, t0, cx, cy, r):
    """interview face in a circle: pops in with a bounce, navy ring draws around, powder ring pulses"""
    lt = t - t0; s = max(back(lt / 0.45), 0.02); rr = int(r * s)
    if rr < 4: return
    put_disc(fr, face_disc(t, rr), cx, cy + 6 * np.sin(t * 1.8))
    ring(fr, cx, cy + 6 * np.sin(t * 1.8), rr + 14, ease((lt - 0.15) / 0.7))
    pul = (lt * 0.8) % 1
    ring(fr, cx, cy, rr + 24 + 60 * pul, 1.0, col=tuple(int(c) for c in np.float32(PINK) * (1 - pul) + np.float32([250, 247, 246]) * pul), th=4)
def card(fr, img, t, t0, cx, cy, w, rot0=0.0, frm=(0, 900), al=1.0):
    """photo as a rounded card with shadow, flying in"""
    lt = t - t0
    if lt < 0: return
    p = ease(lt / 0.45); h = int(img.shape[0] * w / img.shape[1])
    im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    m = np.zeros((h, w), np.uint8); cv2.rectangle(m, (0, 0), (w - 1, h - 1), 255, -1)
    m = cv2.GaussianBlur(m, (0, 0), 1.2); mm = np.zeros_like(m); cv2.rectangle(mm, (18, 18), (w - 19, h - 19), 255, -1)
    mm = cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)))
    rgba = np.concatenate([im * (mm[..., None] / 255.0), mm[..., None] / 255.0], -1).astype(np.float32)
    ang = rot0 * (1 - 0.6 * p); sc = 0.85 + 0.15 * p
    pad = int(max(w, h) * 0.3); big = cv2.copyMakeBorder(rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    Mr = cv2.getRotationMatrix2D((big.shape[1] / 2, big.shape[0] / 2), ang, sc); big = cv2.warpAffine(big, Mr, (big.shape[1], big.shape[0]))
    x = cx - big.shape[1] / 2 + frm[0] * (1 - p); y = cy - big.shape[0] / 2 + frm[1] * (1 - p)
    blend(fr, shadow_of(big, 26, 0.38), x, y + 18, al * p); blend(fr, big, x, y, al * min(1, p * 2))
def route(fr, t, t0, y=1290):
    """animated route Саратов -> Москва"""
    lt = t - t0
    if lt < 0: return
    a, b = np.array([170, y + 60]), np.array([910, y - 40])
    pts = bez(a, b, -0.25, 60); p = ease(lt / 1.2)
    stroke(fr, partial(pts, p), 1.0, th=5, dash=True, col=BLUE)
    for P, txt, tt in ((a, "САРАТОВ", 0.0), (b, "МОСКВА", 1.0)):
        if lt >= tt:
            q = back((lt - tt) / 0.35); cv2.circle(fr, tuple(int(v) for v in P), int(16 * q), tuple(float(c) / 255 for c in BLUE), -1, cv2.LINE_AA)
            cv2.circle(fr, tuple(int(v) for v in P), int(7 * q), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
            if ("rt", txt) not in TC:
                f = osw(52); asc, desc = f.getmetrics(); im = Image.new("RGBA", (int(f.getlength(txt)) + 10, asc + desc + 6), (0, 0, 0, 0))
                ImageDraw.Draw(im).text((5, 3 + asc), txt, font=f, fill=WHITE + (255,), anchor="ls"); TC[("rt", txt)] = to_np(im)
            im = TC[("rt", txt)]; ablend(fr, im, P[0] - im.shape[1] / 2, P[1] + 26, min(1, (lt - tt) / 0.2), WHITE, BLUE)
    if 0 < p < 1:
        q = partial(pts, p)[-1]; cv2.circle(fr, (int(q[0]), int(q[1])), 11, tuple(float(c) / 255 * 0.85 for c in PINK), -1, cv2.LINE_AA)
def dark_to_circle(t, t0, cx, cy, r, dur=0.45):
    """first moments of a white scene: the full-screen interview shrinks into the circle"""
    p = ease((t - t0) / dur)
    if p >= 1: return None
    fr = white_bg(t); R = int(r + (1300 - r) * (1 - p))
    full = iframe(t, 1.0)
    m = np.zeros((H, W), np.float32); cv2.circle(m, (int(540 + (cx - 540) * p), int(900 + (cy - 900) * p)), R, 1, -1, cv2.LINE_AA)
    return fr * (1 - m[..., None]) + full * m[..., None]
def circle_to_full(t, t0, z, cy_src, cx=540, cy=760, r=300, dur=0.45):
    """first moments of a dark scene: the circle grows until the interview fills the frame"""
    p = ease((t - t0) / dur)
    full = iframe(t, z, cy_src)
    if p >= 1: return full
    fr = white_bg(t); R = int(r + (1300 - r) * p)
    m = np.zeros((H, W), np.float32); cv2.circle(m, (int(cx + (540 - cx) * p), int(cy + (900 - cy) * p)), R, 1, -1, cv2.LINE_AA)
    return fr * (1 - m[..., None]) + full * m[..., None]
def push_in(fr_new, t, t0, dur=0.35, frm=1):
    p = ease((t - t0) / dur)
    if p >= 1: return fr_new
    out = np.zeros_like(fr_new); dx = int(W * (1 - p)) * frm
    if frm > 0: out[:, dx:] = fr_new[:, :W - dx]
    else: out[:, :W + dx] = fr_new[:, -dx:]
    return mblur(out, 0.0)

LIGHT_BLOCKS = ("talk1", "talkq", "before", "talk3", "aft1", "aft2", "happy")
def render(t):
    n, t0, t1 = block(t); lt = t - t0; flash = 0; DEFER[0] = True; PENDING.clear(); LIGHT[0] = n in LIGHT_BLOCKS
    if n == "hook":
        fr = morph_frame(t, 1.6, 1.0 + 0.04 * t / 3.3)
        chip(fr, CH_DO, 60, 228, t, 0.1, 2.05); chip(fr, CH_PO, 60, 228, t, 2.45)
    elif n == "talk1":                                                  # white sheet wipes up, face circle pops
        fr = white_bg(t); circle_scene(fr, t, t0 + 0.15, 540, 760, 300)
        p = ease(lt / 0.4)
        if p < 1:
            prev = morph_frame(t0, 1.6, 1.04); cut = int(H * (1 - p)); fr[:cut] = prev[:cut]
        T_CONF.draw(fr, 1180, t, t0 + 0.6, t1 - 0.05)
    elif n == "talk2":
        fr = circle_to_full(t, t0, 1.32 * (1 + 0.03 * lt / 3), 820)
    elif n == "talkq":
        fr = dark_to_circle(t, t0, 540, 700, 260)
        if fr is None:
            fr = white_bg(t); circle_scene(fr, t, t0 + 0.45 - 0.45, 540, 700, 260)
        T_LIKE.draw(fr, 1120, t, t0 + 0.5, t1 - 0.1)
    elif n == "before":                                                 # ДО photos as cards flying onto white
        fr = white_bg(t)
        card(fr, B80, t, t0, 540, 860, 640, rot0=-8, frm=(-700, 0))
        card(fr, B81, t, 12.75, 560, 860, 640, rot0=7, frm=(700, 0))
        card(fr, B82, t, 13.7, 540, 860, 640, rot0=-4, frm=(0, 1100))
        chip(fr, CH_DO, 60, 228, t, t0 + 0.05)
        T_COVER.draw(fr, 1330, t, t0 + 0.15, t1 - 0.05)
    elif n == "talkj":
        fr = push_in(mblur(iframe(t, 1.38 * (1 + 0.03 * lt / 5), 820), lt), t, t0)
        T_JOY.draw(fr, 1240, t, 17.9, t1 - 0.05)
    elif n == "talk3":
        fr = dark_to_circle(t, t0, 540, 640, 250)
        if fr is None:
            fr = white_bg(t); circle_scene(fr, t, t0, 540, 640, 250)
        route(fr, t, 21.6)
        T_TRIP.draw(fr, 1000, t, 21.75, t1 - 0.05)
    elif n == "surgeon":
        fr = mblur(fillp(SURG, (1.12 + 0.07 * lt / 2.3) * trans(t, t0, 0.12), 0.5, 0.38), lt); flash = 0.45 * max(0, 1 - lt / 0.18)
        T_OP.draw(fr, 1210, t, t0 + 0.3, t1 - 0.05)
    elif n == "morph":
        fr = morph_frame(t, 26.7, 1.0 + 0.05 * lt / 4)
        chip(fr, CH_DO, 60, 228, t, t0 + 0.05, 27.2); chip(fr, CH_PO, 60, 228, t, 27.6)
        T_RES.draw(fr, 1240, t, 27.8, t1 - 0.05)
    elif n in ("aft1", "aft2"):                                         # ПОСЛЕ as a big card on white
        fr = white_bg(t)
        if n == "aft1": card(fr, A84, t, t0, 540, 760, 740, rot0=6, frm=(0, 1200))
        else:
            card(fr, A84, t, 30.2, 540, 760, 740, rot0=6, frm=(0, 1200), al=1 - ease(lt / 0.3))
            card(fr, A85, t, t0, 540, 760, 740, rot0=-6, frm=(800, 0))
        chip(fr, CH_PO, 60, 228, t, 30.3)
        bullet(fr, t, 30.45, "через неделю — уже дома", 70, 1345)
        bullet(fr, t, 31.0, "через 3 дня — снова на работе", 70, 1415)
    elif n == "happy":
        fr = white_bg(t); circle_scene(fr, t, t0, 540, 760, 320)
        T_HAPPY.draw(fr, 1210, t, t0 + 0.05, t1)
    else:
        fr = collage({"c1": C1, "c2": C2, "c3": C3}[n], t, t0)
    if t < 3.3: T_HOOK.draw(fr, 1150, t, 0.2, 3.2)
    dark = not LIGHT[0] and n not in ("c1", "c2", "c3")
    if not LIGHT[0]: fr = look(fr, t)
    if dark: fr = fr * (1 - HAZE)
    if n not in ("c1", "c2", "c3"): subs(fr, t)
    DEFER[0] = False
    for args in PENDING: ablend(fr, *args)
    blend(fr, LOGO_D if LIGHT[0] else LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
    if flash > 0: fr = fr * (1 - flash) + flash
    if t > DUR - 0.8: fr *= 1 - 0.95 * (t - (DUR - 0.8)) / 0.8
    return np.clip(fr, 0, 1)

TYPED = [(0.2, 24, "ЖИЗНЬ РАЗДЕЛИЛАСЬНА ДО И ПОСЛЕ"), (3.45, 30, "УВЕРЕННОСТЬ В СЕБЕэто дорогого стоит"), (12.0, 34, "ВСЁ ВРЕМЯПРИКРЫВАЛАСЬ ОДЕЖДОЙ"),
         (17.9, 30, "УДОВОЛЬСТВИЕОТ СВОЕГО ТЕЛА"), (21.75, 32, "ПРИЕХАЛА ИЗДАЛЕКАиз Саратовской области"), (24.3, 34, "ОЧЕНЬ СЛОЖНАЯИ ДЛИННАЯ ОПЕРАЦИЯ"),
         (27.8, 40, "РЕЗУЛЬТАТПОТРЯСАЮЩИЙ"), (34.35, 34, "ОЧЕНЬ ДОВОЛЬНА")]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for ts in sys.argv[1:]:
            cv2.imwrite(f"f/o_{ts}.jpg", cv2.cvtColor((render(float(ts)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        sys.exit()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "video.mp4"], stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        enc.stdin.write((render(i / FPS) * 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
