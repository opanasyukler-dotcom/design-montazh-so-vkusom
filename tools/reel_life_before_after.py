import subprocess, sys, json
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

W, H, FPS = 1080, 1920, 30
SP = "/tmp/claude-0/-home-user-design-montazh-so-vkusom/b26b20a4-6ac5-50f8-8404-dcbad88c88d3/scratchpad/"
BLUE = (16, 46, 70); WHITE = (255, 255, 255); LIME = (214, 236, 46); PINK = (243, 182, 202); SKY = (150, 182, 232)
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

BL = [("hook", 0, 2.0), ("hook2", 2.0, 2.75), ("talk1", 2.75, 6.45), ("talk2", 6.45, 8.85), ("talkq", 8.85, 11.85),
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
    blend(fr, im, x - (1 - p) * 140, y, al)
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
    blend(fr, im, x - 70 * (1 - p), y, al)

class Title:
    """lines rise out of a mask with a blur-in; the key line sits on a pink plate that wipes in (navy text)"""
    def __init__(s, lines, size=78):
        s.items = []
        for txt, kind in lines:
            f = osw(size) if kind in ("H", "P", "B") else mont(500, 34); asc, desc = f.getmetrics(); tw = int(f.getlength(txt))
            hh = asc + desc + 4; col = {"P": PINK, "B": SKY}.get(kind, WHITE)
            im = Image.new("RGBA", (tw + 8, hh), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
            d.text((4, 2 + asc), txt, font=f, fill=col + (255,), anchor="ls")
            s.items.append(dict(kind=kind, im=to_np(im), w=im.size[0], h=hh))
    def draw(s, fr, y, t, t0, t1=1e9, cps=None):
        if t < t0 or t > t1 + 0.3: return
        out = 1 - ease((t - t1) / 0.3) if t > t1 else 1.0; yy = y
        for k, it in enumerate(s.items):
            ti = t0 + k * 0.16; p = ease((t - ti) / 0.45)
            if p <= 0: yy += it["h"] + 6; continue
            x = 540 - it["w"] / 2; im = it["im"]
            sig = (1 - p) * 7
            imb = cv2.GaussianBlur(im, (0, 0), sig) if sig > 0.4 else im
            dy = int((1 - p) * it["h"] * 0.7); vis = imb[:it["h"] - dy] if dy > 0 else imb      # rise from the line's own mask
            blend(fr, vis, x, yy + dy, out * p)
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
                a = to_np(im); SC[w] = (a, shadow_of(a, 7, 0.9))
            im, sh = SC[w]; p = ease((t - t0) / 0.12); al = min(1, (t - t0) / 0.06)
            blend(frame, im, 540 - im.shape[1] / 2, y - im.shape[0] / 2 + 12 * (1 - p), al)

# front collage halves aligned on the stickers (ПОСЛЕ top half, ДО bottom half)
AFT_H, BEF_H = C1[:955], C1[965:]
def half_view(img, sl, sr, z):
    mx, my = (sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2; s = 620 / np.hypot(sr[0] - sl[0], sr[1] - sl[1]) * z
    out = warp(img, s, mx, my, 540, 760)
    msk = warp(np.full(img.shape[:2], 255, np.uint8), s, mx, my, 540, 760)[..., None] if False else None
    return out
def feather_mask(shape_hw, sl, sr, z):
    m = np.full(shape_hw, 255, np.uint8); m = cv2.GaussianBlur(cv2.copyMakeBorder(m[40:-40, 40:-40], 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=0), (0, 0), 25)
    mx, my = (sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2; s = 620 / np.hypot(sr[0] - sl[0], sr[1] - sl[1]) * z
    M = np.float32([[s, 0, 540 - s * mx], [0, s, 760 - s * my]])
    return cv2.warpAffine(m, M, (W, H)).astype(np.float32)[..., None] / 255
SA, SB = ((281, 288), (768, 300)), ((257, 1243 - 965), (852, 1200 - 965))

def collage(img, t, t0):
    lt = t - t0; fr = warp(img, (1 + 0.02 * lt) * trans(t, t0, 0.06), 540, 960); fr = mblur(fr, lt)
    fr[957:963] = np.maximum(fr[957:963], min(1, lt / 0.4))
    chip(fr, CH_PO, 40, 228, t, t0 + 0.25); chip(fr, CH_DO, 40, 990, t, t0 + 0.4); return fr

def render(t):
    n, t0, t1 = block(t); lt = t - t0; flash = 0
    if n == "hook":
        fr = fitp(B80, 1.0 + 0.03 * t); chip(fr, CH_DO, 60, 228, t, 0.1)
    elif n == "hook2":
        fr = mblur(fitp(B81, 1.0 * trans(t, t0, 0.06)), lt); chip(fr, CH_DO, 60, 228, t, t0)
    elif n.startswith("talk") or n == "happy":
        z = dict(talk1=1.0, talk2=1.32, talkq=1.12, talkj=1.38, talk3=1.15, happy=1.25)[n] * (1 + 0.03 * lt / 3) * trans(t, t0)
        fr = mblur(iframe(t, z, 820 if z > 1.2 else 900), lt)
        if n == "talk1": T_CONF.draw(fr, 1180, t, t0 + 0.7, t1 - 0.05, cps=30)
        if n == "talkq": T_LIKE.draw(fr, 1200, t, t0 + 0.15, t1 - 0.1)
        if n == "talkj": T_JOY.draw(fr, 1240, t, 17.9, t1 - 0.05, cps=30)
        if n == "talk3": T_TRIP.draw(fr, 1180, t, 21.75, t1 - 0.05, cps=32)
        if n == "happy": T_HAPPY.draw(fr, 1250, t, t0 + 0.05, t1, cps=34)
    elif n == "before":
        img, s0 = (B80, t0) if t < 12.75 else ((B81, 12.75) if t < 13.7 else (B82, 13.7))
        fr = mblur(fitp(img, (1.0 + 0.03 * (t - s0)) * trans(t, s0, 0.05)), t - s0); chip(fr, CH_DO, 60, 228, t, t0 + 0.05)
        T_COVER.draw(fr, 1170, t, t0 + 0.15, t1 - 0.05, cps=34)
    elif n == "morph":
        # first ПОСЛЕ: smooth ДО -> ПОСЛЕ on the front collage halves aligned on the stickers
        z = 1.0 + 0.05 * lt / 4
        a_ = warp(BEF_H, 620 / np.hypot(SB[1][0] - SB[0][0], SB[1][1] - SB[0][1]) * z, (SB[0][0] + SB[1][0]) / 2, (SB[0][1] + SB[1][1]) / 2, 540, 760)
        b_ = warp(AFT_H, 620 / np.hypot(SA[1][0] - SA[0][0], SA[1][1] - SA[0][1]) * z, (SA[0][0] + SA[1][0]) / 2, (SA[0][1] + SA[1][1]) / 2, 540, 760)
        ma, mb = feather_mask(BEF_H.shape[:2], *SB, z), feather_mask(AFT_H.shape[:2], *SA, z)
        bg = np.float32([16, 16, 18]) / 255
        a_ = a_ * ma + bg * (1 - ma); b_ = b_ * mb + bg * (1 - mb)
        p = ease((t - 26.7) / 1.1); f = 0.35; edge = (xx_ / W * 0.6 + yy_ / H * 0.4)[..., None]
        m = np.clip((p * (1 + f) - edge) / f, 0, 1); fr = a_ * (1 - m) + b_ * m
        chip(fr, CH_DO, 60, 228, t, t0 + 0.05, 27.2); chip(fr, CH_PO, 60, 228, t, 27.6)
        T_RES.draw(fr, 1240, t, 27.8, t1 - 0.05, cps=40)
    elif n == "surgeon":
        fr = mblur(fillp(SURG, (1.12 + 0.07 * lt / 2.3) * trans(t, t0), 0.5, 0.38), lt)
        T_OP.draw(fr, 1240, t, t0 + 0.3, t1 - 0.05, cps=34)
    elif n in ("aft1", "aft2"):
        img = A84 if n == "aft1" else A85
        fr = mblur(fillp(img, (1.35 + 0.05 * lt / 2) * trans(t, t0), 0.5, 0.44), lt); chip(fr, CH_PO, 60, 228, t, t0 + 0.1)
        bullet(fr, t, 30.45, "через неделю — уже дома", 70, 1230)
        bullet(fr, t, 31.0, "через 3 дня — снова на работе", 70, 1310)
    else:
        fr = collage({"c1": C1, "c2": C2, "c3": C3}[n], t, t0)
    if t < 2.75: T_HOOK.draw(fr, 1150, t, 0.2, 2.7, cps=24)
    fr = look(fr, t)
    if n not in ("c1", "c2", "c3"): subs(fr, t)
    blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
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
