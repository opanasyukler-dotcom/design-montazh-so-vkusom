import subprocess, sys, json
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

W, H, FPS = 1080, 1920, 30
SP = "/tmp/claude-0/-home-user-design-montazh-so-vkusom/b26b20a4-6ac5-50f8-8404-dcbad88c88d3/scratchpad/"
BLUE = (16, 46, 70); WHITE = (255, 255, 255); LIME = (214, 236, 46)
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

BL = [("hook", 0, 2.0), ("hook2", 2.0, 2.75), ("talk1", 2.75, 6.45), ("talk2", 6.45, 8.85), ("side", 8.85, 11.85),
      ("before", 11.85, 14.6), ("morph", 14.6, 19.6), ("talk3", 19.6, 23.9), ("surgeon", 23.9, 26.7), ("result", 26.7, 28.0),
      ("talk4", 28.0, 31.2), ("talk5", 31.2, 34.3), ("happy", 34.3, 35.4),
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
A84 = blur_ell(load("dl/IMG_7584.PNG"), (525, 160), (95, 125))          # patient face blurred
A85 = blur_ell(load("dl/IMG_7585.PNG"), (520, 150), (95, 120))
SURG = load("dl/DSC01189.JPG")
C1, C2, C3 = load("dl/IMG_7595.JPG"), load("dl/collage_34.jpg"), load("dl/collage_side.jpg")

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
CH_DO, CH_PO = tag("ДО", 46), tag("ПОСЛЕ", 46)
def chip(fr, im, x, y, t, t0, t1=1e9):
    if t < t0 or t > t1 + 0.25: return
    p = ease((t - t0) / 0.35); al = min(1, (t - t0) / 0.1) * (1 - min(1, max(0, (t - t1) / 0.25)))
    blend(fr, shadow_of(im, 8, 0.4), x - (1 - p) * 140, y + 5, al); blend(fr, im, x - (1 - p) * 140, y, al)
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
def bullet(fr, t, t0, txt, x, y, size=44, t1=1e9):
    if t < t0 or t > t1 + 0.25: return
    key = (txt, size)
    if key not in TC:
        f = mont(600, size); asc, desc = f.getmetrics(); tw = int(f.getlength(txt)); hh = asc + desc
        im = Image.new("RGBA", (tw + 80, hh + 16), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 8, hh, hh + 8), 8, fill=BLUE + (255,)); c = hh // 2
        d.polygon([(c - 9, c + 1), (c - 9, c + 15), (c + 10, c + 8)], fill=WHITE + (255,))
        d.text((hh + 18, 8 + asc), txt, font=f, fill=WHITE + (255,), anchor="ls"); TC[key] = to_np(im)
    im = TC[key]; p = ease((t - t0) / 0.3); al = min(1, (t - t0) / 0.1) * (1 - min(1, max(0, (t - t1) / 0.25)))
    blend(fr, shadow_of(im, 9, 0.75), x - 70 * (1 - p), y + 4, al); blend(fr, im, x - 70 * (1 - p), y, al)

T_HOOK = Txt([("ЖИЗНЬ РАЗДЕЛИЛАСЬ", "H"), ("НА ДО И ПОСЛЕ", "HB")], size=128)
T_CONF = Txt([("УВЕРЕННОСТЬ В СЕБЕ", "HB"), ("это дорогого стоит", "SB")], size=104)
T_COVER = Txt([("ВСЁ ВРЕМЯ", "H"), ("ПРИКРЫВАЛАСЬ ОДЕЖДОЙ", "HB")], size=100)
T_JOY = Txt([("УДОВОЛЬСТВИЕ", "H"), ("ОТ СВОЕГО ТЕЛА", "HB")], size=118)
T_TRIP = Txt([("ПРИЕХАЛА ИЗДАЛЕКА", "HB"), ("из Саратовской области", "SB")], size=96)
T_OP = Txt([("ОЧЕНЬ СЛОЖНАЯ", "H"), ("И ДЛИННАЯ ОПЕРАЦИЯ", "HB")], size=104)
T_RES = Txt([("РЕЗУЛЬТАТ", "H"), ("ПОТРЯСАЮЩИЙ", "HB")], size=130)
T_HAPPY = Txt([("ОЧЕНЬ ДОВОЛЬНА", "HB")], size=110)

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
            blend(frame, sh, 540 - im.shape[1] / 2, y - im.shape[0] / 2 + 3 + 12 * (1 - p), al)
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
        fr = fillp(B80, 1.12 + 0.05 * t, 0.5, 0.42); chip(fr, CH_DO, 60, 228, t, 0.1)
    elif n == "hook2":
        fr = fillp(A84, 1.35 * trans(t, t0, 0.1), 0.52, 0.42); flash = 0.55 * max(0, 1 - lt / 0.2); chip(fr, CH_PO, 60, 228, t, t0 + 0.05)
    elif n.startswith("talk") or n == "happy":
        z = dict(talk1=1.0, talk2=1.32, talk3=1.15, talk4=1.0, talk5=1.38, happy=1.25)[n] * (1 + 0.03 * lt / 3) * trans(t, t0)
        fr = mblur(iframe(t, z, 820 if z > 1.2 else 900), lt)
        if n == "talk1": T_CONF.draw(fr, 1180, t, t0 + 0.7, t1 - 0.05, cps=30)
        if n == "talk3": T_TRIP.draw(fr, 1180, t, 21.75, t1 - 0.05, cps=32)
        if n in ("talk4", "talk5"):
            bullet(fr, t, 28.35, "через неделю — уже дома", 70, 1230)
            bullet(fr, t, 30.6, "через 3 дня — снова на работе", 70, 1310)
        if n == "happy": T_HAPPY.draw(fr, 1250, t, t0 + 0.05, t1, cps=34)
    elif n == "side":
        fr = mblur(fillp(A85, (1.45 + 0.06 * lt / 3) * trans(t, t0), 0.48, 0.44), lt); chip(fr, CH_PO, 60, 228, t, t0 + 0.1)
        bracket(fr, t, t0 + 0.15, ["НРАВИШЬСЯ СЕБЕ РАЗДЕТОЙ", "БОЛЬШЕ, ЧЕМ ОДЕТОЙ"], 1230, 76, t1 - 0.1)
    elif n == "before":
        img, s0 = (B80, t0) if t < 12.75 else ((B81, 12.75) if t < 13.7 else (B82, 13.7))
        fr = mblur(fillp(img, (1.1 + 0.04 * (t - s0)) * trans(t, s0), 0.5, 0.42), t - s0); chip(fr, CH_DO, 60, 228, t, t0 + 0.05)
        T_COVER.draw(fr, 1170, t, t0 + 0.15, t1 - 0.05, cps=34)
    elif n == "morph":
        # smooth ДО -> ПОСЛЕ: same framing (front collage halves aligned on the stickers), soft diagonal wipe
        z = 1.0 + 0.05 * lt / 5
        a_ = warp(BEF_H, 620 / np.hypot(SB[1][0] - SB[0][0], SB[1][1] - SB[0][1]) * z, (SB[0][0] + SB[1][0]) / 2, (SB[0][1] + SB[1][1]) / 2, 540, 760)
        b_ = warp(AFT_H, 620 / np.hypot(SA[1][0] - SA[0][0], SA[1][1] - SA[0][1]) * z, (SA[0][0] + SA[1][0]) / 2, (SA[0][1] + SA[1][1]) / 2, 540, 760)
        ma, mb = feather_mask(BEF_H.shape[:2], *SB, z), feather_mask(AFT_H.shape[:2], *SA, z)
        bg = np.float32([16, 16, 18]) / 255
        a_ = a_ * ma + bg * (1 - ma); b_ = b_ * mb + bg * (1 - mb)
        p = ease((t - 15.7) / 1.1); f = 0.35; edge = (xx_ / W * 0.6 + yy_ / H * 0.4)[..., None]
        m = np.clip((p * (1 + f) - edge) / f, 0, 1); fr = a_ * (1 - m) + b_ * m
        chip(fr, CH_DO, 60, 228, t, t0 + 0.05, 16.2); chip(fr, CH_PO, 60, 228, t, 16.6)
        T_JOY.draw(fr, 1240, t, 17.9, t1 - 0.05, cps=30)
    elif n == "surgeon":
        fr = mblur(fillp(SURG, (1.12 + 0.07 * lt / 2.8) * trans(t, t0), 0.5, 0.38), lt)
        T_OP.draw(fr, 1240, t, t0 + 0.4, t1 - 0.05, cps=34)
    elif n == "result":
        fr = fillp(A84, 1.5 * trans(t, t0, 0.12), 0.5, 0.44); flash = 0.5 * max(0, 1 - lt / 0.2); chip(fr, CH_PO, 60, 228, t, t0 + 0.1)
        T_RES.draw(fr, 1180, t, t0 + 0.05, t1, cps=40)
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
         (26.75, 40, "РЕЗУЛЬТАТПОТРЯСАЮЩИЙ"), (34.35, 34, "ОЧЕНЬ ДОВОЛЬНА")]

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
