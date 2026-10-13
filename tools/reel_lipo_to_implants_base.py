import subprocess, sys, json
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

W, H, FPS = 1080, 1920, 30
SP = "/tmp/claude-0/-home-user-design-montazh-so-vkusom/b26b20a4-6ac5-50f8-8404-dcbad88c88d3/scratchpad/"
BLUE = (16, 46, 70); WHITE = (255, 255, 255); LIME = (214, 236, 46); PINK = (228, 210, 210); SKY = (228, 210, 210); GRAPH = (103, 96, 104)   # surgeon palette: powder / navy / graphite
from retouch import retouch
def osw(s): return ImageFont.truetype(SP + "v14/fonts/BN-Bold.ttf", s)          # Bebas Neue (Cyrillic)
def mont(w, s): return ImageFont.truetype(SP + f"fonts2/M{w}.ttf", s)
SUBS = []; DUR = 41.0
exec(open("helpers_src.py").read())

def load(p): return np.array(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
def blur_ell(img, c, ax):
    c = np.array(c, float); ax = np.array(ax, float)
    x0, y0 = np.maximum(0, (c - ax * 1.5).astype(int)); x1, y1 = np.minimum([img.shape[1], img.shape[0]], (c + ax * 1.5).astype(int))
    crop = img[y0:y1, x0:x1].astype(np.float32); bl = cv2.GaussianBlur(cv2.resize(crop, None, fx=0.15, fy=0.15), (0, 0), 5)
    bl = cv2.resize(bl, (crop.shape[1], crop.shape[0]))
    m = np.zeros(crop.shape[:2], np.float32); cv2.ellipse(m, (int(c[0] - x0), int(c[1] - y0)), (int(ax[0]), int(ax[1])), 0, 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(ax) / 6)[..., None]
    img = img.copy(); img[y0:y1, x0:x1] = (crop * (1 - m) + bl * m).astype(np.uint8); return img
class Src:
    def __init__(s, p): s.cap = cv2.VideoCapture(p); s.pos = -1; s.c = {}
    def get(s, i):
        if i in s.c: return s.c[i]
        if i != s.pos + 1: s.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = s.cap.read(); s.pos = i
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if len(s.c) > 40: s.c.pop(next(iter(s.c)))
        s.c[i] = f; return f
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
CUTS = []
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
def put_disc(fr, img, cx, cy, al=1.0, shadow=True):
    r = img.shape[0] // 2; m = np.zeros((2 * r, 2 * r), np.uint8); cv2.circle(m, (r, r), r - 1, 255, -1, cv2.LINE_AA)
    rgba = np.concatenate([img * (m[..., None] / 255.0), m[..., None] / 255.0], -1).astype(np.float32)
    if shadow: blend(fr, shadow_of(rgba, 22, 0.35), cx - r, cy - r + 14, al)
    blend(fr, rgba, cx - r, cy - r, al)
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
def push_in(fr_new, t, t0, dur=0.35, frm=1):
    p = ease((t - t0) / dur)
    if p >= 1: return fr_new
    out = np.zeros_like(fr_new); dx = int(W * (1 - p)) * frm
    if frm > 0: out[:, dx:] = fr_new[:, :W - dx]
    else: out[:, :W + dx] = fr_new[:, -dx:]
    return mblur(out, 0.0)

