"""@mcg.va — reel «мои осознания в материнстве», montage system «СДВГ» (animated-montage style, her type and palette).
Voice-over over her clips; every phrase is shown: typewriter subtitles, key words on plaques / huge translucent thin caps,
cream «paper» sheets with a window-in-window, chat bubbles, polaroids, her cut-outs, hand-drawn arrow, price tag, progress bar,
whip-blur and zoom-blur transitions, night / b&w treatments.
  python3 sdvg_materinstvo.py MATERIALS_DIR CUTOUTS_DIR WORDS.json OUT.mp4 --music TRACK START_S [--preview T1,T2,...]
(the voice is tightened: pauses cut, studio chain; sound effects synthesized in sdvg_audio.py)
"""
import json, math, os, subprocess, sys
import numpy as np, cv2
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except ImportError: pass
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(HERE, "..", "fonts", "stories")
W, H, FPS = 1080, 1920, 25
CREAM, INK, BROWN, MAUVE, BLUSH, WHITE = (243, 236, 228), (43, 31, 26), (107, 74, 56), (139, 107, 92), (232, 201, 192), (255, 255, 255)
CX = 505                                                  # text centre: middle of the Reels safe zone (x 60–950)
SEGS = [(1.25, 19.70), (20.80, 45.80), (46.85, 55.00)]    # voice kept (two long pauses shortened, «спасибо» dropped)
END = 53.0

# speech runs of the voice (orig seconds, energy > −45 dB, gaps < 0.14 s merged); everything between them is cut:
# sentence pauses become 0.14 s, the rest 0.05 s. The montage is laid out on the old timeline (SEGS) and remapped.
RUNS = [[1.87, 5.06], [5.31, 6.8], [6.95, 7.43], [8.21, 8.57], [9.82, 14.64], [15.11, 19.22], [21.06, 23.81], [23.97, 27.48],
        [28.0, 29.05], [29.27, 31.85], [32.05, 32.97], [34.13, 36.51], [36.96, 40.47], [40.91, 42.49], [42.73, 43.98],
        [44.46, 45.16], [47.37, 49.86], [50.0, 50.07], [50.35, 53.24], [53.52, 54.55]]
PAD = (0.05, 0.10)
NRUNS = [[a - PAD[0], b + PAD[1]] for a, b in RUNS]
GAPS = [(0.14 if NRUNS[k + 1][0] - NRUNS[k][1] >= 0.35 else 0.05) for k in range(len(NRUNS) - 1)] + [0.0]
NSTART = list(np.cumsum([0.0] + [b - a + g for (a, b), g in zip(NRUNS, GAPS)])[:-1])
TAIL = 1.4                                                 # hold after the last word (bar fills, fade)

def old_of_orig(o):
    acc = 0.0
    for a, b in SEGS:
        if o <= b: return acc + max(o - a, 0)
        acc += b - a
    return acc

def orig_of_old(x):
    acc = 0.0
    for a, b in SEGS:
        if x <= acc + (b - a): return a + (x - acc)
        acc += b - a
    return SEGS[-1][1]

def new_of_orig(o):
    for k, (a, b) in enumerate(NRUNS):
        if o < a: return NSTART[k]
        if o <= b: return NSTART[k] + (o - a)
    return NSTART[-1] + NRUNS[-1][1] - NRUNS[-1][0] + (o - NRUNS[-1][1])

NEW_END = NSTART[-1] + NRUNS[-1][1] - NRUNS[-1][0] + TAIL

def old_of_new(t):
    """the montage time (old timeline) shown at time t of the tightened reel"""
    for k, (a, b) in enumerate(NRUNS):
        n0, n1 = NSTART[k], NSTART[k] + (b - a)
        if t <= n1: return old_of_orig(a + max(t - n0, 0))
        if k + 1 < len(NRUNS) and t < NSTART[k + 1]:               # inside a shortened gap: run through the cut pause
            u = (t - n1) / (NSTART[k + 1] - n1); return old_of_orig(b) + u * (old_of_orig(NRUNS[k + 1][0]) - old_of_orig(b))
    x0 = old_of_orig(NRUNS[-1][1]); return min(x0 + (t - (NEW_END - TAIL)) * (END - x0) / TAIL, END - 1e-3)

new_of_old = lambda x: new_of_orig(orig_of_old(x))

_fc = {}
def font(size, wght=300, it=False):
    k = (size, wght, it)
    if k not in _fc:
        f = ImageFont.truetype(os.path.join(FD, "Montserrat-Italic-VariableFont_wght.ttf" if it else "Montserrat-VariableFont_wght.ttf"), size)
        f.set_variation_by_axes([wght]); _fc[k] = f
    return _fc[k]

ease = lambda x: 1 - (1 - min(max(x, 0), 1)) ** 3

# ── voice timeline ──────────────────────────────────────────────────────────────────────────────────────────────────
def vt(o):
    acc = 0.0
    for a, b in SEGS:
        if a - 0.05 <= o <= b + 0.05: return acc + (o - a)
        acc += b - a
    return None

PHRASES = ["мне никто не говорил, что материнство — это не про ребёнка,", "оно про мою гордыню, лень и эго.", "ночь.",
           "Агата плачет,", "мне не хочется вставать, а я встаю,", "всё бесит, а выбора нет.", "хочется сказать «не буду, не хочу»,",
           "а сказать это просто некому.", "и так каждый день я боролась с собой,", "пока в какой-то момент просто не расслабилась",
           "и не приняла себя новую, свою новую жизнь", "и маленького человечка, которому я теперь служу безвозмездно.",
           "материнство — это пипец какой учитель жизни.", "оно закаляет, делает взрослее, серьёзнее, сильнее,",
           "потому что ради своего ребёнка ты теперь можешь всё,", "даже убить.", "у материнства нет выходных, нет отпусков,",
           "и единственное, чему оно по-настоящему учит, —", "это принятие."]

def load_words(path):
    raw = json.load(open(path)); ws = []
    for w, a, b in raw:
        w = w.strip()
        if w.startswith("-") and ws: ws[-1][0] += w; ws[-1][2] = b; continue
        ws.append([w, a, b])
    ws = [w for w in ws if not w[0].startswith("Спасибо")]
    k = next(i for i, w in enumerate(ws) if w[0] == "в" and 26.3 < w[1] < 26.6)       # «приняла себя… новую, свою новую жизнь»
    ws[k:k + 3] = [["себя", 26.42, 26.9], ["новую", 27.05, 27.48], ["свою", 28.0, 28.26]]
    cues, i = [], 0
    for p in PHRASES:
        toks = [t for t in p.split() if any(c.isalnum() for c in t)]; tw = []
        for t in p.split():
            if any(c.isalnum() for c in t): tw.append([t, vt(ws[i][1]), vt(ws[i][2])]); i += 1
            else: tw[-1][0] += " " + t
        cues.append(tw)
    assert i == len(ws), (i, len(ws))
    return cues

# ── sources ─────────────────────────────────────────────────────────────────────────────────────────────────────────
class Stream:
    """frames of a clip, read in order from an ffmpeg pipe; cover-cropped to (w, h); horizontal sources cropped around cx"""
    def __init__(self, src, ss, dur, w=W, h=H, cx=None, speed=1.0, fy=0.5):
        self.w, self.h, self.n, self.last = w, h, 0, None
        if cx is not None:                                     # 4K landscape camera: crop to the target aspect around cx
            iw, ih = 3840, 2160; cw = ih * w / h; x = min(max(cx * iw - cw / 2, 0), iw - cw)
            vf = f"crop={int(cw)}:{ih}:{int(x)}:0,scale={w}:{h}"
        else: vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}:(iw-{w})/2:(ih-{h})*{fy}"
        vf = (f"setpts=PTS/{speed}," if speed != 1 else "") + vf + f",fps={FPS}"
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{dur * speed + 0.5:.3f}", "-i", src, "-vf", vf,
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)

    def frame(self, i):
        while self.n <= i:
            b = self.p.stdout.read(self.w * self.h * 3)
            if len(b) < self.w * self.h * 3: break
            self.last = np.frombuffer(b, np.uint8).reshape(self.h, self.w, 3); self.n += 1
        return self.last

    def close(self):
        try: self.p.kill()
        except Exception: pass

def treat(a, mode):
    if mode is None: return a
    f = a.astype(np.float32)
    if mode == "night":
        g = f.mean(2, keepdims=True); f = (f * 0.35 + g * 0.65) * 0.42 * np.array([0.86, 0.92, 1.12])
    elif mode == "bw": g = f @ np.array([0.3, 0.59, 0.11]); f = np.repeat(g[..., None], 3, 2) * 0.95 + 6
    elif mode == "soft": g = f.mean(2, keepdims=True); f = f * 0.75 + g * 0.25
    elif mode == "warm": f = f * np.array([1.04, 1.0, 0.93]) + np.array([6, 3, 0])
    return np.clip(f, 0, 255).astype(np.uint8)

def zoom(a, z, fy=0.45):
    if z <= 1.0: return a
    h, w = a.shape[:2]; cw, ch = int(w / z), int(h / z); x, y = (w - cw) // 2, int((h - ch) * fy)
    return cv2.resize(a[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_LINEAR)

_grain = None
def paper():
    global _grain
    if _grain is None:
        rng = np.random.default_rng(3); n = rng.normal(0, 3.2, (H, W, 1))
        _grain = np.clip(np.array(CREAM, np.float32) + n, 0, 255).astype(np.uint8)
    return _grain.copy()

# ── sprites ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def text_img(text, f, fill, pad=0, glow=None, alpha=1.0):
    l, t, r, b = f.getbbox(text); w, h = r - l + 2 * pad + 20, b - t + 2 * pad + 20
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.text((pad + 10 - l, pad + 10 - t), text, font=f, fill=fill + (int(255 * alpha),))
    if glow:
        g = Image.new("RGBA", (w + 60, h + 60), (0, 0, 0, 0)); gd = ImageDraw.Draw(g)
        gd.text((pad + 40 - l, pad + 40 - t), text, font=f, fill=glow + (110,)); g = g.filter(ImageFilter.GaussianBlur(14))
        g.alpha_composite(im, (30, 30)); im = g
    return im

def plaque_img(text, size=44, style="blush", it=True):
    f = font(size, 300, it); bg, fg = {"blush": (BLUSH, INK), "ink": (INK, CREAM), "cream": (CREAM, INK)}[style]
    l, t, r, b = f.getbbox(text); px, py = int(size * 0.38), int(size * 0.2)
    im = Image.new("RGBA", (r - l + 2 * px, int(size * 1.25) + 2 * py), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, im.width - 1, im.height - 1), 6, fill=bg + (240,))
    d.text((px - l, im.height / 2), text, font=f, fill=fg + (255,), anchor="lm"); return im

def shadowed(im, r=18, op=0.28, off=(0, 10)):
    s = Image.new("RGBA", (im.width + 4 * r, im.height + 4 * r), (0, 0, 0, 0))
    a = im.split()[3].point(lambda v: int(v * op)); sh = Image.new("RGBA", im.size, (40, 28, 22, 255)); sh.putalpha(a)
    s.alpha_composite(sh, (2 * r + off[0], 2 * r + off[1])); s = s.filter(ImageFilter.GaussianBlur(r)); s.alpha_composite(im, (2 * r, 2 * r))
    return s

class Sprite:
    """an image that appears at t0 (pop / wipe / slide / drop / fade) and leaves at t1; x, y = centre"""
    def __init__(self, im, x, y, t0, t1, anim="pop", angle=0, shake=0, dur=0.24, out=0.14):
        self.im, self.x, self.y, self.t0, self.t1, self.anim, self.angle, self.shake, self.dur, self.out = im, x, y, t0, t1, anim, angle, shake, dur, out
        self.cache = None

    def draw(self, cv, t):
        if not (self.t0 <= t < self.t1): return
        k = ease((t - self.t0) / self.dur); fade = min(1.0, (self.t1 - t) / self.out) if self.out else 1.0
        im = self.im; s = 1.0; dx = dy = 0; a = 1.0
        if self.anim == "pop": s = 1.14 - 0.14 * k; a = min(1, (t - self.t0) / 0.1)
        elif self.anim == "fade": a = k
        elif self.anim == "wipe":
            w = max(1, int(im.width * k)); im = im.crop((0, 0, w, im.height)); dx = -(self.im.width - w) / 2
        elif self.anim == "up": dy = (1 - k) * 260; a = k
        elif self.anim == "drop": dy = -(1 - k) * 120; s = 1.06 - 0.06 * k; a = min(1, k * 2)
        elif self.anim == "left": dx = -(1 - k) * 700
        if self.angle or s != 1:
            if self.anim != "wipe" or k >= 1:
                if self.cache is None or self.cache[0] != (round(s, 3), im.size):
                    j = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BICUBIC) if s != 1 else im
                    j = j.rotate(self.angle, resample=Image.BICUBIC, expand=True) if self.angle else j
                    self.cache = ((round(s, 3), im.size), j)
                im = self.cache[1]
            else: im = im.rotate(self.angle, resample=Image.BICUBIC, expand=True) if self.angle else im
        if self.shake: dx += math.sin(t * 61) * self.shake; dy += math.cos(t * 47) * self.shake * 0.6
        a *= fade
        if a < 1: im = im.copy(); im.putalpha(im.split()[3].point(lambda v: int(v * a)))
        cv.alpha_composite(im, (int(self.x + dx - im.width / 2), int(self.y + dy - im.height / 2)))

class Typed:
    """text typed letter by letter (left-aligned at x, baseline y)"""
    def __init__(self, text, x, y, t0, t1, f, fill, step=0.045, anchor="ls", shadow=False):
        self.text, self.x, self.y, self.t0, self.t1, self.f, self.fill, self.step, self.anchor, self.shadow = text, x, y, t0, t1, f, fill, step, anchor, shadow
    def draw(self, cv, t):
        if not (self.t0 <= t < self.t1): return
        n = int((t - self.t0) / self.step) + 1; s = self.text[:n]; d = ImageDraw.Draw(cv)
        if self.shadow: shadow_text(cv, (self.x, self.y), s, self.f, self.anchor)
        d.text((self.x, self.y), s, font=self.f, fill=self.fill + (255,), anchor=self.anchor)
        if n < len(self.text) and int(t * 4) % 2 == 0:
            x1 = self.x + self.f.getlength(s) if self.anchor[0] == "l" else self.x
            d.line((x1 + 4, self.y - self.f.size * 0.8, x1 + 4, self.y + 4), fill=self.fill + (255,), width=2)

def shadow_text(cv, xy, s, f, anchor):
    sh = Image.new("L", cv.size, 0); ImageDraw.Draw(sh).text((xy[0], xy[1] + 2), s, font=f, fill=170, anchor=anchor)
    bb = sh.getbbox()
    if not bb: return
    pad = 20; bb = (max(bb[0] - pad, 0), max(bb[1] - pad, 0), min(bb[2] + pad, cv.width), min(bb[3] + pad, cv.height))
    part = sh.crop(bb).filter(ImageFilter.GaussianBlur(5)); layer = Image.new("RGBA", part.size, (20, 14, 10, 0)); layer.putalpha(part)
    cv.alpha_composite(layer, bb[:2])

class Fn:
    def __init__(self, fn, t0, t1): self.fn, self.t0, self.t1 = fn, t0, t1
    def draw(self, cv, t):
        if self.t0 <= t < self.t1: self.fn(cv, t)

# ── building blocks ─────────────────────────────────────────────────────────────────────────────────────────────────
def big(text, x, y, t0, t1, size=150, col=BLUSH, alpha=0.92, glow=(50, 34, 26), anim="pop", angle=0):
    return Sprite(text_img(text, font(size, 200), col, glow=glow, alpha=alpha), x, y, t0, t1, anim, angle, dur=0.3)

def plaque(text, x, y, t0, t1, style="blush", angle=-3, size=44, anim="wipe", shake=0, it=True):
    return Sprite(shadowed(plaque_img(text, size, style, it), 10, 0.22, (0, 5)), x, y, t0, t1, anim, angle, shake, dur=0.2)

def window_frame(w, h):
    """rounded window with a title bar (three dots); returns RGBA and the inner box offset"""
    bar = 46; im = Image.new("RGBA", (w + 16, h + bar + 16), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, im.width - 1, im.height - 1), 22, fill=(250, 246, 241, 255), outline=(218, 205, 196, 255), width=2)
    for i, c in enumerate((BLUSH, MAUVE, (200, 190, 182))): d.ellipse((22 + i * 30, 15, 38 + i * 30, 31), fill=c + (255,))
    return im, (8, bar + 8)

def polaroid(path, w, crop_ratio=1.0, angle=0):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB"); h = int(w * crop_ratio)
    s = max(w / im.width, h / im.height); im = im.resize((int(im.width * s) + 1, int(im.height * s) + 1), Image.LANCZOS)
    im = im.crop(((im.width - w) // 2, (im.height - h) // 3, (im.width - w) // 2 + w, (im.height - h) // 3 + h))
    b = 22; card = Image.new("RGBA", (w + 2 * b, h + b + 86), (252, 250, 247, 255)); card.paste(im, (b, b))
    tape = Image.new("RGBA", (170, 48), (226, 214, 196, 170)); tape = tape.rotate(-6, expand=True, resample=Image.BICUBIC)
    out = shadowed(card, 16, 0.3, (0, 10)); out.alpha_composite(tape, (out.width // 2 - tape.width // 2, 10)); return out

def cutout_img(path, h):
    im = Image.open(path).convert("RGBA"); im = im.resize((int(im.width * h / im.height), h), Image.LANCZOS)
    return shadowed(im, 14, 0.22, (8, 10))

def arrow_fn(pts, t0, dur=0.45, col=INK, wd=4):
    """hand-drawn arc (quadratic bezier p0→p1 via c) drawn progressively, with a head"""
    (x0, y0), (cx, cy), (x1, y1) = pts
    def fn(cv, t):
        k = ease((t - t0) / dur); n = max(2, int(60 * k)); d = ImageDraw.Draw(cv)
        P = [((1 - u) ** 2 * x0 + 2 * (1 - u) * u * cx + u * u * x1 + math.sin(u * 9) * 2, (1 - u) ** 2 * y0 + 2 * (1 - u) * u * cy + u * u * y1)
             for u in np.linspace(0, k, n)]
        d.line(P, fill=col + (255,), width=wd, joint="curve")
        if k >= 1:
            ax, ay = P[-1]; bx, by = P[-4]; a = math.atan2(ay - by, ax - bx)
            for s in (-0.5, 0.5): d.line((ax, ay, ax - 26 * math.cos(a + s), ay - 26 * math.sin(a + s)), fill=col + (255,), width=wd)
    return fn

def bubble_img(text):
    f = font(50, 300); w = int(f.getlength(text)) + 70; im = Image.new("RGBA", (w, 96), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, 95), 44, fill=BLUSH + (255,)); d.text((35, 48), text, font=f, fill=INK + (255,), anchor="lm")
    return shadowed(im, 10, 0.18, (0, 5))

def tag_img(text):
    f = font(64, 250); w = int(f.getlength(text)) + 110; im = Image.new("RGBA", (w, 120), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.polygon([(0, 60), (34, 0), (w - 1, 0), (w - 1, 119), (34, 119)], fill=BLUSH + (255,))
    d.ellipse((24, 50, 44, 70), fill=CREAM + (255,)); d.text((62, 62), text, font=f, fill=INK + (255,), anchor="lm")
    return shadowed(im, 10, 0.2, (0, 6))

def pill_img(text, fg=CREAM, bg=(20, 16, 22)):
    f = font(30, 350); w = int(f.getlength(text)) + 74; im = Image.new("RGBA", (w, 58), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, 57), 29, fill=bg + (190,)); d.ellipse((20, 21, 36, 37), fill=BLUSH + (255,))
    d.text((48, 29), text, font=f, fill=fg + (255,), anchor="lm"); return im

# ── subtitles ───────────────────────────────────────────────────────────────────────────────────────────────────────
SUB_F = font(38, 300)
def wrap_words(words, f, maxw):
    lines, cur = [], []
    for w in words:
        if cur and f.getlength(" ".join(c[0] for c in cur + [w])) > maxw: lines.append(cur); cur = [w]
        else: cur.append(w)
    return lines + ([cur] if cur else [])

def subs_draw(cv, t, cues, ink):
    for k, cue in enumerate(cues):
        a = cue[0][1] - 0.05; b = (cues[k + 1][0][1] - 0.02) if k + 1 < len(cues) else cue[-1][2] + 1.2
        b = min(b, cue[-1][2] + 1.0)
        if not (a <= t < b): continue
        lines = wrap_words(cue, SUB_F, 800); y0 = 1404 - (len(lines) - 1) * 50; d = ImageDraw.Draw(cv)
        for j, ln in enumerate(lines):
            vis = []
            for w, ws, we in ln:
                if t < ws: break
                n = int((t - ws) / 0.03) + 1; vis.append(w[:n])
            s = " ".join(vis)
            if not s: continue
            full = " ".join(w for w, _, _ in ln); x = CX - SUB_F.getlength(full) / 2
            if not ink: shadow_text(cv, (x, y0 + j * 50), s, SUB_F, "ls")
            d.text((x, y0 + j * 50), s, font=SUB_F, fill=(INK if ink else WHITE) + (255,), anchor="ls")

# ── the reel ────────────────────────────────────────────────────────────────────────────────────────────────────────
def build(M, C):
    dl = lambda n: os.path.join(M, n)
    V = lambda n, ss, **k: (lambda dur: Stream(dl(n), ss, dur, **k))
    sc = []                                                   # (t0, t1, kind, spec, extra)
    def S(t0, t1, kind, spec=None, **kw): sc.append(dict(t0=t0, t1=t1, kind=kind, spec=spec, **kw))
    S(0.00, 2.95, "clip", V("C0894.MP4", 24.0, cx=576 / 1280))
    S(2.95, 4.60, "clip", V("C0894.MP4", 8.0, cx=0.50), z=1.06)
    S(4.60, 7.00, "clip", V("C0894.MP4", 44.0, cx=316 / 1280), mode="soft")
    S(7.00, 8.60, "clip", V("IMG_0800.MOV", 0.5), mode="night")
    S(8.60, 9.60, "clip", V("430b2de9-bc03-410c-bebc-cd66671eeb74.mp4", 1.5), mode="night")
    S(9.60, 11.75, "clip", V("IMG_6515.MOV", 8.0), mode="night")
    S(11.75, 12.20, "clip", V("IMG_6257.MOV", 4.5))
    S(12.20, 12.65, "clip", V("IMG_5552.MOV", 8.0))
    S(12.65, 13.05, "clip", V("IMG_6740.MOV", 6.0))
    S(13.05, 13.50, "clip", V("IMG_6376.MOV", 3.0))
    S(13.50, 18.60, "paper")
    S(18.60, 20.00, "days", [V("IMG_5545.MOV", 1.0, w=800, h=730), V("IMG_6739.MOV", 4.0, w=800, h=730),
                             V("IMG_7146.MOV", 2.0, w=800, h=730), V("IMG_5539.MOV", 2.0, w=800, h=730)])
    S(20.00, 21.25, "clip", V("IMG_5612.MOV", 1.5), mode="bw")
    S(21.25, 23.40, "clip", V("IMG_0262.MOV", 1.0, speed=0.5), mode="warm")
    S(23.40, 26.75, "window", V("C0895.MP4", 51.0, w=840, h=760, cx=0.57))
    S(26.75, 29.20, "paper")
    S(29.20, 31.90, "paper")
    S(31.90, 34.50, "zoomblur", (dl("C0894.MP4"), 36.0, 300 / 1280))
    S(34.50, 36.00, "clip", V("IMG_5598.MOV", 1.0))
    S(36.00, 37.30, "clip", V("IMG_6740.MOV", 2.5))
    S(37.30, 38.40, "clip", V("IMG_5598.MOV", 5.5))
    S(38.40, 41.75, "clip", V("C0895.MP4", 18.6, cx=549 / 1280), mode="warm")
    S(41.75, 43.60, "clip", V("C0894.MP4", 108.0, cx=529 / 1280), mode="bw", z=1.12)
    S(43.60, 46.70, "split", [V("IMG_6257.MOV", 9.0, h=950), V("IMG_6376.MOV", 5.0, h=950)])
    S(46.70, 50.30, "clip", V("IMG_6739.MOV", 0.3), mode="warm")
    S(50.30, END, "clip", V("C0895.MP4", 60.0, cx=745 / 1280), mode="warm")

    el = []; A = el.append
    # hook
    A(big("МАТЕРИНСТВО", CX, 1190, 2.00, 2.95, 112))
    A(plaque("не про ребёнка", CX, 1262, 3.30, 4.60, "blush", -3))
    A(plaque("гордыню", 330, 1020, 4.69, 7.00, "ink", -4, 50))
    A(plaque("лень", 620, 1105, 5.49, 7.00, "blush", 3, 50))
    A(plaque("и эго", 420, 1200, 5.97, 7.00, "ink", -2, 60))
    # night
    A(big("НОЧЬ.", CX, 760, 7.05, 8.60, 190))
    A(Sprite(pill_img("03:12"), CX, 920, 7.30, 8.60, "pop"))
    A(plaque("Агата плачет", 600, 610, 8.71, 9.60, "blush", -2, 46))
    A(Typed("не хочется", 90, 560, 9.70, 11.75, font(72, 250, True), WHITE, 0.05, shadow=True))
    A(Typed("вставать", 90, 650, 10.30, 11.75, font(72, 250, True), WHITE, 0.05, shadow=True))
    A(plaque("…а я встаю", 260, 760, 10.95, 11.75, "blush", -3, 46))
    A(big("ВСЁ БЕСИТ", CX, 330, 11.85, 13.50, 120))
    A(plaque("а выбора нет", CX, 470, 12.71, 13.50, "ink", 2, 46))
    # chat: «не буду, не хочу» — некому
    A(Sprite(cutout_img(os.path.join(C, "DSC01020.png"), 1060), 790, 860, 13.50, 18.60, "up", dur=0.45, out=0.1))
    A(Fn(lambda cv, t: ImageDraw.Draw(cv).text((90, 430), "кому:", font=font(44, 300), fill=MAUVE + (255,), anchor="ls"), 13.6, 18.6))
    A(Fn(lambda cv, t: (int(t * 2.5) % 2 == 0) and ImageDraw.Draw(cv).line((232, 392, 232, 438), fill=INK + (255,), width=2), 13.6, 17.39))
    A(Typed("…", 96, 560, 14.15, 14.59, font(40, 400), MAUVE, 0.12))
    A(Sprite(bubble_img("не буду"), 90 + bubble_img("не буду").width / 2, 540, 14.59, 18.60, "pop"))
    A(Typed("…", 96, 710, 15.0, 15.31, font(40, 400), MAUVE, 0.12))
    A(Sprite(bubble_img("не хочу"), 90 + bubble_img("не хочу").width / 2, 690, 15.31, 18.60, "pop"))
    A(Typed("некому", 232, 432, 17.39, 18.60, font(54, 300, True), INK, 0.06))
    A(Fn(lambda cv, t: [ImageDraw.Draw(cv).text((104, y), "не доставлено", font=font(24, 350), fill=MAUVE + (255,), anchor="ls") for y in (626, 776)], 17.6, 18.6))
    A(big("НЕКОМУ.", 380, 960, 17.45, 18.60, 120, MAUVE, 0.55, None))
    # every day — fought — relaxed
    A(Fn(days_fn, 18.6, 20.0))
    A(plaque("боролась с собой", CX, 700, 20.03, 21.25, "ink", -3, 52, "pop", shake=5))
    A(Sprite(text_img("расслабилась", font(86, 250, True), WHITE, glow=(120, 90, 70)), CX, 560, 22.47, 23.40, "fade", dur=0.5))
    # accepted myself, new life
    A(plaque("приняла себя", CX, 1140, 23.39, 26.75, "blush", -3, 48))
    A(Typed("свою", 150, 1245, 24.73, 26.75, font(46, 300, True), INK, 0.05))
    A(big("НОВУЮ ЖИЗНЬ", CX, 1270, 25.91, 26.75, 96, MAUVE, 0.6, None))
    # little human — polaroids + arrow
    A(Sprite(polaroid(dl("IMG_7014.HEIC"), 520, 1.05, -5), 430, 690, 26.73, 29.20, "drop", -5, dur=0.3))
    A(Sprite(polaroid(dl("ace80e44-c1eb-481b-8bbd-e450f06c00df.jpg"), 400, 1.1, 6), 700, 1040, 27.55, 29.20, "drop", 6, dur=0.3))
    A(Typed("маленький человечек", 80, 1290, 27.55, 29.20, font(46, 300, True), INK, 0.04))
    A(Fn(arrow_fn(((300, 1240), (330, 1130), (500, 1110)), 28.0), 28.0, 29.2))
    # serve for free
    A(big("БЕЗВОЗМЕЗДНО", CX, 560, 29.73, 31.90, 94, MAUVE, 0.6, None))
    A(Sprite(text_img("служу", font(96, 250, True), INK), 330, 440, 29.21, 31.90, "pop"))
    A(Sprite(cutout_img(os.path.join(C, "IMG_6987.png"), 860), 770, 900, 29.40, 31.90, "up", dur=0.45))
    A(Sprite(tag_img("0 ₽"), 300, 900, 30.60, 31.90, "drop", -8))
    # teacher of life
    A(Sprite(text_img("пипец какой", font(72, 250, True), WHITE), CX, 690, 32.35, 34.50, "pop"))
    A(big("УЧИТЕЛЬ", CX, 840, 33.27, 34.50, 150))
    A(big("ЖИЗНИ", CX, 1000, 33.62, 34.50, 150))
    for txt, t0, x, y, a, st in (("закаляет", 34.63, 260, 1060, -5, "blush"), ("взрослее", 35.79, 380, 1140, 3, "ink"),
                                 ("серьёзнее", 36.87, 280, 1220, -3, "blush"), ("сильнее", 37.55, 420, 1300, 4, "ink")):
        A(plaque(txt, x, y, t0, 38.40, st, a, 46))
    # for my child — anything — even kill
    A(Typed("ради своего ребёнка", 90, 1080, 38.75, 41.75, font(50, 300, True), WHITE, 0.04, shadow=True))
    A(big("ВСЁ", CX, 1210, 41.25, 41.75, 210))
    A(plaque("даже убить.", CX, 1270, 42.13, 43.60, "ink", -2, 60, "pop"))
    # no weekends — no holidays
    A(plaque("(1) нет выходных", 270, 820, 44.64, 46.70, "blush", -2, 44))
    A(plaque("(2) нет отпусков", 270, 1080, 45.60, 46.70, "ink", 2, 44))
    # acceptance
    A(big("ПРИНЯТИЕ", CX, 1195, 50.38, END, 128, BROWN, 0.9, WHITE))
    A(Fn(progress_fn(50.5, 51.9), 50.45, END))
    return sc, el

def days_fn(cv, t):
    d = ImageDraw.Draw(cv); f = font(38, 300); days = "пн вт ср чт пт сб вс".split(); k = min(int((t - 18.6) / 0.2), 6)
    xs = [CX - 3 * 112 + i * 112 for i in range(7)]
    for i, (x, s) in enumerate(zip(xs, days)):
        if i == k: d.ellipse((x - 38, 1215 - 38, x + 38, 1215 + 38), fill=BLUSH + (255,))
        d.text((x, 1215), s, font=f, fill=(INK if i <= k else MAUVE) + (255,), anchor="mm")

def progress_fn(t0, t1):
    def fn(cv, t):
        d = ImageDraw.Draw(cv); k = ease((t - t0) / (t1 - t0)); x0, x1, y = 150, 790, 1310
        sh = Image.new("RGBA", cv.size, (0, 0, 0, 0)); ds = ImageDraw.Draw(sh); ds.rounded_rectangle((x0, y - 16, x1, y + 16), 16, fill=(20, 14, 10, 90))
        cv.alpha_composite(sh.filter(ImageFilter.GaussianBlur(8)))
        d.rounded_rectangle((x0, y - 16, x1, y + 16), 16, fill=CREAM + (210,), outline=WHITE + (255,), width=2)
        if k > 0.02: d.rounded_rectangle((x0 + 4, y - 12, x0 + 4 + (x1 - x0 - 8) * k, y + 12), 12, fill=BLUSH + (255,))
        shadow_text(cv, (x1 + 22, y + 11), f"{int(100 * k)}%", font(30, 350), "ls"); d.text((x1 + 22, y + 11), f"{int(100 * k)}%", font=font(30, 350), fill=WHITE + (255,), anchor="ls")
    return fn

# ── render ──────────────────────────────────────────────────────────────────────────────────────────────────────────
WHIPS = [11.75, 12.20, 12.65, 13.05, 13.50, 20.00, 34.50, 41.75, 43.60]

def background(s, t, live):
    lt = t - s["t0"]; i = int(round(lt * FPS)); k = s["kind"]; dur = s["t1"] - s["t0"]
    if k == "clip":
        if "st" not in live: live["st"] = s["spec"](dur)
        a = live["st"].frame(i); a = zoom(a, s.get("z", 1.0) + 0.02 * lt / dur); return treat(a, s.get("mode"))
    if k == "paper": return paper()
    if k in ("window", "days"):
        bg = paper(); cv = Image.fromarray(bg).convert("RGBA")
        specs = s["spec"] if k == "days" else [s["spec"]]; j = min(int(lt / 0.35), len(specs) - 1) if k == "days" else 0
        key = f"st{j}"
        if key not in live: live[key] = specs[j](0.6 if k == "days" else dur)
        fr = live[key].frame(i - (int(j * 0.35 * FPS) if k == "days" else 0))
        w, h = fr.shape[1], fr.shape[0]; win, off = window_frame(w, h)
        win.paste(Image.fromarray(fr), off); win = shadowed(win, 22, 0.25, (0, 14))
        kk = ease(lt / 0.3); y = (330 if k == "window" else 380) + (1 - kk) * 120
        cv.alpha_composite(win, (int(CX - win.width / 2), int(y - 44)))
        return np.asarray(cv.convert("RGB")).copy()
    if k == "split":
        if "a" not in live: live["a"] = s["spec"][0](dur); live["b"] = s["spec"][1](dur)
        out = np.empty((H, W, 3), np.uint8); out[:] = CREAM
        out[:950] = live["a"].frame(i); out[970:] = live["b"].frame(i); return out
    if k == "zoomblur":
        if "img" not in live:
            src, ss, cx = s["spec"]; st = Stream(src, ss, 0.2, cx=cx); f0 = st.frame(0).copy(); st.close()
            g = treat(f0, "bw").astype(np.float32); live["img"] = np.clip(cv2.GaussianBlur(g, (0, 0), 22) * 0.62, 0, 255).astype(np.uint8)
            live["sharp"] = treat(f0, "bw")
        z = 1.0 + 0.25 * (1 - ease(lt / 0.45)); base = live["img"] if lt > 0.25 else live["sharp"]
        a = zoom(base, z, 0.5)
        if lt < 0.45:                                        # radial zoom blur at the start
            acc = a.astype(np.float32)
            for q in (1.04, 1.08, 1.12): acc += zoom(base, z * q, 0.5)
            a = (acc / 4).astype(np.uint8)
        return a
    raise ValueError(k)

def whip(a, t):
    for c in WHIPS:
        d = t - c
        if abs(d) < 0.12:
            s = 1 - abs(d) / 0.12; k = int(1 + s * 150)
            a = cv2.blur(a, (k, 1)); a = np.roll(a, int(np.sign(d or 1) * s * 70), axis=1)
    return a

def render(M, C, words, out, preview=None):
    cues = load_words(words); sc, el = build(M, C)
    times = [old_of_new(x) for x in (preview if preview else [i / FPS for i in range(int(NEW_END * FPS))])]
    enc = None if preview else subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                                 "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                                                 "-pix_fmt", "yuv420p", out + ".video.mp4"], stdin=subprocess.PIPE)
    live = {}; cur = None; shots = []
    for t in times:
        s = next(x for x in sc if x["t0"] <= t < x["t1"] or (x is sc[-1] and t >= x["t0"]))
        if s is not cur:
            for v in live.values():
                if isinstance(v, Stream): v.close()
            live = {}; cur = s
        a = background(s, t, live)
        cv = Image.fromarray(a).convert("RGBA")
        for e in el: e.draw(cv, t)
        subs_draw(cv, t, cues, ink=s["kind"] in ("paper", "window", "days"))
        a = np.asarray(cv.convert("RGB"))
        a = whip(a, t)
        if t > END - 0.6: k = (t - (END - 0.6)) / 0.6; a = (a * (1 - k) + np.array(CREAM) * k).astype(np.uint8)
        if preview: shots.append(Image.fromarray(a))
        else: enc.stdin.write(np.ascontiguousarray(a).tobytes())
    if preview:
        sheet = Image.new("RGB", (len(shots) * 360, 640))
        for i, im in enumerate(shots): sheet.paste(im.resize((360, 640)), (i * 360, 0))
        sheet.save(out); return
    enc.stdin.close(); enc.wait()
    sound(M, out + ".voice.wav", MUSIC)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out + ".video.mp4", "-i", out + ".voice.wav", "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out], check=True)
    os.remove(out + ".video.mp4"); os.remove(out + ".voice.wav")

def sfx_events():
    """(old-timeline time, sound) for every visual event"""
    import sdvg_audio as A
    ev = []
    for t in WHIPS: ev.append((t - 0.16, A.whoosh(0.34, gain=0.42)))
    for t in (18.6, 23.4, 26.75, 29.2): ev.append((t - 0.1, A.whoosh(0.3, 400, 3000, 0.22)))
    ev.append((31.85, A.whoosh(0.5, 200, 2500, 0.45)))                              # zoom-blur
    for t in (3.30, 4.69, 5.49, 5.97, 8.71, 10.95, 12.71, 20.03, 23.39, 34.63, 35.79, 36.87, 37.55, 44.64, 45.60):
        ev.append((t, A.pop()))
    for t in (2.00, 11.85, 17.45, 25.91, 29.73, 33.27, 41.25, 50.38): ev.append((t, A.swell()))
    ev += [(7.05, A.boom(0.45)), (42.13, A.boom(0.75)), (7.30, A.tick(0.2, 1600))]
    for t0, text, step in ((9.70, "не хочется", 0.05), (10.30, "вставать", 0.05), (17.39, "некому", 0.06), (24.73, "свою", 0.05),
                           (27.55, "маленький человечек", 0.04), (38.75, "ради своего ребёнка", 0.04)):
        ev += [(t0 + i * step, A.key()) for i, c in enumerate(text) if c != " "]
    ev += [(14.59, A.bloop()), (15.31, A.bloop()), (17.6, A.error()), (26.73, A.shutter()), (27.55, A.shutter()),
           (13.50, A.whoosh(0.45, 200, 1800, 0.3)), (29.40, A.whoosh(0.45, 200, 1800, 0.3)), (30.60, A.ding()), (22.47, A.shimmer(0.1, 1.4))]
    ev += [(18.6 + i * 0.2, A.tick(0.16, 1800 + i * 90)) for i in range(7)]
    ev += [(50.5 + i * 0.14, A.tick(0.1, 1500 + i * 120)) for i in range(10)] + [(51.9, A.shimmer(0.14, 1.3))]
    return ev

MUSIC = None

def sound(M, out, music):
    import sdvg_audio as A
    v = out + ".v.wav"; A.studio_voice(os.path.join(M, "Новая запись 9.m4a"), NRUNS, GAPS, NEW_END, v)
    ev = [(new_of_old(t), x) for t, x in sfx_events()]
    A.mix(v, music[0], music[1], ev, NEW_END, out); os.remove(v)

def voice(src, out):
    parts = "".join(f"[0:a]atrim={a}:{b},asetpts=PTS-STARTPTS,afade=t=in:d=0.03,afade=t=out:st={b - a - 0.06:.3f}:d=0.06[a{i}];"
                    for i, (a, b) in enumerate(SEGS))
    fc = parts + "".join(f"[a{i}]" for i in range(len(SEGS))) + f"concat=n={len(SEGS)}:v=0:a=1,loudnorm=I=-15:TP=-1.5:LRA=11,apad=whole_dur={END}[o]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex", fc, "-map", "[o]", "-ar", "48000", "-ac", "2", "-t", str(END), out], check=True)

if __name__ == "__main__":
    a = sys.argv; pv = [float(x) for x in a[a.index("--preview") + 1].split(",")] if "--preview" in a else None
    if "--music" in a: MUSIC = (a[a.index("--music") + 1], float(a[a.index("--music") + 2]))
    render(a[1], a[2], a[3], a[4], pv)
