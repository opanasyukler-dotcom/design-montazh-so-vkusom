"""4 Instagram stories (1080x1920): lipofilling storytelling — precious fat, buttocks results, repost, consultation CTA.

Fonts: Inter Bold caps headlines (the bold face from the reference story), Montserrat Light body,
Montserrat Medium subtitles on the slate plate from the reference.  Brand palette from the surgeon's
posts: navy of her logo + white.  Subtitles are hand-timed cues built from the word timings of
the voice clips; lilac/powder arrows mark the zones on the before/after collages.

usage: python3 stories_lipofilling_fat.py SRC_DIR OUT_DIR [story numbers...]
"""
import json, os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except ImportError:
    pass

W, H, FPS = 1080, 1920, 30
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
MONT_I = os.path.join(ROOT, "fonts/Montserrat-Italic-VariableFont_wght.ttf")
BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

WHITE = (255, 255, 255)
LILAC = (252, 211, 255)
POWDER = (233, 207, 199)
GRAPHITE = (20, 23, 31)
NAVY = (2, 31, 71)             # the surgeon's colour: navy of her sticker/logo, with white
SLATE = (78, 86, 112)          # subtitle plate from the reference story

SRC, OUT = sys.argv[1], sys.argv[2]
ONLY = {int(a) for a in sys.argv[3:]}
ASR = os.path.join(SRC, "asr")  # faster-whisper word timings: [{w: [[start, end, word], ...]}, ...]
os.makedirs(OUT, exist_ok=True)
MARGIN = 84


def mont(size, wght=300, italic=False):
    f = ImageFont.truetype(MONT_I if italic else MONT, size); f.set_variation_by_axes([wght]); return f


def bold(size): return ImageFont.truetype(BOLD, size)


def wrap(text, f, width):
    out, line = [], ""
    for w in text.split(" "):
        t = (line + " " + w).strip()
        if f.getlength(t) <= width or not line: line = t
        else: out.append(line); line = w
    return out + [line]


def heart(size):
    f = ImageFont.truetype(EMOJI, 109)
    im = Image.new("RGBA", (136, 128), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), "❤️", font=f, embedded_color=True)
    im = im.crop(im.getbbox())
    return im.resize((int(im.width * size / im.height), size), Image.LANCZOS)


def ease(x): x = min(max(x, 0.0), 1.0); return 1 - (1 - x) ** 3


# ---------------- text items: each is its own small RGBA sprite with an appear time ----------------
class Sprite:
    def __init__(self, img, x, y, t0, dur=0.6, dy=28):
        self.a = np.asarray(img, np.float32) / 255.0; self.x, self.y = int(x), int(y)
        self.t0, self.dur, self.dy = t0, dur, dy

    def draw(self, fr, t):
        k = ease((t - self.t0) / self.dur)
        if k <= 0: return
        y = self.y + int(round(self.dy * (1 - k)))
        h, w = self.a.shape[:2]
        y0, y1, x0, x1 = max(y, 0), min(y + h, H), max(self.x, 0), min(self.x + w, W)
        if y1 <= y0 or x1 <= x0: return
        s = self.a[y0 - y:y1 - y, x0 - self.x:x1 - self.x]
        al = s[..., 3:4] * k
        fr[y0:y1, x0:x1] = fr[y0:y1, x0:x1] * (1 - al) + s[..., :3] * al


def text_img(lines, f, col=WHITE, lh=1.22, align="center", width=W - 2 * MARGIN, shadow=True, tail=None):
    step = int(f.size * lh); pad = 24
    hh = step * len(lines) + 2 * pad
    im = Image.new("RGBA", (width + 2 * pad, hh), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        lw = f.getlength(ln) + ((tail.width + 14) if (tail is not None and i == len(lines) - 1) else 0)
        x = pad + (width - lw) / 2 if align == "center" else (pad + width - lw if align == "right" else pad)
        d.text((x, pad + i * step + step / 2), ln, font=f, fill=col + (255,), anchor="lm")
        if tail is not None and i == len(lines) - 1:
            im.alpha_composite(tail, (int(x + f.getlength(ln) + 14), int(pad + i * step + step / 2 - tail.height / 2)))
    if shadow:
        sh = Image.new("RGBA", im.size, GRAPHITE + (0,))
        sh.putalpha(im.getchannel("A").point(lambda v: v * 150 // 255))
        im = Image.alpha_composite(sh.filter(ImageFilter.GaussianBlur(9)), im)
    return im, pad


class Column:
    """vertical stack laid out once; every item appears at its own time"""
    def __init__(self, y, align="center", width=W - 2 * MARGIN):
        self.y = y; self.align = align; self.width = width; self.items = []; self.bottom = y

    def _add(self, im, pad, t0, gap):
        x = (W - im.width) // 2 if self.align == "center" else MARGIN - pad
        self.items.append(Sprite(im, x, self.y - pad, t0))
        self.y += im.height - 2 * pad + gap; self.bottom = self.y

    def head(self, text, t0, size=66, gap=26, col=WHITE, with_heart=False):
        f = bold(size); tail = heart(int(size * 0.9)) if with_heart else None
        lines = wrap(text, f, self.width - (tail.width + 14 if tail else 0))
        im, pad = text_img(lines, f, col, 1.14, self.align, self.width, tail=tail); self._add(im, pad, t0, gap); return self

    def body(self, text, t0, size=42, gap=22, wght=350, col=WHITE, italic=False):
        f = mont(size, wght, italic); im, pad = text_img(wrap(text, f, self.width), f, col, 1.3, self.align, self.width)
        self._add(im, pad, t0, gap); return self

    def rule(self, t0, length=150, gap=30):
        im = Image.new("RGBA", (length, 4), WHITE + (255,))
        x = {"left": MARGIN, "right": W - MARGIN - length}.get(self.align, (W - length) // 2)
        self.items.append(Sprite(im, x, self.y, t0, dy=0)); self.y += 4 + gap; self.bottom = self.y; return self

    def icon_row(self, icon, text, t0, size=38, gap=26, ih=170):
        ic = icon.resize((int(icon.width * ih / icon.height), ih), Image.LANCZOS)
        f = mont(size, 400); tw = self.width - ic.width - 30
        tim, pad = text_img(wrap(text, f, tw), f, WHITE, 1.3, "left", tw)
        hh = max(ic.height, tim.height - 2 * pad)
        im = Image.new("RGBA", (self.width + 2 * pad, hh + 2 * pad), (0, 0, 0, 0))
        im.alpha_composite(ic, (pad, pad + (hh - ic.height) // 2))
        im.alpha_composite(tim, (pad + ic.width + 30 - pad, pad + (hh - (tim.height - 2 * pad)) // 2 - pad))
        self._add(im, pad, t0, gap); return self

    def space(self, h): self.y += h; self.bottom = self.y; return self

    def to_bottom(self, yb):
        """move the whole block so its last line ends at yb; returns the new top"""
        d = int(yb - self.bottom)
        for it in self.items: it.y += d
        self.top = self.items[0].y if self.items else yb; self.bottom += d; return self


# ---------------- subtitles ----------------
def load_words(name, a, b):
    d = json.load(open(os.path.join(ASR, name + ".json")))
    ws = [(w[0], w[1], w[2].strip()) for s in d for w in s["w"] if w[0] >= a - 0.05 and w[1] <= b + 0.3]
    return [(s - a, e - a, t) for s, e, t in ws]


def chunks(words, maxc=34):
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w); txt = " ".join(x[2] for x in cur)
        nxt = words[i + 1] if i + 1 < len(words) else None
        brk = nxt is None or len(txt) + 1 + len(nxt[2]) > maxc or w[2][-1] in ".?!…" \
            or (nxt[0] - w[1] > 0.45) or (w[2][-1] == "," and len(txt) > 18)
        if brk: out.append(cur); cur = []
    res = []
    for i, c in enumerate(out):
        end = out[i + 1][0][0] if i + 1 < len(out) else c[-1][1] + 0.6
        end = min(end, c[-1][1] + 0.7)
        res.append((c[0][0], end, " ".join(x[2] for x in c).replace(" -", "-")))
    return res


def sub_sprite(text, y=1645):
    f = mont(42, 500); lines = wrap(text, f, W - 220)
    step = int(42 * 1.28); px, py = 28, 14
    wmax = max(f.getlength(l) for l in lines); hh = step * len(lines) + 2 * py
    im = Image.new("RGBA", (int(wmax + 2 * px), hh), SLATE + (225,)); d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        d.text((im.width / 2, py + i * step + step / 2), l, font=f, fill=WHITE + (255,), anchor="mm")
    return np.asarray(im, np.float32) / 255.0, (W - im.width) // 2, y - hh // 2


class Subs:
    def __init__(self, cues, fixes=None):
        fixes = fixes or {}
        self.cues = []
        for s, e, t in cues:
            for k, v in fixes.items(): t = t.replace(k, v)
            self.cues.append((s, e, sub_sprite(t[0].upper() + t[1:])))

    def draw(self, fr, t):
        for s, e, (a, x, y) in self.cues:
            if s - 0.02 <= t < e:
                k = min(1.0, (t - s) / 0.12)
                h, w = a.shape[:2]; al = a[..., 3:4] * k
                fr[y:y + h, x:x + w] = fr[y:y + h, x:x + w] * (1 - al) + a[..., :3] * al


# ---------------- backgrounds ----------------
def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:stream_side_data=rotation",
                        "-of", "json", path], capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]; w, h = s["width"], s["height"]
    rot = [d.get("rotation", 0) for d in s.get("side_data_list", [])]
    if rot and abs(rot[0]) == 90: w, h = h, w
    return w, h


class Clip:
    """9:16 window of a video segment. ct = fraction of the frame height cut from the top (face crop)."""
    def __init__(self, name, a, b, ct=0.0, cx=0.5, zoom=(1.0, 1.04), zc=(0.5, 0.45), frost=0, sat=1.0):
        self.path = os.path.join(SRC, "drive", name); self.name = name
        self.a, self.b = a, b; self.ct, self.cx, self.zoom, self.zc, self.frost = ct, cx, zoom, zc, frost
        w, h = probe(self.path)
        ch = h * (1 - ct); cw = ch * 9 / 16
        if cw > w: cw = w; ch = cw * 16 / 9
        x = (w - cw) * cx; y = h * ct
        vf = f"crop={int(cw)}:{int(ch)}:{int(x)}:{int(y)},scale={W}:{H}:flags=lanczos,fps={FPS}"
        if sat != 1.0: vf += f",eq=saturation={sat}"
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a), "-i", self.path,
                                   "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.last = None; self.n = int(round((b - a) * FPS))

    def frame(self, i):
        buf = self.p.stdout.read(W * H * 3)
        if len(buf) == W * H * 3: self.last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        fr = self.last
        z = self.zoom[0] + (self.zoom[1] - self.zoom[0]) * i / max(1, self.n)
        if abs(z - 1) > 1e-3:
            M = cv2.getRotationMatrix2D((W * self.zc[0], H * self.zc[1]), 0, z)
            fr = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if self.frost: fr = frosted(fr, self.frost)
        return fr


def frosted(fr, band):
    """heavy blur on the top band (feathered) - nothing of a face survives there"""
    top = fr[:band + 160]
    bl = cv2.GaussianBlur(cv2.resize(top, (W // 8, (band + 160) // 8)), (0, 0), 3)
    bl = cv2.resize(bl, (W, band + 160), interpolation=cv2.INTER_LINEAR)
    m = np.clip((band + 160 - np.arange(band + 160, dtype=np.float32)) / 160, 0, 1)[:, None, None]
    out = fr.copy(); out[:band + 160] = (top * (1 - m) + bl * m).astype(np.uint8); return out


def load_img(name): return ImageOps.exif_transpose(Image.open(os.path.join(SRC, "drive", name))).convert("RGB")


def crop_frac(im, cx, cy, cw, aspect):
    """crop by fractions: centre (cx, cy), width cw (of image width), given w/h aspect"""
    w = im.width * cw; h = w / aspect
    x0 = im.width * cx - w / 2; y0 = im.height * cy - h / 2
    x0 = min(max(x0, 0), im.width - w); y0 = min(max(y0, 0), im.height - h)
    return im.crop((int(x0), int(y0), int(x0 + w), int(y0 + h)))


def vgrad(y0, y1, a0, a1):
    ys = np.arange(H, dtype=np.float32); p = np.clip((ys - y0) / (y1 - y0), 0, 1); p = p * p * (3 - 2 * p)
    return (a0 + (a1 - a0) * p)[:, None, None]


def chip(text, fill=LILAC):
    f = bold(30); w = int(f.getlength(text)) + 44
    im = Image.new("RGBA", (w, 56), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, 55), radius=28, fill=fill + (240,))
    d.text((w / 2, 29), text, font=f, fill=GRAPHITE + (255,), anchor="mm"); return im


# ---------------- render core ----------------
def render(num, slug, dur, bg, sprites, shade=None, subs=None, audio=None, animate=False):
    """bg(i, t) -> uint8 frame; shade: per-row darkening (H,1,1) or fn(t); audio: (file, a, b).
    Text stands still from the first frame unless animate=True."""
    if not animate:
        for sp in sprites: sp.t0, sp.dur = -1.0, 0.01
    out = os.path.join(OUT, f"{num:02d}_{slug}.mp4"); tmp = out + ".v.mp4"
    n = int(round(dur * FPS))
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
                           stdin=subprocess.PIPE)
    still_t = max(s.t0 + s.dur for s in sprites) + 0.3 if sprites else dur / 2
    still_i = min(n - 1, int(max(still_t, 2.0) * FPS))
    preview = os.environ.get("PREVIEW")
    for i in range(n):
        t = i / FPS
        if preview and i != still_i:
            bg(i, t); continue
        fr = bg(i, t).astype(np.float32) / 255.0
        if shade is not None:
            sh = shade(t) if callable(shade) else shade
            fr = fr * (1 - sh)                                    # plain black, no tint
        for s in sprites: s.draw(fr, t)
        if subs: subs.draw(fr, t)
        o = (np.clip(fr, 0, 1) * 255).astype(np.uint8)
        enc.stdin.write(o.tobytes())
        if i == still_i: Image.fromarray(o).save(out[:-4] + ".jpg", quality=92)
        if preview: break
    enc.stdin.close(); enc.wait()
    if preview:
        os.remove(tmp); print("preview", out[:-4] + ".jpg", flush=True); return
    if audio:
        f, a, b = audio
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", str(a), "-t", str(b - a), "-i", os.path.join(SRC, "drive", f),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af",
                        f"loudnorm=I=-16:TP=-1.5,afade=t=in:d=0.08,afade=t=out:st={b - a - 0.25:.2f}:d=0.25",
                        "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out], check=True)
    else:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c", "copy", "-movflags", "+faststart", out], check=True)
    os.remove(tmp); print("wrote", out, flush=True)


def clip_bg(c): return lambda i, t: c.frame(i)





# ---------------- helpers for this series ----------------
RETOUCH = False   # light retouch (bruise lift + skin softening) - switched off at the client's request


def retouch(rgb, bruise=1.0, smooth=0.45):
    """light retouch: lift purple/red bruises that are darker than the surrounding skin, soften skin a little.
    Large-radius statistics are computed at 1/4 size (fast enough for every video frame)."""
    H0, W0 = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    q = 4; small = cv2.resize(hsv, (W0 // q, H0 // q), interpolation=cv2.INTER_AREA)
    k = max(4.0, min(H0, W0) / 22 / q)
    up = lambda a: cv2.resize(a, (W0, H0), interpolation=cv2.INTER_LINEAR)
    v_loc = up(cv2.GaussianBlur(small[..., 2], (0, 0), k)); s_loc = up(cv2.GaussianBlur(small[..., 1], (0, 0), k))
    sh, ss, sv = small[..., 0], small[..., 1], small[..., 2]
    skin0 = ((sh < 22) & (ss > 35) & (ss < 170) & (sv > 110)).astype(np.float32)        # skin around a bruise
    near_skin = up(np.clip(cv2.GaussianBlur(skin0, (0, 0), k * 0.6) * 3.0, 0, 1))
    purple = np.clip(np.maximum((h - 112) / 10, (9 - h) / 5), 0, 1)
    dark = np.clip((v_loc - v - 2) / 22, 0, 1)
    sat = np.clip((s - 25) / 35, 0, 1)
    bright = np.clip((v - 70) / 30, 0, 1)
    m = np.clip(cv2.GaussianBlur(purple * dark * sat * bright, (0, 0), 5) * 1.6, 0, 1) * near_skin * bruise
    hsv[..., 2] = v + (v_loc - v).clip(0) * m * 0.85
    hsv[..., 1] = s - (s - np.minimum(s, s_loc * 0.7)) * m
    hsv[..., 0] = h + (190 - h) * m * 0.6 * (h > 90) + (10 - h).clip(0) * m * 0.6 * (h <= 90)
    out = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    skin = (((h < 25) | (h > 165)) & (s > 30) & (s < 190) & (v > 70)).astype(np.float32)
    skin = cv2.GaussianBlur(skin, (0, 0), 4)[..., None] * smooth
    sm = cv2.bilateralFilter(out, 9, 40, 9)
    out = (out * (1 - skin) + sm * skin).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


class RClip(Clip):
    """Clip with the retouch applied to every decoded frame (before the slow zoom)"""
    def frame(self, i):
        buf = self.p.stdout.read(W * H * 3)
        if len(buf) == W * H * 3:
            fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            self.last = retouch(fr) if RETOUCH else fr
        fr = self.last
        z = self.zoom[0] + (self.zoom[1] - self.zoom[0]) * i / max(1, self.n)
        if abs(z - 1) > 1e-3:
            M = cv2.getRotationMatrix2D((W * self.zc[0], H * self.zc[1]), 0, z)
            fr = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return fr


def top_shade(bottom, a=0.72):
    ys = np.arange(H, dtype=np.float32)
    p = np.clip((bottom - ys) / 380, 0, 1); p = p * p * (3 - 2 * p)
    return (a * p)[:, None, None]


def bottom_shade(top, a=0.75):
    ys = np.arange(H, dtype=np.float32)
    p = np.clip((ys - top) / 380, 0, 1); p = p * p * (3 - 2 * p)
    return (a * p)[:, None, None]


def cues(lst, a):
    """[(start, end, text)] in source time -> Subs relative to the clip start"""
    return Subs([(s0 - a, e0 - a, t) for s0, e0, t in lst])


_PHOTOS = {}


def photo(name):
    """patient photo with the light retouch (cached); before/after collages are left untouched"""
    if name not in _PHOTOS:
        im = load_img(name)
        if im.height > 2400: im = im.resize((im.width * 2400 // im.height, 2400), Image.LANCZOS)
        _PHOTOS[name] = Image.fromarray(retouch(np.asarray(im))) if RETOUCH else im
    return _PHOTOS[name]


def frame_crop(im, cx, cy, cw):
    """9:16 window: centre (cx, cy) and width cw as fractions of the image"""
    return crop_frac(im, cx, cy, cw, W / H).resize((W, H), Image.LANCZOS)


def still_bg(img, z1=1.04, zc=(0.5, 0.45), dur=10.0):
    a = np.asarray(img.convert("RGB"))
    def bg(i, t):
        z = 1.0 + (z1 - 1.0) * t / dur
        M = cv2.getRotationMatrix2D((W * zc[0], H * zc[1]), 0, z)
        return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return bg


def arrow_img(p0, p1, col, width=9, bend=0.18, head=34):
    """curved arrow from p0 to p1 (pixels), drawn 3x supersampled; returns full-frame RGBA sprite"""
    s = 3; im = Image.new("RGBA", (W * s, H * s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2; dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - dy * bend, my + dx * bend
    pts = [((1 - u) ** 2 * x0 + 2 * (1 - u) * u * cx + u * u * x1, (1 - u) ** 2 * y0 + 2 * (1 - u) * u * cy + u * u * y1)
           for u in np.linspace(0, 0.93, 40)]
    sh = [(x * s + 3 * s, y * s + 3 * s) for x, y in pts]
    d.line(sh, fill=GRAPHITE + (110,), width=(width + 4) * s, joint="curve")
    d.line([(x * s, y * s) for x, y in pts], fill=col + (255,), width=width * s, joint="curve")
    tx, ty = pts[-1]; ang = np.arctan2(y1 - ty, x1 - tx)
    hp = [(x1, y1), (x1 - head * np.cos(ang - 0.45), y1 - head * np.sin(ang - 0.45)),
          (x1 - head * np.cos(ang + 0.45), y1 - head * np.sin(ang + 0.45))]
    d.polygon([(x * s, y * s) for x, y in hp], fill=col + (255,))
    im = im.resize((W, H), Image.LANCZOS)
    bb = im.getbbox(); crop = im.crop(bb)
    return Sprite(crop, bb[0], bb[1], 0, dy=0)


def at(sp, t0, dur=0.6):
    sp.t0, sp.dur = t0, dur; return sp


def low(col, yb=1560): return col.to_bottom(yb)



def erase_band(y0, y1, a=1.0, feather=45):
    """opaque graphite band over the text burnt into the old story videos (y in output px)"""
    ys = np.arange(H, dtype=np.float32)
    p = np.clip(np.minimum(ys - (y0 - feather), (y1 + feather) - ys) / feather, 0, 1)  # fully opaque inside [y0, y1]
    return (a * p)[:, None, None]


FIXES = {"грудное оскармливание": "грудное вскармливание", "Оно смущало": "Смущало", "грудную оскарму": "грудное вскармливание", "скупулезно": "скрупулёзно", "и нам мы ориентируемся": "и мы ориентируемся",
         "и мне и хотелось": "и мне хотелось", "кориолы": "ареолы"}


FIXES = {"желтенький,": "жёлтенький,", "идет,": "идёт,", "идет": "идёт"}


class WordSubs:
    """one word at a time, lower case, Montserrat Light, centred at y"""
    def __init__(self, name, y=990, size=50):
        d = json.load(open(os.path.join(ASR, name + ".json")))
        ws = []
        for s in d:
            for a, b, t in s["w"]:
                t = t.strip()
                if ws and t.startswith("-"): ws[-1] = (ws[-1][0], b, ws[-1][2] + t); continue
                ws.append((a, b, t))
        f = mont(size, 300); self.items = []
        for i, (a, b, t) in enumerate(ws):
            t = FIXES.get(t, t).lower()
            e = min(ws[i + 1][0] if i + 1 < len(ws) else b + 0.5, b + 0.45)
            im, pad = text_img([t], f, WHITE, 1.2, "center", W - 120)
            self.items.append((a, e, np.asarray(im, np.float32) / 255.0, (W - im.width) // 2, y - im.height // 2))

    def draw(self, fr, t):
        for a, e, arr, x, y in self.items:
            if a - 0.03 <= t < e:
                h, w = arr.shape[:2]; al = arr[..., 3:4]
                fr[y:y + h, x:x + w] = fr[y:y + h, x:x + w] * (1 - al) + arr[..., :3] * al
                return


def plate(col, pad=26, alpha=0.8, r=28):
    """compact navy plate right behind a text block (no haze over the picture)"""
    ys = [it.y for it in col.items]; ye = [it.y + it.a.shape[0] for it in col.items]
    xs = [it.x for it in col.items]; xe = [it.x + it.a.shape[1] for it in col.items]
    x0, y0, x1, y1 = min(xs) + 24 - pad, min(ys) + 24 - pad, max(xe) - 24 + pad, max(ye) - 24 + pad
    im = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, im.width - 1, im.height - 1), r, fill=NAVY + (int(255 * alpha),))
    col.items.insert(0, Sprite(im, x0, y0, 0, dy=0)); return col


def arrow_on(im, p0, p1, col=WHITE, width=6, bend=0.18, head=22):
    """curved arrow drawn onto an RGBA image (3x supersampled)"""
    s = 3; lay = Image.new("RGBA", (im.width * s, im.height * s), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2; dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - dy * bend, my + dx * bend
    pts = [((1 - u) ** 2 * x0 + 2 * (1 - u) * u * cx + u * u * x1, (1 - u) ** 2 * y0 + 2 * (1 - u) * u * cy + u * u * y1)
           for u in np.linspace(0, 0.9, 40)]
    d.line([(x * s + 2 * s, y * s + 2 * s) for x, y in pts], fill=(0, 0, 0, 120), width=(width + 3) * s, joint="curve")
    d.line([(x * s, y * s) for x, y in pts], fill=col + (255,), width=width * s, joint="curve")
    tx, ty = pts[-1]; ang = np.arctan2(y1 - ty, x1 - tx)
    hp = [(x1, y1), (x1 - head * np.cos(ang - 0.45), y1 - head * np.sin(ang - 0.45)),
          (x1 - head * np.cos(ang + 0.45), y1 - head * np.sin(ang + 0.45))]
    d.polygon([(x * s, y * s) for x, y in hp], fill=col + (255,))
    im.alpha_composite(lay.resize(im.size, Image.LANCZOS)); return im


class VideoCircle:
    """a round video window (white ring) drawn over everything; sequential frames, looped"""
    def __init__(self, name, a, b, crop, d, x, y, ring=6):
        self.t0, self.dur = -1.0, 0.01
        self.d, self.x, self.y, self.ring = d, x, y, ring
        vf = "pad=3000:3840:420:0:black,crop=3000:3000:0:420" if crop is None else "crop={}:{}:{}:{}".format(*crop)
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a), "-i", os.path.join(SRC, "drive", name),
                            "-vf", f"{vf},scale={d}:{d}:flags=lanczos,fps={FPS}", "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        self.frames = np.frombuffer(p, np.uint8).reshape(-1, d, d, 3)
        if crop is None: self.frames = self.cut_bg(name, a, b, d)
        D = d + 2 * ring; yy, xx = np.mgrid[:D, :D]; r = np.hypot(xx - D / 2 + 0.5, yy - D / 2 + 0.5)
        self.ring_a = np.clip(D / 2 - r, 0, 1)[..., None].astype(np.float32)
        self.in_a = np.clip(d / 2 - np.hypot(xx[:d, :d] - d / 2 + 0.5, yy[:d, :d] - d / 2 + 0.5), 0, 1)[..., None].astype(np.float32)

    def cut_bg(self, name, a, b, d):
        """remove the backdrop from every frame (rembg isnet), body on pure black; cached next to the sources"""
        cache = os.path.join(SRC, "cut", f"{name}_{a}_{b}_{d}.npy")
        if os.path.exists(cache): return np.load(cache)
        from rembg import remove, new_session
        sess = new_session("isnet-general-use"); out = []
        for f in self.frames:
            m = np.asarray(remove(Image.fromarray(f), session=sess, only_mask=True), np.float32)[..., None] / 255
            m = np.clip((m - 0.12) / 0.3, 0, 1)                   # arms/hands fully opaque, soft edge kept
            out.append((f * m).astype(np.uint8))
        out = np.stack(out); np.save(cache, out); return out

    def draw(self, fr, t):
        i = int(t * FPS) % len(self.frames)
        D = self.d + 2 * self.ring; x, y = self.x, self.y
        fr[y:y + D, x:x + D] = fr[y:y + D, x:x + D] * (1 - self.ring_a) + self.ring_a
        v = self.frames[i].astype(np.float32) / 255; xi, yi = x + self.ring, y + self.ring
        fr[yi:yi + self.d, xi:xi + self.d] = fr[yi:yi + self.d, xi:xi + self.d] * (1 - self.in_a) + v * self.in_a


def card(img, w, h, radius=26, border=4):
    im = img.convert("RGB").resize((w, h), Image.LANCZOS)
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius, fill=255)
    c = Image.new("RGBA", (w + 2 * border, h + 2 * border), (0, 0, 0, 0))
    ImageDraw.Draw(c).rounded_rectangle((0, 0, c.width - 1, c.height - 1), radius + border, fill=WHITE + (255,))
    im.putalpha(m); c.alpha_composite(im, (border, border)); return c


def navy_chip(text, size=30):
    f = bold(size); w = int(f.getlength(text)) + 40
    im = Image.new("RGBA", (w, size + 26), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, size + 25), radius=(size + 26) // 2, fill=NAVY + (235,))
    d.text((w / 2, (size + 26) / 2 + 1), text, font=f, fill=WHITE + (255,), anchor="mm"); return im


def small_label(text, size=30, wght=400):
    f = mont(size, wght); im, pad = text_img([text], f, WHITE, 1.2, "center", int(f.getlength(text)) + 10); return im


def arrow_thin(im, p0, p1, col=WHITE, width=3, head=13):
    """small straight minimalist arrow"""
    sc = 3; lay = Image.new("RGBA", (im.width * sc, im.height * sc), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    (x0, y0), (x1, y1) = p0, p1; ang = np.arctan2(y1 - y0, x1 - x0)
    xe, ye = x1 - head * 0.7 * np.cos(ang), y1 - head * 0.7 * np.sin(ang)
    for off, c in (((2, 2), (0, 0, 0, 110)), ((0, 0), col + (255,))):
        d.line([((x0 + off[0]) * sc, (y0 + off[1]) * sc), ((xe + off[0]) * sc, (ye + off[1]) * sc)], fill=c, width=width * sc)
        hp = [(x1, y1), (x1 - head * np.cos(ang - 0.42), y1 - head * np.sin(ang - 0.42)),
              (x1 - head * np.cos(ang + 0.42), y1 - head * np.sin(ang + 0.42))]
        d.polygon([((x + off[0]) * sc, (y + off[1]) * sc) for x, y in hp], fill=c)
    im.alpha_composite(lay.resize(im.size, Image.LANCZOS)); return im


def label_lines(im, lines, x, y, size=26, wght=500, anchor="ls", gap=1.25):
    f = mont(size, wght); d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        yy = y + i * size * gap
        d.text((x + 2, yy + 2), t, font=f, fill=(0, 0, 0, 150), anchor=anchor)
        d.text((x, yy), t, font=f, fill=WHITE + (255,), anchor=anchor)


def pill(im, text, cx, cy, size=23):
    """small dark rounded label right next to the zone it names; returns its box"""
    f = mont(size, 500); tw = f.getlength(text); w, h = int(tw + 30), int(size * 1.75)
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    d.rounded_rectangle((x0, y0, x0 + w, y0 + h), h // 2, fill=(0, 0, 0, 150))
    d.text((cx, cy + 1), text, font=f, fill=WHITE + (255,), anchor="mm")
    im.alpha_composite(lay); return (x0, y0, x0 + w, y0 + h)


def arrow_from(im, box, tgt, gap=7, **kw):
    """short arrow that starts just outside a label box (never inside the text) and ends at the target"""
    x0, y0, x1, y1 = box; tx, ty = tgt
    cx, cy = min(max(tx, x0), x1), min(max(ty, y0), y1)          # nearest point of the box
    if (cx, cy) == (tx, ty): return im
    v = np.array([tx - cx, ty - cy], float); v /= np.linalg.norm(v)
    return arrow_thin(im, (cx + v[0] * gap, cy + v[1] * gap), tgt, **kw)


def black_fade(top, length=320, a=0.78):
    """light black darkening from `top` down to the bottom (stretched, no tint)"""
    return vgrad(top - length, top + 60, 0.0, a)


def interview(num, slug, clip, header, extra=None, y=None, bottom=1720, align="left", width=W - 2 * MARGIN, sub_y=990,
              more=(), head_size=50, fade_top=None):
    """interview story: blurred-face clip, word-by-word subtitles, text over a stretched black fade at the bottom"""
    name = clip
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              os.path.join(SRC, "drive", name)], capture_output=True, text=True).stdout)
    c = Clip(name, 0.0, d, zoom=(1.0, 1.03), zc=(0.5, 0.3))
    col = Column(y or 0, align=align, width=width)
    col.head(header, 0, size=head_size, gap=14)
    if extra: extra(col)
    if y is None: col.to_bottom(bottom)
    else: col.top = col.items[0].y
    if align == "right":
        for it in col.items: it.x = W - MARGIN - it.a.shape[1] + 24
    top = fade_top if fade_top is not None else col.top
    print(f"s{num} text", col.top, col.bottom)
    render(num, slug, d, clip_bg(c), col.items + list(more), shade=black_fade(top),
           subs=WordSubs(name, y=sub_y), audio=(name, 0.0, d))


def emoji(ch, size):
    f = ImageFont.truetype(EMOJI, 109)
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), ch, font=f, embedded_color=True)
    im = im.crop(im.getbbox())
    return im.resize((int(im.width * size / im.height), size), Image.LANCZOS)


# ---------------- stories ----------------
# drive/lipo_blur.mp4 = IMG_4752 0-29 s with the patient's body blurred (tools/body_blur.py); asr/ has its word timings
# drive/IMG_8256..8259.JPG buttocks collages; IMG_4755.MP4 the reel to repost; C0595.MP4 the surgeon (cover, no sound)


def s1():
    col = Column(0, align="left")
    col.head("ПОСМОТРИТЕ, КАКОЙ ДРАГОЦЕННЫЙ ЖИР!", 0, size=62, gap=18)
    col.body("Не надо худеть, если хотите липофилинг", 0, size=40, wght=500, gap=0)
    col.to_bottom(1720)
    d = 29.0
    c = Clip("lipo_blur.mp4", 0.0, d, zoom=(1.0, 1.03), zc=(0.5, 0.3))
    render(1, "dragocennyj_zhir", d, clip_bg(c), col.items, shade=black_fade(col.items[0].y + 40, 360, 0.8),
           subs=WordSubs("lipo_blur.mp4", y=1180), audio=("lipo_blur.mp4", 0.0, d))


def s_breast(num):
    # three breast lipofilling cases (cropped from drive/cases.jpg): ДО | ПОСЛЕ rows on black
    src = load_img("cases.jpg")
    rows = [(213, 607), (962, 1356), (1712, 2105)]; cols = [(96, 639), (680, 1223)]
    bg = Image.new("RGB", (W, H), (0, 0, 0)).convert("RGBA"); d = ImageDraw.Draw(bg)
    f = bold(54); y = 150
    for ln in wrap("ЖИР С ЖИВОТА И БЁДЕР МОЖЕТ СТАТЬ ОБЪЁМОМ ВАШЕЙ ГРУДИ!", f, W - 2 * MARGIN):
        d.text((W / 2, y), ln, font=f, fill=WHITE + (255,), anchor="mt"); y += 64
    pw, gap = 490, 20; x0 = (W - 2 * pw - gap) // 2; y += 50
    fl = mont(26, 500)
    d.text((x0, y), "до", font=fl, fill=WHITE + (255,)); d.text((x0 + pw + gap, y), "после", font=fl, fill=WHITE + (255,)); y += 46
    for ya, yb in rows:
        for j, (xa, xb) in enumerate(cols):
            ph = src.crop((xa, ya, xb, yb)); ph_h = int(pw * ph.height / ph.width)
            bg.alpha_composite(card(ph, pw, ph_h, radius=18, border=0), (x0 + j * (pw + gap), y))
        y += ph_h + gap
    d.text((W / 2, y + 40), "липофилинг груди собственным жиром", font=mont(30, 400), fill=WHITE + (255,), anchor="mt")
    render(num, "zhir_v_grud", 10.0, still_bg(bg.convert("RGB"), z1=1.0), [])


def s2():
    bg = Image.new("RGB", (W, H), (0, 0, 0)).convert("RGBA")
    f = bold(72); d = ImageDraw.Draw(bg)
    d.text((W / 2, 250), "ИЛИ ЯГОДИЦ", font=f, fill=WHITE + (255,), anchor="mm")
    sz, gap = 500, 20; x0 = (W - 2 * sz - gap) // 2; y0 = 360
    for i, n in enumerate(["IMG_8256.JPG", "IMG_8257.JPG", "IMG_8258.JPG", "IMG_8259.JPG"]):
        c = card(load_img(n), sz, sz, radius=22, border=0)
        bg.alpha_composite(c, (x0 + (i % 2) * (sz + gap), y0 + (i // 2) * (sz + gap)))
    render(3, "yagodicy", 10.0, still_bg(bg.convert("RGB"), z1=1.0), [])


def s3():
    # repost: the finished reel, full screen, with its own sound and captions
    out = os.path.join(OUT, "04_repost_rolika.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(SRC, "drive", "IMG_4755.MP4"),
                    "-vf", f"scale={W}:{H}:flags=lanczos,fps={FPS}", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "3", "-i", out, "-frames:v", "1", out[:-4] + ".jpg"], check=True)
    print("wrote", out)


def s4():
    # the surgeon in the OR (drive/surgeon_or_photo2.jpg), photo story; CTA low on the gown, face free
    img = ImageOps.fit(load_img("surgeon_or_photo2.jpg"), (W, H), Image.LANCZOS, centering=(0.5, 0.0))
    col = Column(0, align="center")
    col.head("ПИШИТЕ «КОНСУЛЬТАЦИЯ»,", 0, size=58, gap=14)
    col.body("чтобы разобрать именно ваш случай и то, подойдёт ли вам липофилинг", 0, size=38, wght=500, gap=30)
    col.body("жду вас", 0, size=46, wght=300, gap=0)
    col.to_bottom(1720)
    render(5, "konsultaciya", 10.0, still_bg(img, z1=1.0), col.items, shade=black_fade(col.items[0].y + 40, 380, 0.82))


PHOTO_ONLY = {2, 3, 5}
STORIES = {1: s1, 2: lambda: s_breast(2), 3: s2, 4: s3, 5: s4}
for k, fn in STORIES.items():
    if not ONLY or k in ONLY:
        fn()
        if k in PHOTO_ONLY:
            for f in os.listdir(OUT):
                if f.startswith(f"{k:02d}_") and f.endswith(".mp4"): os.remove(os.path.join(OUT, f))
