"""6 Instagram stories (1080x1920): breast correction without implants — fat grafting only (marking, donor sites, result).

Fonts: Inter Bold caps headlines (the bold face from the reference story), Montserrat Light body,
Montserrat Medium subtitles on the slate plate from the reference.  Brand palette from the surgeon's
posts: navy of her logo + white.  Subtitles are hand-timed cues built from the word timings of
the voice clips; lilac/powder arrows mark the zones on the before/after collages.

usage: python3 stories_breast_fat_only.py SRC_DIR OUT_DIR [story numbers...]
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


FIXES = {"липофиллинг": "липофилинг", "глифе,": "галифе,", "мошеннистая": "морщинистая", "ими": "", "права.": "шрамов.", "об": "в", "идее": "идеале", "приподнимительно.": "приподнимется.", "биодра": "бёдра", "Фланки": "фланги", "объем,": "объём,", "бедра,": "бёдра,", "бедра": "бёдра", "заберем,": "заберём,", "еще": "ещё"}


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


# ---------------- materials ----------------
# drive/v1..v5.mp4 (the client's numbered videos), IMG_1708.JPG the before photo
# cut with tools/cut_segments.py (every cut sits inside a pause, no word clipped):
#   c1 = v2 4.2-33.0   c2 = v1 1.3-27.4   c3 = v1 37.5-44.8 47.4-49.8 52.3-61.9 94.6-97.2 100.6-102.0
#   c4 = v3 without pauses (1.44-8.83 9.02-20.51 20.88-26.51)   c5 = v4 0-31.4
# FINAL: d1..d5 = the same pieces through tools/clean_speech.py — every pause and «а/э» removed, 12 ms fades at the
#   joins, drawn vowels / «ага» / «ммм» / a false start cut (--drawn --cut), the patient's voice pitch+formant
#   shifted wherever she speaks (d1: 4.2-14.75 18.4-23.5 29.15-33.0, d2: 8.9-27.6,
#   d3: 41.55-41.95 48.25-49.8 52.7-54.6 96.85-97.3 100.6-102.0) — exact commands in README
# subtitles: word timings of the ORIGINAL take mapped through the cuts (cleaner than re-recognising the cut), except c2
CUTS = {"c1.mp4": ("v2.mp4", [(4.2, 33.0)]), "c3.mp4": ("v1.mp4", [(37.5, 44.8), (47.4, 49.8), (52.3, 61.9), (94.6, 97.2), (100.6, 102.0)]),
        "c4.mp4": ("v3.mp4", [(1.44, 8.83), (9.02, 20.51), (20.88, 26.51)]), "c5.mp4": ("v4.mp4", [(0.0, 31.4)])}


def words_for(clip):
    wj = os.path.join(SRC, "drive", clip + ".words.json")          # clean_speech.py output: words on the edited timeline
    if os.path.exists(wj): return [(a, b, t) for a, b, t in json.load(open(wj))[0]["w"]]
    if clip not in CUTS:
        d = json.load(open(os.path.join(ASR, clip + ".json")))
        return [(a, b, t.strip()) for s in d for a, b, t in s["w"]]
    src, segs = CUTS[clip]; d = json.load(open(os.path.join(ASR, src + ".json"))); out = []; off = 0.0
    for a0, b0 in segs:
        for s in d:
            for a, b, t in s["w"]:
                if a >= a0 - 0.02 and b <= b0 + 0.05: out.append((a - a0 + off, min(b, b0) - a0 + off, t.strip()))
        off += b0 - a0
    return out


class PillSubs:
    """one word at a time, lower case, Montserrat Light on a navy pill (the surgeon's colour), small pop-in"""
    def __init__(self, clip, y=1000, size=46):
        ws = []
        for a, b, t in words_for(clip):
            if ws and t.startswith("-"): ws[-1] = (ws[-1][0], b, ws[-1][2] + t); continue
            ws.append((a, b, t))
        f = mont(size, 300); self.items = []
        for i, (a, b, t) in enumerate(ws):
            t = FIXES.get(t, t).lower()
            if not t: continue
            e = min(ws[i + 1][0] if i + 1 < len(ws) else b + 0.5, b + 0.45)
            tw = f.getlength(t); w, h = int(tw + 44), int(size * 1.7)
            im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); dd = ImageDraw.Draw(im)
            dd.rounded_rectangle((0, 0, w - 1, h - 1), h // 2, fill=NAVY + (225,))
            dd.text((w / 2, h / 2 + 1), t, font=f, fill=WHITE + (255,), anchor="mm")
            self.items.append((a, e, np.asarray(im, np.float32) / 255.0, y))

    def draw(self, fr, t):
        for a, e, arr, y in self.items:
            if a - 0.03 <= t < e:
                k = min(1.0, (t - a + 0.03) / 0.12); sc = 0.85 + 0.15 * k
                a2 = cv2.resize(arr, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR) if sc < 0.999 else arr
                h, w = a2.shape[:2]; x, yy = (W - w) // 2, y - h // 2
                al = a2[..., 3:4] * k
                fr[yy:yy + h, x:x + w] = fr[yy:yy + h, x:x + w] * (1 - al) + a2[..., :3] * al
                return


def dur_of(clip):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 os.path.join(SRC, "drive", clip)], capture_output=True, text=True).stdout)


def story(num, slug, clip, col, sub_y, ct=0.0, more=(), animate=False, fade=True):
    d = dur_of(clip)
    c = Clip(clip, 0.0, d, ct=ct, zoom=(1.0, 1.03), zc=(0.5, 0.3))
    shade = black_fade(col.items[0].y + 40, 380, 0.82) if fade else None
    render(num, slug, d, clip_bg(c), col.items + list(more), shade=shade, subs=PillSubs(clip, y=sub_y),
           audio=(clip, 0.0, d), animate=animate)


# ---------------- stories ----------------
def s1():
    col = Column(0, align="left")
    col.head("ПОЧЕМУ БЕЗ ПОДТЯЖКИ?", 0, size=64, gap=14)
    col.body("история пациентки", 0, size=36, wght=400, gap=0)
    col.to_bottom(1720)
    story(1, "pochemu_bez_podtyazhki", "d1.mp4", col, sub_y=1240)


def s2():
    col = Column(0, align="left")
    col.head("ПОДТЯЖКИ ТУТ НЕ БУДЕТ", 0, size=60, gap=10)
    col.body("таково желание пациентки", 0, size=36, wght=500, gap=26)
    col.body("работать будем только за счёт жировой ткани  >>>", 0, size=36, wght=400, gap=0)
    col.to_bottom(1720)
    story(2, "podtyazhki_ne_budet", "d2.mp4", col, sub_y=1180)


def s3():
    # donor sites appear one by one (light animation), the knees thesis below
    col = Column(0, align="left")
    col.head("ОТКУДА ЗАБИРАЕМ ЖИР:", 0, size=56, gap=16)
    for i, t in enumerate(["внутренняя поверхность бёдер", "колени", "галифе", "фланги"]):
        col.body(t, 0.4 + i * 0.35, size=38, wght=500, gap=4)
    col.space(18)
    col.body("Колени: жир забираем не спереди, иначе кожа станет морщинистой, а по контуру, чтобы линия была ровнее",
             1.9, size=30, wght=400, gap=0)
    col.to_bottom(1740)
    for it in col.items[:1]: it.t0, it.dur = -1.0, 0.01
    story(3, "otkuda_zhir", "d3.mp4", col, sub_y=1020, ct=0.10, animate=True)


def s4():
    col = Column(0, align="left")
    col.head("СЕГОДНЯ ЛИПОФИЛИНГ ГРУДИ", 0, size=56, gap=12)
    col.body("жир забираем из жировых ловушек и переносим в грудь", 0, size=34, wght=400, gap=0)
    col.to_bottom(1720)
    story(4, "segodnya_lipofiling", "d4.mp4", col, sub_y=1250)


def s5():
    col = Column(0, align="left")
    col.head("ОПЕРАЦИЯ ЗАВЕРШЕНА", 0, size=60, gap=12)
    col.body("грудь исправили только с помощью липофилинга, без подтяжки и рубцов", 0, size=34, wght=400, gap=0)
    col.to_bottom(1720)
    story(5, "operaciya_zavershena", "d5.mp4", col, sub_y=1250)


def s6():
    # first look after surgery + «наше до» in a circle (top right), CTA at the bottom
    d = dur_of("v5.mp4")
    c = Clip("v5.mp4", 0.0, d, zoom=(1.0, 1.04), zc=(0.5, 0.6))
    D = 380; ph = load_img("IMG_1708.JPG").crop((180, 220, 1360, 1400)).resize((D, D), Image.LANCZOS)
    circ = Image.new("RGBA", (D + 12, D + 12), (0, 0, 0, 0)); dd = ImageDraw.Draw(circ)
    dd.ellipse((0, 0, D + 11, D + 11), fill=WHITE + (255,))
    m = Image.new("L", (D, D), 0); ImageDraw.Draw(m).ellipse((0, 0, D - 1, D - 1), fill=255)
    ph = ph.convert("RGBA"); ph.putalpha(m); circ.alpha_composite(ph, (6, 6))
    cx, cy = W - 60 - circ.width, 150
    lab = small_label("наше до", 30)
    items = [Sprite(circ, cx, cy, 0, dy=0), Sprite(lab, cx + (circ.width - lab.width) // 2, cy + circ.height + 2, 0, dy=0)]
    col = Column(0, align="left")
    col.head("КОРРЕКЦИЯ ГРУДИ БЕЗ ИМПЛАНТОВ", 0, size=58, gap=12)
    col.body("встретимся на перевязке", 0, size=36, wght=400, gap=24)
    col.body("Напишите, как вам первый результат?", 0, size=38, wght=600, gap=0)
    col.to_bottom(1720)
    render(6, "bez_implantov", d, clip_bg(c), items + col.items, shade=black_fade(col.items[0].y + 40, 380, 0.82),
           audio=("v5.mp4", 0.0, d))


STORIES = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6}
for k, fn in STORIES.items():
    if not ONLY or k in ONLY: fn()
