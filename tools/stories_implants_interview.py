"""13 Instagram stories (1080x1920): a patient after breast augmentation tells her story (interview, before/after, poll).

Fonts: Inter Bold caps headlines (the bold face from the reference story), Montserrat Light body,
Montserrat Medium subtitles on the slate plate from the reference.  Brand palette from the surgeon's
posts: navy of her logo + white.  Subtitles are hand-timed cues built from the word timings of
the voice clips; lilac/powder arrows mark the zones on the before/after collages.

usage: python3 stories_implants_interview.py SRC_DIR OUT_DIR [story numbers...]
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


# ---------------- materials ----------------
# drive/int*.mp4 - interview IMG_1616 cut per story with the patient's face blurred (tools/face_blur_track.py):
#   int1 15.0-29.8   int2 39.5-60.3   int3 72.0-76.9 + 79.0-85.65 + 86.6-88.0   int4 101.1-109.5
#   int6 112.0-128.5   int7 143.8-150.3 + 155.2-159.5   int11 171.2-179.1          (asr/ has their word timings)
# drive/IMG_6358.MOV before video (4K), IMG_1614.MP4 after video, IMG_6362.MOV the surgeon's markup
# drive/IMG_63xx.heic before photos (front, 3/4, profile), IMG_15xx.JPG after photos (4.5 months)
import pillow_heif
pillow_heif.register_heif_opener()

FIXES = {"попу,": "попа,", "яненькие,": "ямки", "обхаживала,": "ходила,", "видеться": "видеть себя",
         "такую,": "такой,", "какую": "какой", "спор,": "спорт,", "стал,": "стала,", "объемчика": "объёмчик",
         "обсуждала,": "ждала,", "все,": "всё,", "все-таки": "всё-таки", "все": "всё", "еще": "ещё"}


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


# ---------------- stories ----------------
def before_cards():
    """«наше до»: front and profile, same size, side by side"""
    w, h = 490, 653
    front = load_img("IMG_6325.heic").crop((158, 230, 158 + 1700, 230 + 2267)).resize((w, h), Image.LANCZOS)
    prof = load_img("IMG_6340.heic").crop((259, 270, 259 + 1700, 270 + 2267)).resize((w, h), Image.LANCZOS)
    return card(front, w, h), card(prof, w, h)


# front photos aligned on the nipple stickers: before (683,1129)-(1476,1083), after (531,838)-(1079,819)
FRONT = {"before": ("IMG_6325.heic", 0.66, (1080, 1106)), "after": ("IMG_1588.JPG", 0.956, (805, 828))}


def front_frame(key, ym=560):
    n, s, (mx, my) = FRONT[key]
    im = load_img(n); im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    ox, oy = int(540 - mx * s), int(ym - my * s)
    fr = Image.new("RGB", (W, H), (10, 10, 12)); fr.paste(im, (ox, oy)); return np.asarray(fr)


class ChipSwap:
    def __init__(self, t_sw):
        self.t0, self.dur, self.t_sw = -1.0, 0.01, t_sw
        def lab(t):
            im = Image.new("RGBA", (400, 70), (0, 0, 0, 0)); label_lines(im, [t], 4, 50, size=34, wght=500); return im
        self.do = np.asarray(lab("до"), np.float32) / 255; self.po = np.asarray(lab("после"), np.float32) / 255

    def draw(self, fr, t):
        a = self.do if t < self.t_sw else self.po
        h, w = a.shape[:2]; x, y = MARGIN, 1520
        al = a[..., 3:4]; fr[y:y + h, x:x + w] = fr[y:y + h, x:x + w] * (1 - al) + a[..., :3] * al


def cutout(name):
    """photo with the background removed (rembg isnet masks in cut/), on pure black"""
    im = load_img(name); m = Image.open(os.path.join(SRC, "cut", name.split(".")[0] + "_mask.png")).convert("L")
    bg = Image.new("RGB", im.size, (0, 0, 0)); bg.paste(im, (0, 0), m); return bg


# collages: after on top, before below, each cut from the neck (under the chin) to the fingertips, same body height
COLL = {  # name: (before file, before neck y, fingertips y, body x0, x1), (after ...)
    "anfas": (("IMG_6325.heic", 230, 2660, 255, 1760), ("IMG_1588.JPG", 260, 1940, 196, 1343)),
    "tri_chetverti": (("IMG_6331.heic", 260, 2960, 316, 1808), ("IMG_1592.JPG", 560, 2150, 341, 1170)),
    "profil": (("IMG_6340.heic", 270, 2830, 565, 1653), ("IMG_1595.JPG", 230, 1915, 246, 1020)),
}
HALF = 860


def half(spec):
    n, y0, y1, x0, x1 = spec
    im = cutout(n); s = HALF / (y1 - y0)
    cx = (x0 + x1) / 2; ww = W / s
    out = Image.new("RGB", (int(ww), y1 - y0), (0, 0, 0))
    bx = int(cx - ww / 2); src = im.crop((max(bx, 0), y0, min(bx + int(ww), im.width), y1))
    out.paste(src, (max(bx, 0) - bx, 0))
    out = out.resize((W, HALF), Image.LANCZOS)
    a = np.asarray(out, np.float32); k = np.clip(np.arange(HALF, dtype=np.float32) / 30, 0, 1)[:, None, None]
    return Image.fromarray((a * k).astype(np.uint8))               # soft straight cut under the chin


def spaced(text, size, wght=400, track=0.22):
    f = mont(size, wght); ws = [f.getlength(c) for c in text]; tw = sum(ws) + track * size * (len(text) - 1)
    im = Image.new("RGBA", (int(tw) + 8, int(size * 1.6)), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 4
    for c, w in zip(text, ws): d.text((x, size * 0.8), c, font=f, fill=WHITE + (255,), anchor="lm"); x += w + track * size
    return im


BEBAS = os.path.join(ROOT, "skills/montage-svg-typography/Шрифты/Коллекция шрифтов/BebasNeueCyrillic.ttf")


def bebas_txt(text, size, track=0.04):
    f = ImageFont.truetype(BEBAS, size); ws = [f.getlength(c) for c in text]; tw = sum(ws) + track * size * (len(text) - 1)
    im = Image.new("RGBA", (int(tw) + 8, int(size * 1.1)), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 4
    for c, w in zip(text, ws): d.text((x, size * 0.55), c, font=f, fill=WHITE + (255,), anchor="lm"); x += w + track * size
    return im


def collage(num, key, caption):
    """black, background-free bodies: after on top, before below; thin spaced captions like a premium collage"""
    bspec, aspec = COLL[key]
    bg = Image.new("RGB", (W, H), (0, 0, 0))
    top = 60; seam = top + HALF
    bg.paste(half(aspec), (0, top)); bg.paste(half(bspec), (0, seam + 90))
    bg = bg.convert("RGBA")
    po, do = bebas_txt("ПОСЛЕ", 50), bebas_txt("ДО", 50)
    bg.alpha_composite(po, (MARGIN - 4, seam - po.height - 10)); bg.alpha_composite(do, (MARGIN - 4, seam + 90 + HALF - do.height - 10))
    d = ImageDraw.Draw(bg); cy = seam + 45
    if caption:
        cap = bebas_txt(caption, 54); bg.alpha_composite(cap, ((W - cap.width) // 2, cy - cap.height // 2))
        d.line((MARGIN, cy, (W - cap.width) // 2 - 24, cy), fill=WHITE + (160,), width=2)
        d.line(((W + cap.width) // 2 + 24, cy, W - MARGIN, cy), fill=WHITE + (160,), width=2)
    f = mont(22, 400); ImageDraw.Draw(bg).text((W / 2, H - 22), "@dr.tvaurianna", font=f, fill=(150, 150, 150, 255), anchor="mm")
    render(num, f"kollazh_{key}", 10.0, still_bg(bg.convert("RGB"), z1=1.0), [chrome(num)])



# ================= series style (references: light panels, pills, series navigation, dashed connectors, cards) =================
LIGHT = (243, 242, 240)          # neutral light panel
INK = NAVY                        # text on light panels: the surgeon's navy
GREY = (110, 118, 136)
HANDLE = "@dr.tvaurianna"
TAG = "увеличение груди"
TOTAL = 13


def wrap_px(text, f, width):
    out, line = [], ""
    for w in text.split(" "):
        t = (line + " " + w).strip()
        if f.getlength(t) <= width or not line: line = t
        else: out.append(line); line = w
    return out + [line]


def chrome(num, dark=True):
    """series navigation: progress dots top left, topic tag pill top right"""
    im = Image.new("RGBA", (W, 90), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = WHITE if dark else INK
    for i in range(TOTAL):
        x = MARGIN + i * 22
        if i + 1 == num: d.ellipse((x - 7, 38, x + 7, 52), fill=c + (255,))
        else: d.ellipse((x - 5, 40, x + 5, 50), outline=c + (200,), width=2)
    f = mont(24, 500); tw = f.getlength(TAG); x1 = W - MARGIN; x0 = x1 - tw - 36
    d.rounded_rectangle((x0, 26, x1, 64), 19, outline=c + (220,), width=2)
    d.text(((x0 + x1) / 2, 45), TAG, font=f, fill=c + (255,), anchor="mm")
    return Sprite(im, 0, 140, 0, dy=0)


class Panel:
    """light card from y0 to the bottom, rounded top corners; text is laid out inside in navy"""
    def __init__(self, y0, r=56, color=LIGHT, alpha=255, handle=True):
        self.y0 = y0; self.im = Image.new("RGBA", (W, H - y0), (0, 0, 0, 0)); self.d = ImageDraw.Draw(self.im)
        self.d.rounded_rectangle((0, 0, W - 1, H - y0 + r), r, fill=color + (alpha,))
        self.y = 56; self.x = MARGIN
        if handle:
            f = mont(22, 400); self.d.text((W / 2, H - y0 - 46), HANDLE, font=f, fill=GREY + (255,), anchor="mm")

    def head(self, text, size=50, gap=16, width=W - 2 * MARGIN, color=INK, align="left"):
        f = bold(size); lines = wrap_px(text, f, width)
        for ln in lines:
            x = self.x if align == "left" else (W - f.getlength(ln)) / 2
            self.d.text((x, self.y), ln, font=f, fill=color + (255,)); self.y += int(size * 1.12)
        self.y += gap; return self

    def body(self, text, size=32, wght=400, gap=14, width=W - 2 * MARGIN, color=GREY, align="left"):
        f = mont(size, wght); lines = wrap_px(text, f, width)
        for ln in lines:
            x = self.x if align == "left" else (W - f.getlength(ln)) / 2
            self.d.text((x, self.y), ln, font=f, fill=color + (255,)); self.y += int(size * 1.35)
        self.y += gap; return self

    def rule(self, length=110, gap=22):
        self.d.line((self.x, self.y, self.x + length, self.y), fill=INK + (255,), width=3); self.y += gap; return self

    def sprite(self): return Sprite(self.im, 0, self.y0, 0, dy=0)


def pill_img(text, size=30, fill=NAVY, color=WHITE, wght=500, pad=34, outline=None, icon=None):
    f = mont(size, wght); tw = f.getlength(text); h = int(size * 2.0); iw = (h - 18 + 12) if icon else 0
    im = Image.new("RGBA", (int(tw + 2 * pad + iw), h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    if fill: d.rounded_rectangle((0, 0, im.width - 1, h - 1), h // 2, fill=fill + (255,))
    if outline: d.rounded_rectangle((1, 1, im.width - 2, h - 2), h // 2, outline=outline + (255,), width=2)
    if icon: icon(d, pad, 9, h - 18, color)
    d.text((pad + iw + tw / 2, h / 2 + 1), text, font=f, fill=color + (255,), anchor="mm"); return im


def sound_icon(d, x, y, s, c):
    """thin speaker with two waves"""
    d.polygon([(x, y + s * .35), (x + s * .25, y + s * .35), (x + s * .5, y + s * .12), (x + s * .5, y + s * .88),
               (x + s * .25, y + s * .65), (x, y + s * .65)], fill=c + (255,))
    for r in (0.22, 0.4):
        d.arc((x + s * .5 - s * r, y + s * .5 - s * r * 1.6, x + s * .5 + s * r * 1.4, y + s * .5 + s * r * 1.6), -55, 55, fill=c + (255,), width=3)


def swipe_cue():
    """thin outline pill with an arrow (keep watching)"""
    im = Image.new("RGBA", (150, 44), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((1, 1, 148, 42), 21, outline=INK + (255,), width=2)
    d.line((34, 22, 112, 22), fill=INK + (255,), width=2); d.line((104, 15, 113, 22), fill=INK + (255,), width=2)
    d.line((104, 29, 113, 22), fill=INK + (255,), width=2); return im


def dashed_curve(im, p0, p1, bend=0.25, col=WHITE, width=3, dash=12, head=12):
    """thin dashed curved connector with a small arrow head (dark cinematic stories)"""
    (x0, y0), (x1, y1) = p0, p1; mx, my = (x0 + x1) / 2, (y0 + y1) / 2; dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - dy * bend, my + dx * bend
    pts = [((1 - u) ** 2 * x0 + 2 * (1 - u) * u * cx + u * u * x1, (1 - u) ** 2 * y0 + 2 * (1 - u) * u * cy + u * u * y1)
           for u in np.linspace(0, 1, 200)]
    d = ImageDraw.Draw(im); acc, on = 0.0, True
    for a, b in zip(pts, pts[1:]):
        seg = np.hypot(b[0] - a[0], b[1] - a[1]); acc += seg
        if acc > dash: acc = 0; on = not on
        if on: d.line((a, b), fill=col + (230,), width=width)
    ang = np.arctan2(y1 - pts[-8][1], x1 - pts[-8][0])
    for s_ in (-0.5, 0.5):
        d.line(((x1, y1), (x1 - head * np.cos(ang + s_), y1 - head * np.sin(ang + s_))), fill=col + (255,), width=width)


def white_card(text, size=36, width=640, wght=600, color=INK, font=None, align="center", pad=(40, 30)):
    f = font or mont(size, wght); lines = wrap_px(text, f, width - 2 * pad[0])
    lh = int(f.size * 1.3); h = lh * len(lines) + 2 * pad[1]
    im = Image.new("RGBA", (width, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, width - 1, h - 1), 26, fill=WHITE + (245,))
    for i, ln in enumerate(lines):
        x = width / 2 if align == "center" else pad[0]
        d.text((x, pad[1] + i * lh + lh / 2), ln, font=f, fill=color + (255,), anchor="mm" if align == "center" else "lm")
    return im


def iv(num, slug, clip, items, sub_y, shade=None, dark_chrome=True):
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              os.path.join(SRC, "drive", clip)], capture_output=True, text=True).stdout)
    c = Clip(clip, 0.0, d, zoom=(1.0, 1.03), zc=(0.5, 0.3))
    render(num, slug, d, clip_bg(c), [chrome(num, dark_chrome)] + items, shade=shade,
           subs=WordSubs(clip, y=sub_y), audio=(clip, 0.0, d))


# ---------------- stories ----------------
def s1():
    # video on top, light panel below; «наше до» circle sits on the panel edge; poll goes in the panel (left/bottom)
    py = 1240; pn = Panel(py)
    pn.y = 60; pn.head("ТЕЛО ПОСЛЕ", 58, gap=0, width=430).head("РОДОВ,", 58, gap=0, width=430).head("КАКОЕ ОНО?", 58, gap=0, width=430)
    D = 470; cx0 = W - 50 - D - 12; cy0 = py - 250
    circ = VideoCircle("IMG_6358.MOV", 0.0, 14.8, (2160, 2160, 0, 560), D, cx0, cy0)
    f = mont(26, 500); pn.d.text((cx0 + (D + 12) / 2, cy0 + D + 12 - py + 34), "наше до", font=f, fill=GREY + (255,), anchor="mm")
    iv(1, "telo_posle_rodov", "int1.mp4", [pn.sprite(), circ], sub_y=880)
    # poll «У кого такие же изменения, как и у моей пациентки?» — «Моя история.. / Частично про меня»: panel, y ~1500..1820


def s2():
    py = 1010; pn = Panel(py)
    pn.y = 44; pn.head("КОГДА ВЫ ДОЛГО РЕШАЕТЕСЬ НА ОПЕРАЦИЮ, ЭТО НОРМАЛЬНО", 42, gap=8)
    pn.body("Должен внутри случиться импульс «я готова»", 30, 500, gap=0, color=INK)
    w, h = 440, 587; y = py + pn.y + 46; fx = MARGIN - 4
    front = load_img("IMG_6325.heic").crop((158, 230, 1858, 2497)).resize((w, h), Image.LANCZOS)
    prof = load_img("IMG_6340.heic").crop((259, 270, 1959, 2537)).resize((w, h), Image.LANCZOS)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    lay.alpha_composite(card(front, w, h, border=0), (fx, y)); lay.alpha_composite(card(prof, w, h, border=0), (W - fx - w, y))
    sc = w / 1700
    def P(sx, sy): return (fx + (sx - 158) * sc, y + (sy - 230) * sc)
    for text, (cx, cy), tgt in [("пустой верхний полюс", (fx + 230, y + 104), P(1380, 870)),
                                ("опущение груди", (fx + 120, y + 345), P(690, 1262)),
                                ("потеря объёма", (fx + 335, y + 312), P(1545, 1090))]:
        arrow_from(lay, pill(lay, text, cx, cy, size=20), tgt, width=3, head=10)
    f = mont(24, 500); ImageDraw.Draw(lay).text((W / 2, y - 22), "наше до", font=f, fill=GREY + (255,), anchor="mm")
    iv(2, "dolgo_reshaetes", "int2.mp4", [pn.sprite(), Sprite(lay, 0, 0, 0, dy=0)], sub_y=870)


def s3():
    # dark cinematic: white caps header, dashed connector, the key line on a white card
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    f = bold(56); y = 1270
    for ln in ["И КОГДА СТРАШНО,", "ЭТО ТОЖЕ НОРМАЛЬНО"]:
        d.text((MARGIN, y), ln, font=f, fill=WHITE + (255,)); y += 64
    cardi = white_card("Но я рядом с вами", 38, 520)
    lay.alpha_composite(cardi, (W - MARGIN - cardi.width, 1560))
    dashed_curve(lay, (MARGIN + 60, y + 24), (W - MARGIN - cardi.width - 20, 1560 + cardi.height // 2), bend=-0.3)
    iv(3, "strashno", "int3.mp4", [Sprite(lay, 0, 0, 0, dy=0)], sub_y=1000, shade=black_fade(1250, 380, 0.82))


def s4():
    py = 1360; pn = Panel(py)
    pn.y = 52; pn.head("МОИ ПАЦИЕНТКИ ДЕЛЯТСЯ С ВАМИ СВОЕЙ ИСТОРИЕЙ, ЧТОБЫ ПОМОЧЬ ВАМ ПРИНЯТЬ РЕШЕНИЕ", 40, gap=26)
    pl = pill_img("обязательно включайте звук", 28, icon=sound_icon); pn.im.alpha_composite(pl, (MARGIN, pn.y))
    iv(4, "delyatsya_istoriey", "int4.mp4", [pn.sprite()], sub_y=1010)


def s5():
    b, a = front_frame("before").astype(np.float32), front_frame("after").astype(np.float32)
    T1, T2, T3 = 2.0, 3.2, 6.0
    def bg(i, t):
        k = min(max((t - T1) / (T2 - T1), 0), 1); k = k * k * (3 - 2 * k)
        fr = b * (1 - k) + a * k
        z = 1.0 + 0.03 * t / T3
        M = cv2.getRotationMatrix2D((W / 2, H * 0.35), 0, z)
        return cv2.warpAffine(fr.astype(np.uint8), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pl = pill_img("увеличение груди имплантами", 32, fill=WHITE, color=INK, wght=600)
    render(5, "do_posle_perehod", T3, bg, [chrome(5), Sprite(pl, (W - pl.width) // 2, 1660, 0, dy=0), ChipSwap(2.6)])


def s6():
    s = 0.75; ox, oy = int(540 - 770 * s), int(30 - 230 * s)
    im = cutout("IMG_1588.JPG"); im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    bg = Image.new("RGB", (W, H), (0, 0, 0)); bg.paste(im, (ox, oy))
    a = np.asarray(bg, np.float32); k = np.clip((np.arange(H, dtype=np.float32) - 30) / 40, 0, 1)[:, None, None]
    bg = Image.fromarray((a * k).astype(np.uint8)).convert("RGBA")
    def P(x, y): return (x * s + ox, y * s + oy)
    for text, (cx, cy), tgt in [("наполненный верхний полюс", (380, 268), P(560, 640)),
                                ("увеличенный объём", (860, 585), P(1150, 860)),
                                ("округлая форма", (380, 675), P(590, 985))]:
        arrow_from(bg, pill(bg, text, cx, cy, size=24), tgt, width=3, head=11)
    pn = Panel(1500); pn.y = 60
    pn.head("РЕЗУЛЬТАТ: 4,5 МЕСЯЦА ПОСЛЕ ОПЕРАЦИИ", 44, gap=8).body("увеличение груди имплантами", 30, 500, color=INK)
    render(6, "rezultat_anfas", 10.0, still_bg(bg.convert("RGB"), z1=1.0), [chrome(6), pn.sprite()])


def s7():
    py = 1430; pn = Panel(py)
    pn.y = 56; pn.head("ВЕСЬ ЭТОТ ПУТЬ ПРИВЁЛ К КРАСИВОМУ РЕЗУЛЬТАТУ", 44, gap=10).body("выполнено увеличение груди имплантами", 30, 500, color=INK, gap=20)
    pn.im.alpha_composite(swipe_cue(), (W - MARGIN - 150, pn.y))
    iv(7, "krasivyj_rezultat", "int6.mp4", [pn.sprite()], sub_y=1040)


def s8():
    # dark: header, two calm lines, dashed connector to the conclusion on a white card
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    f = bold(54); y = 1150
    for ln in wrap_px("СПОРТ НЕ ВЕРНЁТ ФОРМУ ГРУДИ ПОСЛЕ РОДОВ", f, W - 2 * MARGIN):
        d.text((MARGIN, y), ln, font=f, fill=WHITE + (255,)); y += 62
    fb = mont(32, 400); y += 18
    for ln in wrap_px("Он подтянет тело. Но не наполнит грудь, потерявшую объём после ГВ.", fb, 620):
        d.text((MARGIN, y), ln, font=fb, fill=WHITE + (235,)); y += 44
    cardi = white_card("Это анатомия.", 38, 420); cy = y + 70
    lay.alpha_composite(cardi, (W - MARGIN - cardi.width, cy))
    dashed_curve(lay, (MARGIN + 40, y + 16), (W - MARGIN - cardi.width - 18, cy + cardi.height // 2), bend=-0.25)
    iv(8, "sport_ne_vernet", "int7.mp4", [Sprite(lay, 0, 0, 0, dy=0)], sub_y=960, shade=black_fade(1130, 380, 0.82))


def collage_s(num, key):
    collage(num, key, "4,5 МЕСЯЦА ПОСЛЕ ОПЕРАЦИИ")


def s9(): collage_s(9, "anfas")
def s10(): collage_s(10, "tri_chetverti")
def s11(): collage_s(11, "profil")


def s12():
    # the key quote on a white card (navy Inter Bold)
    q = white_card("«НАДО ПЕРЕБОРОТЬ СВОЙ СТРАХ И ПОТОМ ТОЧНО НЕ ПОЖАЛЕЕТЕ»", font=bold(44), width=W - 2 * 60, pad=(48, 40))
    f = mont(24, 400); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); y = 1300
    lay.alpha_composite(q, (60, y))
    ImageDraw.Draw(lay).text((W / 2, y + q.height + 36), "— пациентка, 4,5 месяца после операции", font=f, fill=WHITE + (255,), anchor="mm")
    iv(12, "pereborot_strah", "int11.mp4", [Sprite(lay, 0, 0, 0, dy=0)], sub_y=1150, shade=black_fade(1290, 300, 0.6))


def s13():
    # surgeon's markup on top (her face free), light panel with the question; the poll goes inside the panel
    c = Clip("IMG_6362.MOV", 0.0, 12.0, ct=0.05, zoom=(1.0, 1.02), zc=(0.5, 0.4))
    py = 1180; pn = Panel(py)
    pn.y = 54; pn.head("ПОДЕЛИТЕСЬ, ЧТО ВАС СЕЙЧАС ОСТАНАВЛИВАЕТ?", 46, gap=10)
    pn.body("Почему не решаетесь, если тоже столкнулись с изменениями в теле?", 30, 400, gap=0)
    print("s13 poll from", py + pn.y + 20)
    # quiz/poll in the panel: Страх наркоза/боли · Не понимаю что мне подойдёт · Восстановление с детьми · Пока просто присматриваюсь
    render(13, "chto_ostanavlivaet", 12.0, clip_bg(c), [chrome(13), pn.sprite()])


PHOTO_ONLY = {6, 9, 10, 11}        # static stories are delivered as JPG, not video


STORIES = {k + 1: f for k, f in enumerate([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13])}
for k, fn in STORIES.items():
    if not ONLY or k in ONLY:
        fn()
        if k in PHOTO_ONLY:
            for f in os.listdir(OUT):
                if f.startswith(f"{k:02d}_") and f.endswith(".mp4"): os.remove(os.path.join(OUT, f))
