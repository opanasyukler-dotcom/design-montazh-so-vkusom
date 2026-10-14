"""15 Instagram stories (1080x1920): face lift storytelling - "will the scars give it away?".

Fonts: Inter Bold caps headlines (the bold face from the reference story), Montserrat Light body,
Montserrat Medium subtitles on the slate plate from the reference.  Brand palette from the surgeon's
posts (graphite / lilac / powder).  Subtitles are hand-timed cues built from the word timings of
the voice clips; lilac/powder arrows mark the zones on the before/after collages.

usage: python3 stories_facelift_scars.py SRC_DIR OUT_DIR [story numbers...]
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
        im = Image.new("RGBA", (length, 4), LILAC + (255,))
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
            fr = fr * (1 - sh) + np.array(GRAPHITE, np.float32) / 255 * sh
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
        if len(buf) == W * H * 3: self.last = retouch(np.frombuffer(buf, np.uint8).reshape(H, W, 3))
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
        _PHOTOS[name] = Image.fromarray(retouch(np.asarray(im)))
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


# ---------------- stories ----------------
def s1():
    ph = photo("IMG_7778.JPG"); ph = ph.resize((W, int(ph.height * W / ph.width)), Image.LANCZOS)   # 1080x1440
    m = np.ones((ph.height, ph.width), np.float32)                                   # feather the photo into the black
    for k in range(90): m[:, k] *= k / 90; m[:, -1 - k] *= k / 90
    for k in range(260): m[-1 - k, :] *= k / 260
    m[:60] *= np.linspace(0, 1, 60)[:, None]
    base = Image.new("RGB", (W, H), (0, 0, 0))
    base.paste(ph, (60, 490), Image.fromarray((m * 255).astype(np.uint8)))           # face sits below the headline
    col = Column(230, align="left")
    col.head("«А ПО РУБЦАМ БУДЕТ ВИДНО, ЧТО Я ДЕЛАЛА ПОДТЯЖКУ ЛИЦА?»", 0, size=56, gap=20).rule(0, gap=22)
    col.body("Это один из самых понятных страхов перед операцией на лице.", 0, size=36, gap=0)
    print("s1 bottom", col.bottom)
    # poll «Вас бы это волновало?» goes low over the shirt (~1450..1680)
    render(1, "strah_rubcov", 10.0, still_bg(base, z1=1.03, zc=(0.6, 0.45)), col.items, shade=np.maximum(top_shade(col.bottom + 40, 0.6), bottom_shade(1330, 1.0)))


S2 = [(1.52, 4.36, "Добрый день, хочу сегодня показать пациентку,"), (4.36, 7.0, "которую мы делали сутки назад."),
      (7.12, 10.24, "Комплексную операцию: мы сделали эндоскопическую подтяжку"), (10.24, 12.5, "плюс SMAS-подтяжку"),
      (12.52, 15.0, "достаточно коротким рубцом."), (15.06, 16.9, "Рубец мы выводим обычно в бакенбард,"),
      (16.9, 20.5, "для того чтобы не были потом у нас лысые виски."), (20.66, 24.25, "Это один из таких признаков того, что делали"),
      (24.3, 25.4, "подтяжку лица,"), (25.46, 27.4, "поэтому немножко другую методику используем.")]


def s2():
    a, b = 1.45, 27.4
    c = RClip("IMG_9032.MP4", a, b, zoom=(1.0, 1.04), zc=(0.5, 0.3))
    col = Column(0, align="right")
    col.head("РУБЕЦ ПЛАНИРУЕТСЯ ЕЩЁ ДО ОПЕРАЦИИ", 0, size=58, gap=20).rule(0, gap=22)
    col.body("Не только его длина, но и место, где он будет расположен.", 0, size=36, gap=0)
    low(col)
    render(2, "rubec_planiruetsya", b - a, clip_bg(c), col.items, shade=bottom_shade(col.top - 260, 0.82), subs=cues(S2, a),
           audio=("IMG_9032.MP4", a, b))


def s3():
    img = frame_crop(photo("IMG_7767.JPG"), 0.36, 0.55, 0.62)
    col = Column(0, align="left")
    col.body("Вот почему в хирургии лица важен не только вопрос:", 0.3, size=38, gap=8)
    col.head("«БУДЕТ ЛИ РУБЕЦ?»", 1.3, size=60, col=POWDER, gap=18)
    col.body("Рубец после операции есть.", 2.6, size=38, wght=500, gap=18).rule(3.4, gap=20)
    col.body("Важнее другое:", 4.0, size=38, gap=6)
    col.body("где он будет проходить и насколько естественно эта зона будет выглядеть после заживления.", 4.6, size=38, wght=450, gap=0)
    low(col, 1650)
    # sutures: along the top of the ear (~600..730, 745) and down the front edge of the ear (~650..665, 750..990)
    arr = [arrow_img((940, 560), (712, 742), LILAC, bend=0.18), arrow_img((960, 1010), (676, 905), LILAC, bend=-0.18)]
    render(3, "gde_prohodit_rubec", 12.0, still_bg(img, zc=(0.62, 0.45)), col.items + arr,
           shade=bottom_shade(col.top - 300, 0.85))


S4 = [(39.62, 42.9, "Для первых суток пациентка очень неплохо выглядит."), (43.28, 46.2, "Все эти отёки, синяки, естественно, сойдут."),
      (46.32, 50.0, "Обычно я говорю… а тут вроде бы нам всё нравится."),
      (50.96, 54.6, "Брыли хорошо мы расправили, сделали липосакцию подбородка."), (57.14, 58.1, "Результат налицо."),
      (58.16, 61.4, "Понятно, что это всё ещё пока не зажившее,"), (61.48, 64.6, "но я могу сказать профессиональным глазом: будет это очень красиво.")]


def s4():
    a, b = 39.5, 64.9
    c = RClip("IMG_9032.MP4", a, b, zoom=(1.0, 1.03), zc=(0.5, 0.3))
    ins = [(frame_crop(photo(n), cx, cy, cw), t0) for n, cx, cy, cw, t0 in
           [("IMG_7765.JPG", 0.5, 0.42, 0.74, 6.0), ("IMG_7766.JPG", 0.5, 0.42, 0.74, 9.2), ("IMG_7767.JPG", 0.45, 0.42, 0.74, 12.4)]]
    ins = [(np.asarray(im).astype(np.float32), t0) for im, t0 in ins]
    def bg(i, t):
        fr = c.frame(i).astype(np.float32)
        for k, (im, t0) in enumerate(ins):        # early day-1 photos while she talks about swelling
            t1 = t0 + 3.2
            if t0 - 0.35 <= t <= t1 + 0.35:
                al = min(1.0, (t - t0 + 0.35) / 0.35, (t1 + 0.35 - t) / 0.35)
                z = 1.0 + 0.04 * (t - t0) / 3.2
                M = cv2.getRotationMatrix2D((W / 2, H * 0.4), 0, z)
                fr = fr * (1 - al) + cv2.warpAffine(im, M, (W, H), borderMode=cv2.BORDER_REPLICATE) * al
        return fr.astype(np.uint8)
    col = Column(0, align="left")
    col.head("ЭТО ТОЛЬКО ПЕРВЫЕ СУТКИ", 0, size=60, gap=20).rule(0, gap=22)
    col.body("То, что вы видите сразу после операции, ещё нельзя воспринимать как окончательный результат.", 0, size=36, gap=0)
    low(col)
    render(4, "pervye_sutki", b - a, bg, col.items, shade=bottom_shade(col.top - 260, 0.82), subs=cues(S4, a),
           audio=("IMG_9032.MP4", a, b))


S5 = [(12.92, 16.5, "— Как вам первый результат?"), (16.66, 19.7, "— Ну, всё так натянуто…"), (19.78, 23.9, "В прямом смысле этого слова."),
      (25.98, 29.4, "Вот я вижу вот этот угол — мне очень нравится."), (29.46, 31.9, "Вот сейчас, когда с меня сняли масочку…")]


def s5():
    a, b = 12.85, 31.9
    c = RClip("IMG_9043.MP4", a, b, zoom=(1.0, 1.03), zc=(0.5, 0.3))
    col = Column(0, align="center")
    col.head("ДАЖЕ В ПЕРВЫЕ СУТКИ ПАЦИЕНТКА УЖЕ ЗАМЕЧАЕТ ИЗМЕНЕНИЯ", 0, size=54, gap=20).rule(0, gap=22)
    col.body("Хотя отёк и следы операции ещё сохраняются.", 0, size=36, gap=0)
    low(col)
    render(5, "pervyj_rezultat", b - a, clip_bg(c), col.items, shade=bottom_shade(col.top - 260, 0.82), subs=cues(S5, a),
           audio=("IMG_9043.MP4", a, b))


S6 = [(31.96, 35.0, "Вот… нравится."), (35.12, 36.3, "Очень красиво."), (36.34, 37.9, "Мне кажется, очень красиво.")]


def s6():
    a, b = 31.9, 37.9
    c = RClip("IMG_9043.MP4", a, b, zoom=(1.0, 1.05), zc=(0.5, 0.3))
    col = Column(0, align="left")
    col.body("первый взгляд на результат", 0, size=34, wght=400, col=POWDER, italic=True, gap=0)
    low(col, 1570)
    render(6, "ochen_krasivo", b - a, clip_bg(c), col.items, shade=bottom_shade(1300, 0.6), subs=cues(S6, a),
           audio=("IMG_9043.MP4", a, b))


def s7():
    c = RClip("IMG_9049.MP4", 0.0, 9.4, zoom=(1.0, 1.05), zc=(0.6, 0.7))
    col = Column(240, align="left")
    col.body("И вот здесь начинается часть, о которой пациентки иногда думают меньше всего:", 0, size=38, gap=16)
    col.head("ЗА РУБЦОМ НУЖНО УХАЖИВАТЬ", 0, size=58, col=LILAC, gap=16).rule(0, gap=20)
    col.body("Он не существует отдельно от всего послеоперационного восстановления.", 0, size=36, gap=0)
    render(7, "uhod_za_rubcom", 9.4, clip_bg(c), col.items, shade=top_shade(col.bottom + 180, 0.85))


S8 = [(0.0, 3.8, "И если сушит, можно с гиалуронкой какие-нибудь назначить капли,"),
      (3.84, 7.9, "которые просто увлажняющие. Увлажняющие."), (7.94, 11.0, "И ушной палочкой протираем.")]


def s8():
    a, b = 0.0, 11.0
    c = RClip("IMG_9031.MP4", a, b, zoom=(1.0, 1.03), zc=(0.4, 0.3))
    col = Column(0, align="right")
    col.head("ПОСЛЕ ОПЕРАЦИИ ПАЦИЕНТКА ПОЛУЧАЕТ РЕКОМЕНДАЦИИ ПО УХОДУ", 0, size=52, gap=20).rule(0, gap=22)
    col.body("Уход зависит от операции и конкретного этапа восстановления.", 0, size=36, gap=0)
    low(col)
    render(8, "rekomendacii", b - a, clip_bg(c), col.items, shade=bottom_shade(col.top - 260, 0.82), subs=cues(S8, a),
           audio=("IMG_9031.MP4", a, b))


def s9():
    img = frame_crop(photo("IMG_7778.JPG"), 0.52, 0.36, 0.62)
    col = Column(0, align="left")
    col.body("Поэтому, когда вы смотрите на рубец после подтяжки лица, важно учитывать сразу несколько вещей:", 0.2, size=36, gap=18)
    for k, t in enumerate(["место расположения разреза", "срок после операции", "особенности заживления", "послеоперационный уход"]):
        col.body("—  " + t, 1.4 + 0.7 * k, size=40, wght=600, col=WHITE, gap=6)
    col.space(14).rule(4.4, gap=18)
    col.body("А не оценивать результат по одному кадру в первые дни.", 4.8, size=36, col=POWDER, italic=True, wght=500, gap=0)
    low(col, 1650)
    render(9, "chto_uchityvat", 12.0, still_bg(img, zc=(0.4, 0.3)), col.items, shade=bottom_shade(col.top - 280, 0.86))


def s10():
    img = frame_crop(photo("IMG_7778.JPG"), 0.40, 0.33, 0.46)
    col = Column(250, align="center")
    col.head("ПОПРОБУЙТЕ НАЙТИ РУБЕЦ", 0, size=66, gap=0)
    # poll «Удалось сразу?» goes low (~1380..1640) over the shirt
    render(10, "najdite_rubec", 10.0, still_bg(img, z1=1.03, zc=(0.4, 0.35)), col.items,
           shade=np.maximum(top_shade(col.bottom + 160, 0.6), bottom_shade(1250, 0.45)))


def s11():
    ph = photo("IMG_7782.PNG").resize((W, 1440), Image.LANCZOS)
    base = Image.new("RGB", (W, H), GRAPHITE); base.paste(ph, (0, 0))
    col = Column(0, align="right")
    col.body("Если вы давно думаете о подтяжке лица, но каждый раз возникает:", 0, size=36, gap=10)
    col.head("«А ВДРУГ РУБЦЫ БУДУТ ЗАМЕТНЫ?»", 0, size=54, col=POWDER, gap=18).rule(0, gap=20)
    col.body("Этот вопрос лучше обсудить ещё до операции.", 0, size=38, wght=600, gap=14)
    col.body("На консультации можно заранее понять, где будут проходить разрезы и какой объём операции нужен именно в вашем случае.", 0, size=34, gap=0)
    low(col, 1660)
    render(11, "obsudit_do_operacii", 10.0, still_bg(base, zc=(0.5, 0.25)), col.items, shade=bottom_shade(col.top - 300, 0.9))


def collage_bf(num, slug, name):
    im = load_img(name).resize((W, H), Image.LANCZOS).convert("RGBA")
    im.alpha_composite(chip("ПОСЛЕ"), (MARGIN - 20, 250))
    im.alpha_composite(chip("ДО", POWDER), (MARGIN - 20, H // 2 + 40))
    render(num, slug, 8.0, still_bg(im, z1=1.02, zc=(0.5, 0.5)), [])


def s12(): collage_bf(12, "do_posle_anfas", "IMG_7783.PNG")


def s13(): collage_bf(13, "do_posle_3-4", "IMG_7784.PNG")


def s14(): collage_bf(14, "do_posle_profil", "IMG_7785.PNG")


def s15():
    tiles = [frame_crop(photo("IMG_7767.JPG"), 0.34, 0.45, 0.45), frame_crop(photo("IMG_7766.JPG"), 0.40, 0.40, 0.5),
             frame_crop(photo("IMG_7778.JPG"), 0.40, 0.33, 0.42), frame_crop(photo("IMG_7785.PNG"), 0.36, 0.20, 0.55)]
    tw, th = (W - 18) // 2, 560
    base = Image.new("RGB", (W, H), GRAPHITE)
    for k, t in enumerate(tiles):
        t = t.resize((tw, int(tw * H / W)), Image.LANCZOS)
        t = t.crop((0, (t.height - th) // 2, tw, (t.height - th) // 2 + th))
        base.paste(t, ((k % 2) * (tw + 18), 120 + (k // 2) * (th + 18)))
    col = Column(0, align="left")
    col.head("БОЛЬШЕ РЕАЛЬНЫХ ПРИМЕРОВ СОБРАЛА В АКТУАЛЬНОМ «РУБЦЫ»", 0, size=50, gap=18).rule(0, gap=22)
    col.body("Там показываю рубцы после разных операций и на разных этапах восстановления.", 0, size=34, gap=26)
    col.body("Для записи напишите в Direct «КОНСУЛЬТАЦИЯ» или свяжитесь с нами по телефону", 0, size=32, gap=6)
    col.head("+7 926 636 30 00", 0, size=52, col=LILAC, gap=4)
    col.body("WhatsApp · Telegram · MAX", 0, size=32, wght=500, gap=0)
    low(col, 1690)
    print("s15 top", col.top)
    render(15, "aktualnoe_rubcy", 10.0, still_bg(base, z1=1.02, zc=(0.5, 0.3)), col.items,
           shade=bottom_shade(col.top - 260, 0.92))


STORIES = {k + 1: f for k, f in enumerate([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15])}
for k, fn in STORIES.items():
    if not ONLY or k in ONLY: fn()
