"""12 Instagram stories (1080x1920): "fear of implants -> lift+lipofilling -> implants a year later".

Fonts: Inter Bold caps headlines (the bold face from the reference story), Montserrat Light body,
Montserrat Medium subtitles on the slate plate from the reference.  Brand palette from the surgeon's
posts (graphite / lilac / powder).  Patient faces: interview clips come pre-blurred; before/after clips
are cropped below the chin and get a frosted top band; photos are framed from the neck down.

usage: python3 stories_implants_journey.py SRC_DIR OUT_DIR [story numbers...]
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
        x = pad + (width - lw) / 2 if align == "center" else pad
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
        x = MARGIN - pad if self.align == "left" else (W - im.width) // 2
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
        x = MARGIN if self.align == "left" else (W - length) // 2
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


def sub_sprite(text, y=1590):
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
def render(num, slug, dur, bg, sprites, shade=None, subs=None, audio=None):
    """bg(i, t) -> uint8 frame; shade: per-row darkening (H,1,1) or fn(t); audio: (file, a, b)"""
    out = os.path.join(OUT, f"{num:02d}_{slug}.mp4"); tmp = out + ".v.mp4"
    n = int(round(dur * FPS))
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
                           stdin=subprocess.PIPE)
    still_t = max(s.t0 + s.dur for s in sprites) + 0.3 if sprites else dur / 2
    still_i = min(n - 1, int(still_t * FPS))
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


INTERVIEW = "IMG_3119.MOV"
FIX = {"чем то": "чем-то", "двоечка был": "двоечка была", "300-325": "300–325"}


# ---------------- stories ----------------
def s1():
    c = Clip("IMG_3059.MOV", 44.0, 56.0, ct=0.20, zoom=(1.0, 1.04), zc=(0.5, 0.6), frost=200, sat=0.95)
    # poll sticker goes on the frosted top band (~260..560); text sits low over the hips
    col = Column(1090)
    col.head("«А ЧТО ЕСЛИ Я ХОЧУ КРАСИВУЮ ГРУДЬ, НО БОЮСЬ ИМПЛАНТОВ?»", 0.4, size=62).rule(1.0)
    col.body("Вот с этой мыслью год назад пришла наша пациентка.", 1.6)
    col.body("Она хотела красивую форму, но к имплантам тогда была не готова.", 3.0)
    print("s1 column bottom", col.bottom)
    render(1, "boyus_implantov", 12.0, clip_bg(c), col.items, shade=np.maximum(bottom_shade(760, 0.8), top_shade(560, 0.35)))


def top_shade(bottom, a=0.72):
    ys = np.arange(H, dtype=np.float32)
    p = np.clip((bottom - ys) / 380, 0, 1); p = p * p * (3 - 2 * p)
    return (a * p)[:, None, None]


def bottom_shade(top, a=0.75):
    ys = np.arange(H, dtype=np.float32)
    p = np.clip((ys - top) / 380, 0, 1); p = p * p * (3 - 2 * p)
    return (a * p)[:, None, None]


def interview(num, slug, a, b, build, extra_shade=None):
    c = Clip(INTERVIEW, a, b, zoom=(1.0, 1.05), zc=(0.5, 0.35))
    col = build()
    sh = np.maximum(top_shade(col.bottom + 120), bottom_shade(1380, 0.55))
    if extra_shade is not None: sh = np.maximum(sh, extra_shade)
    render(num, slug, b - a, clip_bg(c), col.items, shade=sh,
           subs=Subs(chunks(load_words(INTERVIEW, a, b)), FIX), audio=(INTERVIEW, a, b))


def s2():
    def b():
        col = Column(240)
        col.head("УЗНАЁТЕ СЕБЯ?", 0.3, size=72).rule(0.8)
        col.body("Хочется изменить грудь.", 1.4)
        col.body("Но внутри есть мысль:", 2.8)
        col.body("«А вдруг имплант будет ощущаться чем-то чужим?»", 4.0, wght=500, col=POWDER, italic=True)
        col.body("И поэтому хочется сначала выбрать более мягкий вариант.", 6.2)
        return col
    interview(2, "uznaete_sebya", 0.0, 11.25, b)


def s3():
    def b():
        col = Column(240)
        col.body("Тогда выбрали:", 0.3)
        col.head("ПОДТЯЖКУ + ЛИПОФИЛИНГ", 0.9, size=68).rule(1.3)
        col.body("То есть поработали с формой груди и добавили объём собственной жировой тканью.", 2.2)
        col.body("И на тот момент это было именно то решение, к которому пациентка была готова.", 4.4)
        return col
    interview(3, "podtyazhka_lipofiling", 11.2, 18.2, b)


def s4():
    a, b = 0.0, 18.6
    c = Clip("IMG_3116.MOV", a, b, zoom=(1.0, 1.04), zc=(0.5, 0.4))
    col = Column(240)
    col.head("А ЧЕРЕЗ ГОД ЗАПРОС ИЗМЕНИЛСЯ", 0.4, size=66).rule(1.0)
    col.body("Форма уже нравится.", 1.8)
    col.body("Но теперь пациентке хочется больше объёма и более пышную грудь.", 3.2)
    sh = np.maximum(top_shade(col.bottom + 120), bottom_shade(1380, 0.5))
    render(4, "zapros_izmenilsya", b - a, clip_bg(c), col.items, shade=sh,
           subs=Subs(chunks(load_words("IMG_3116.MOV", a, b)), FIX), audio=("IMG_3116.MOV", a, b))


def s5():
    def b():
        col = Column(240)
        col.head("И ВОТ ЗДЕСЬ ОЧЕНЬ ВАЖНАЯ МЫСЛЬ", 0.4, size=64).rule(1.0)
        col.body("Иногда женщина боится имплантов не потому, что они ей точно не подходят.", 2.0)
        col.body("А потому что она пока к ним не готова.", 4.6, wght=500)
        col.body("И это тоже нормально.", 6.6, wght=500, col=POWDER, italic=True)
        return col
    interview(5, "vazhnaya_mysl", 18.15, 35.4, b)


def s6():
    def b():
        col = Column(250)
        col.head("«ХОЧЕТСЯ РАСПРАВИТЬ ПЛЕЧИ И ЛЮБОВАТЬСЯ СОБОЙ»", 0.4, size=74, gap=30).rule(1.0)
        col.body("И мне кажется, вот здесь многие женщины поймут её особенно хорошо.", 2.4)
        # question sticker goes ~950..1350
        return col
    interview(6, "rasparvit_plechi", 35.35, 49.1, b)


def s7():
    a, b = 0.6, 27.88
    c = Clip("IMG_3144.MOV", a, b, zoom=(1.0, 1.03), zc=(0.5, 0.4))
    col = Column(240)
    col.head("ПЕРЕД ОПЕРАЦИЕЙ ЕЩЁ РАЗ ОБСУЖДАЕМ ОБЪЁМ", 0.4, size=62).rule(1.0)
    col.body("Не по принципу:", 2.0)
    col.body("«Мне нравится 325 мл, значит ставим 325».", 3.0, wght=500, col=POWDER, italic=True)
    col.body("А смотрим на пропорции пациентки и на тот результат, который она хочет получить.", 5.0)
    sh = np.maximum(top_shade(col.bottom + 120), bottom_shade(1380, 0.5))
    render(7, "vybor_obema", b - a, clip_bg(c), col.items, shade=sh,
           subs=Subs(chunks(load_words("IMG_3144.MOV", a, b)), FIX), audio=("IMG_3144.MOV", a, b))


def s8():
    c = Clip("IMG_1360.MP4", 8.0, 20.0, ct=0.21, zoom=(1.0, 1.04), zc=(0.5, 0.3), frost=120)
    col = Column(1330)
    col.head("ПЕРВЫЙ РЕЗУЛЬТАТ", 1.0, size=70, with_heart=True).rule(1.5)
    col.body("Именно тот более наполненный объём, которого пациентке не хватало после первого этапа.", 2.2)
    sh = lambda t: bottom_shade(1080, 0.78) * ease((t - 0.8) / 0.6)
    render(8, "pervyj_rezultat", 12.0, clip_bg(c), col.items, shade=sh)


def s9():
    a, b = 1.1, 16.25
    c = Clip("IMG_3138.MOV", a, b, zoom=(1.0, 1.04), zc=(0.6, 0.4))
    col = Column(250)
    col.head("«КАК БУДТО ТАК И БЫЛО ВСЕГДА»", 4.8, size=70, with_heart=True).rule(5.3)
    col.body("Это лучший момент всей истории.", 6.4, wght=400)
    sh = lambda t: top_shade(col.bottom + 120, 0.7) * ease((t - 4.4) / 0.8)
    render(9, "kak_budto_vsegda", b - a, clip_bg(c), col.items, shade=sh,
           subs=Subs(chunks(load_words("IMG_3138.MOV", a, b)), FIX), audio=("IMG_3138.MOV", a, b))


def collage(after, before):
    """top = ПОСЛЕ, bottom = ДО; each half 1080x957 with a 6px graphite seam"""
    hh = (H - 6) // 2
    cv_ = Image.new("RGB", (W, H), GRAPHITE)
    cv_.paste(after.resize((W, hh), Image.LANCZOS), (0, 0))
    cv_.paste(before.resize((W, H - hh - 6), Image.LANCZOS), (0, hh + 6))
    im = cv_.convert("RGBA")
    im.alpha_composite(chip("ПОСЛЕ"), (48, 250))
    im.alpha_composite(chip("ДО", POWDER), (48, hh + 6 + 40))
    return np.asarray(im.convert("RGB"))


def photos():
    asp = W / ((H - 6) // 2)
    A_front = crop_frac(load_img("IMG_7678.JPG"), 0.50, 0.265, 0.70, asp)    # shoulders..navel, no face
    B_front = crop_frac(load_img("IMG_7690.HEIC"), 0.50, 0.35, 1.00, asp)
    A_side = crop_frac(load_img("IMG_7679.JPG"), 0.53, 0.285, 0.68, asp)
    B_side = crop_frac(load_img("IMG_7691.HEIC"), 0.50, 0.35, 1.00, asp)
    return collage(A_front, B_front), collage(A_side, B_side)


def s10():
    C1, C2 = photos()
    col = Column(330)
    T = 3.2
    col.head("И ВОТ ЗДЕСЬ ГЛАВНЫЙ ВЫВОД", T, size=64).rule(T + 0.5)
    col.body("У одной и той же пациентки в разное время может быть разный запрос.", T + 1.3)
    col.body("Сначала:", T + 3.0, gap=4)
    col.body("«Хочу без имплантов»", T + 3.5, wght=600, col=POWDER, italic=True, size=50)
    col.body("Потом:", T + 4.7, gap=4)
    col.body("«Теперь хочу больше объёма»", T + 5.2, wght=600, col=LILAC, italic=True, size=50)
    col.body("И оба решения могут быть правильными именно в тот момент, когда женщина к ним готова.", T + 6.8, wght=400)
    B1 = cv2.GaussianBlur(C1, (0, 0), 14); B2 = cv2.GaussianBlur(C2, (0, 0), 14)

    def bg(i, t):
        k2 = ease((t - 1.6) / 0.8)                                   # front -> 3/4 before text
        clean = C1 * (1 - k2) + C2 * k2
        blur = B1 * (1 - k2) + B2 * k2
        kb = ease((t - T + 0.4) / 0.7)
        z = 1.0 + 0.03 * t / 16
        fr = (clean * (1 - kb) + blur * kb).astype(np.uint8)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
        return cv2.warpAffine(fr, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    sh = lambda t: np.full((H, 1, 1), 0.62, np.float32) * ease((t - T + 0.4) / 0.7)
    render(10, "do_posle_vyvod", 16.0, bg, col.items, shade=sh)


def icon(name):
    return Image.open(os.path.join(SRC, "drive", name)).convert("RGBA")


def s11():
    c = Clip("IMG_1365.MP4", 0.0, 13.0, ct=0.10, zoom=(1.0, 1.04), zc=(0.5, 0.3), frost=70)
    col = Column(980)
    col.head("А ВЫ СЕЙЧАС ГДЕ?", 0.5, size=74).rule(1.0)
    col.space(300)  # poll sticker
    col.icon_row(icon("IMG_7693.PNG"), "Если сомневаетесь, загляните в актуальное «Тест». "
                 "Там можно предварительно понять, какой вариант вам ближе.", 2.0, size=36)
    render(11, "vy_seichas_gde", 13.0, clip_bg(c), col.items, shade=bottom_shade(700, 0.8))


def s12():
    ph = load_img("7E7475C1-682A-40D7-BACC-4487368771D1.png")
    ph = ph.resize((W, int(ph.height * W / ph.width)), Image.LANCZOS)        # 1080x1350
    base = Image.new("RGB", (W, H), GRAPHITE); base.paste(ph, (0, -140))
    base = np.asarray(base).astype(np.float32)
    ys = np.arange(H, dtype=np.float32)
    p = np.clip((ys - 640) / 460, 0, 1); p = p * p * (3 - 2 * p)
    base = base * (1 - p[:, None, None]) + np.array(GRAPHITE, np.float32) * p[:, None, None]
    base = base.astype(np.uint8)
    col = Column(870)
    col.head("ЕСЛИ ХОТИТЕ ПОДРОБНЕЕ РАЗОБРАТЬСЯ", 0.4, size=52, gap=18).rule(0.8, gap=24)
    col.icon_row(icon("IMG_7692.PNG"), "В актуальном «Липофилинг» показываю, кому может подойти увеличение собственной жировой тканью.", 1.2, size=32, ih=150, gap=14)
    col.icon_row(icon("IMG_7694.PNG"), "В актуальном «Импланты» рассказываю про формы, объёмы и подбор имплантов.", 2.0, size=32, ih=150, gap=22)
    col.body("А если хотите понять, что подойдёт именно вам, приходите на консультацию.", 2.8, size=36, wght=500, gap=14)
    col.body("Для записи напишите в Direct «КОНСУЛЬТАЦИЯ» или свяжитесь с нами по телефону", 3.6, size=32, gap=4)
    col.head("+7 926 636 30 00", 3.9, size=50, col=LILAC, gap=2)
    col.body("в WhatsApp, Telegram или MAX", 4.2, size=32)
    print("s12 column bottom", col.bottom)
    z0 = 1.0
    def bg(i, t):
        z = z0 + 0.025 * t / 12
        M = cv2.getRotationMatrix2D((W / 2, 420), 0, z)
        return cv2.warpAffine(base, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    render(12, "konsultaciya", 12.0, bg, col.items)


STORIES = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6, 7: s7, 8: s8, 9: s9, 10: s10, 11: s11, 12: s12}
for k, fn in STORIES.items():
    if not ONLY or k in ONLY: fn()
