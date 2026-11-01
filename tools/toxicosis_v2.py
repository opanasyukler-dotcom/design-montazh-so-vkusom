"""@mcg.va — «Как справляться с токсикозом?», version 2: kinetic-typography montage in the style of the agency references
(big key words placed in the layout instead of subtitles, thin / heavy / italic contrast in one phrase, typed words with a cursor,
cut-out figures and objects with soft shadows, dotted connector lines, chat bubbles, huge «01.–05.» numbers, cream ↔ ink scenes
with wipe transitions, glitch and shake on the hard words) — her fonts (Montserrat heavy / thin / thin italic) and palette.
Reuses the voice and word timings of toxicosis_reel.py (WORK/voice.wav, WORK/timing.json).
  python3 toxicosis_v2.py VOICE_WORK CUTOUTS_DIR MATERIALS_DIR OUT.mp4 [STICKERS_DIR]
"""
import glob, json, math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import toxicosis_reel as T
from toxicosis_reel import W, H, FPS, SR, CREAM, INK, BROWN, MAUVE, GREY, BLUSH, font, ease, ease_back, paste

C = {"ink": INK, "cream": CREAM, "acc": BROWN, "mute": MAUVE, "blush": BLUSH, "grey": GREY}
ON_INK = {"ink": CREAM, "acc": BLUSH, "mute": GREY}                        # the same roles on a dark scene
TYPE_STEP = 0.034
SAFE = (60, 260, 950, 1480)       # what Instagram Reels leaves visible: header on top, like/comment column on the right, caption below

# ---------- the montage: per scene a background and the elements that come in on the words ----------
# (key, kind, opts): key = a word prefix of the scene (or seconds from its start); kinds T text, R outline repeats,
# IMG picture (icon:/cut:/vid:/photo:), CARD photo card, LINE dotted connector, BUB chat bubble, STAMP, GLITCH, SHAKE, SFX
def num(n, col): return (0.0, "T", dict(text=f"{n:02d}.", st="N", size=210, x=80, y=500, al="l", col=col, anim="pop"))

MONTAGE = [
    dict(bg="ink", els=[
        ("как", "T", dict(text="как справляться с", st="L", size=48, x=540, y=560, col="mute", anim="type")),
        ("токсик", "T", dict(text="ТОКСИКОЗОМ", st="H", size=118, x=540, y=680, col="ink", anim="pop")),
        ("токсик", "R", dict(text="ТОКСИКОЗОМ", size=118, x=540, y=680, n=1, step=128, delay=0.3)),
        ("токсик", "T", dict(text="?", st="H", size=230, x=860, y=930, col="acc", anim="pop", motion="wiggle")),
        (0.0, "BGFULL", dict(src="vid:IMG_0755")),
        ("токсик", "SHAKE", {})]),
    dict(bg="cream", els=[
        ("ой", "T", dict(text="ой,", st="I", size=70, x=90, y=420, al="l", col="mute", anim="type")),
        ("токсик", "T", dict(text="токсикоз —", st="H", size=80, x=90, y=530, al="l", col="ink", anim="up")),
        ("отдельн", "T", dict(text="ОТДЕЛЬНАЯ", st="H", size=84, x=90, y=650, al="l", col="acc", anim="pop")),
        ("истор", "T", dict(text="история", st="L", size=84, x=90, y=780, al="l", col="ink", anim="type")),
        ("беремен", "T", dict(text="(беременности)", st="I", size=56, x=90, y=890, al="l", col="acc", anim="type")),
        (0.0, "IMG", dict(src="vid:IMG_0755", h=860, x=800, y=1040, anim="up")),
        ("беремен", "LINE", dict(p=[(470, 905), (560, 960), (620, 1020)]))]),
    dict(bg="ink", els=[
        ("буквально", "T", dict(text="буквально", st="I", size=58, x=540, y=450, col="acc", anim="type")),
        ("тошнит", "T", dict(text="ТОШНИТ", st="H", size=150, x=540, y=630, col="ink", anim="pop")),
        ("тошнит", "R", dict(text="ТОШНИТ", size=150, x=540, y=630, n=1, step=150, delay=0.15, until="буквально")),
        ("тошнит", "GLITCH", {}), ("тошнит", "SHAKE", {}),
        ("всего", "T", dict(text="от всего", st="L", size=72, x=540, y=820, col="ink", anim="up")),
        ("запах", "CIRCLE", dict(src="photo:0E168C5F-0AAE-4E41-96DE-CF882206C536.png", crop=0.56, d=340, x=270, y=1060, fy=0.5, anim="pop", motion="float")),
        ("запах", "IMG", dict(src="icon:smell", h=170, x=270, y=830, anim="fade", motion="float", shadow=False)),
        ("запах", "CARD", dict(src="IMG_2460.heic", x=720, y=1100, w=340, h=420, rot=-6, anim="slide_r")),
        ("запах", "T", dict(text="даже запахи", st="H", size=66, x=540, y=1370, col="acc", anim="pop")),
        ("замеча", "T", dict(text="(которые раньше не замечала)", st="I", size=40, x=540, y=1450, col="mute", anim="type"))]),
    dict(bg="split", els=[
        ("хочешь", "T", dict(text="хочешь", st="L", size=66, x=270, y=520, col="ink", anim="type", side="l")),
        ("поесть", "T", dict(text="ПОЕСТЬ", st="H", size=96, x=270, y=620, col="acc", anim="pop", side="l")),
        ("поесть", "IMG", dict(src="icon:plate", h=330, x=270, y=900, anim="pop", motion="float")),
        ("но", "T", dict(text="но…", st="I", size=84, x=810, y=520, col="ink", anim="type", side="r")),
        ("мысли", "T", dict(text="от одной мысли", st="L", size=44, x=810, y=630, col="mute", anim="type", side="r")),
        ("плохо", "T", dict(text="ПЛОХО", st="H", size=112, x=810, y=750, col="acc", anim="pop", side="r")),
        ("плохо", "IMG", dict(src="cut:0E168C5F-0AAE-4E41-96DE-CF882206C536", crop=0.56, h=560, x=810, y=1150, anim="up")),
        ("еде", "IMG", dict(src="icon:cross", h=260, x=270, y=900, anim="pop", shadow=False)), ("плохо", "SHAKE", {})]),
    dict(bg="cream", els=[
        ("поделиться", "T", dict(text="хочу поделиться", st="I", size=56, x=540, y=430, col="mute", anim="type")),
        ("попробовать", "T", dict(text="ЧТО МОЖНО\nПОПРОБОВАТЬ", st="H", size=90, x=540, y=600, col="ink", anim="pop")),
        *[("можно", "IMG", dict(src=f"icon:{n}", h=170, x=150 + 195 * i, y=980, anim="pop", motion="float", delay=0.15 * i))
          for i, n in enumerate(("lemon", "mineral", "cola", "fruits", "blister"))],
        *[("можно", "T", dict(text=f"{i + 1:02d}", st="I", size=36, x=150 + 195 * i, y=1110, col="acc", anim="up", delay=0.15 * i)) for i in range(5)],
        ("облегч", "LINE", dict(p=[(540, 1150), (520, 1200), (540, 1250)])),
        ("облегч", "T", dict(text="(чтобы облегчить состояние)", st="I", size=48, x=540, y=1320, col="acc", anim="type")),
]),
    dict(bg="ink", num=1, els=[
        num(1, "acc"), (0.0, "T", dict(text="ЛИМОН", st="H", size=104, x=90, y=690, al="l", col="ink", anim="up")),
        (0.0, "IMG", dict(src="icon:lemon", h=320, x=790, y=560, anim="pop", motion="wiggle")),
        ("понюх", "T", dict(text="понюхать", st="L", size=62, x=90, y=900, al="l", col="ink", anim="type")),
        ("понюх", "IMG", dict(src="icon:smell", h=150, x=650, y=420, anim="fade", motion="float", shadow=False)),
        ("добав", "T", dict(text="или пару долек", st="L", size=62, x=90, y=990, al="l", col="ink", anim="type")),
        ("добав", "IMG", dict(src="icon:water", h=380, x=790, y=1010, anim="pop")),
        ("дол", "IMG", dict(src="icon:slice", h=140, x=790, y=900, anim="drop")), ("воду", "SFX", dict(name="splash")),
        ("кисл", "T", dict(text="КИСЛОЕ", st="H", size=96, x=90, y=1230, al="l", col="acc", anim="pop")),
        ("тошнот", "T", dict(text="(помогает с тошнотой)", st="I", size=44, x=90, y=1330, al="l", col="mute", anim="type")),
        ("кисл", "CIRCLE", dict(src="photo:IMG_0764.JPG", d=230, x=820, y=1340, fx=0.55, fy=0.56, zoom=2.4, anim="pop", motion="float"))]),
    dict(bg="cream", num=2, els=[
        num(2, "acc"), (0.0, "T", dict(text="БОРЖОМИ", st="H", size=110, x=90, y=690, al="l", col="ink", anim="up")),
        (0.0, "IMG", dict(src="icon:mineral", h=480, x=880, y=820, anim="pop", motion="float", until="вода")),
        ("холодн", "T", dict(text="холодная", st="L", size=62, x=90, y=860, al="l", col="ink", anim="type")),
        ("холодн", "IMG", dict(src="icon:snow", h=170, x=960, y=470, anim="pop", motion="spin", shadow=False)),
        ("глотк", "T", dict(text="маленькими\nглотками", st="H", size=72, x=90, y=1010, al="l", col="acc", anim="pop")),
        *[("глотк", "IMG", dict(src="icon:drop", h=s, x=x, y=y, anim="pop", motion="float", delay=0.12 * i, shadow=False))
          for i, (x, y, s) in enumerate(((150, 1210, 90), (240, 1250, 70), (315, 1215, 56)))],
        ("газиров", "BUB", dict(text="а если газировка — хуже?", x=560, y=1330, side="r")), ("газиров", "SFX", dict(name="fizz")),
        ("вода", "IMG", dict(src="icon:water", h=460, x=820, y=800, anim="pop")),
        ("вода", "IMG", dict(src="icon:check", h=180, x=980, y=600, anim="pop", shadow=False)),
        ("вода", "T", dict(text="обычная вода", st="I", size=58, x=540, y=1470, col="acc", anim="type"))]),
    dict(bg="ink", num=3, els=[
        num(3, "acc"), (0.0, "T", dict(text="КОЛА", st="H", size=124, x=90, y=690, al="l", col="ink", anim="up")),
        (0.0, "IMG", dict(src="icon:cola", h=480, x=790, y=700, anim="pop", motion="float", until="литрам")), (0.2, "SFX", dict(name="fizz")),
        ("да-да", "BUB", dict(text="да-да", x=300, y=880, side="l")),
        ("странно", "T", dict(text="звучит странно", st="I", size=56, x=90, y=1010, al="l", col="mute", anim="type")),
        ("странно", "FULL", dict(src="photo:IMG_7959.HEIC", text="звучит странно?", sub="(но некоторым помогает)", until="глотков")),
        ("странно", "IMG", dict(src="icon:qmark", h=200, x=560, y=880, anim="pop", motion="wiggle", shadow=False, until="литрам")),
        ("глотков", "T", dict(text="пару глотков", st="H", size=72, x=90, y=1120, al="l", col="acc", anim="pop")),
        ("литрам", "IMG", dict(src="icon:colabottle", h=520, x=790, y=760, anim="pop")),
        ("литрам", "IMG", dict(src="icon:cross", h=420, x=790, y=760, anim="pop", shadow=False)),
        ("литрам", "T", dict(text="НЕ ЛИТРАМИ", st="H", size=92, x=540, y=1290, col="ink", anim="pop")), ("литрам", "SHAKE", {}),
        ("сахар", "BUB", dict(text="сахар", x=320, y=1440, side="l")), ("кофеин", "BUB", dict(text="кофеин", x=760, y=1440, side="r"))]),
    dict(bg="cream", num=4, els=[
        num(4, "acc"), (0.0, "T", dict(text="ГРАНАТ", st="H", size=92, x=90, y=680, al="l", col="ink", anim="up", until="изжог")),
        ("цитрус", "T", dict(text="& цитрусы", st="I", size=72, x=90, y=790, al="l", col="acc", anim="type", until="изжог")),
        (0.0, "IMG", dict(src="icon:fruits", h=320, x=770, y=760, anim="pop", motion="float", until="изжог")),
        ("кислые", "T", dict(text="кислые фрукты", st="L", size=58, x=90, y=930, al="l", col="ink", anim="type", until="изжог")),
        ("реакц", "T", dict(text="СМОТРИ\nНА РЕАКЦИЮ", st="H", size=62, x=90, y=1110, al="l", col="acc", anim="pop", until="изжог")),
        ("реакц", "CIRCLE", dict(src="vid:IMG_0755", d=290, x=790, y=1300, fx=0.55, fy=0.24, zoom=2.0, anim="pop", motion="float", until="изжог")),
        ("изжог", "IMG", dict(src="icon:flame", h=380, x=540, y=880, anim="pop", motion="wiggle")),
        ("изжог", "T", dict(text="ИЗЖОГА", st="H", size=130, x=540, y=1200, col="acc", anim="pop")),
        ("изжог", "T", dict(text="(у кого-то наоборот)", st="I", size=48, x=540, y=1310, col="mute", anim="type")),
        ("изжог", "GLITCH", {}), ("изжог", "SHAKE", {})]),
    dict(bg="ink", num=5, els=[
        num(5, "acc"), (0.0, "T", dict(text="ДУОНИКА", st="H", size=104, x=90, y=690, al="l", col="ink", anim="up")),
        (0.0, "IMG", dict(src="icon:blister", h=250, x=790, y=480, anim="pop", motion="float")),
        ("сильн", "T", dict(text="если токсикоз сильный", st="I", size=50, x=90, y=880, al="l", col="mute", anim="type")),
        ("терпеть", "T", dict(text="НЕ ТЕРПИ", st="H", size=112, x=90, y=1000, al="l", col="acc", anim="pop")), ("терпеть", "SHAKE", {}),
        ("врач", "IMG", dict(src="icon:med", h=200, x=880, y=1050, anim="pop", motion="pulse")),
        ("врач", "T", dict(text="обсудить с врачом", st="L", size=58, x=90, y=1130, al="l", col="ink", anim="type")),
        ("самостоят", "STAMP", dict(text="ТОЛЬКО С ВРАЧОМ", y=1360))]),
    dict(bg="cream", els=[
        ("важным", "T", dict(text="не заставляй себя", st="I", size=58, x=540, y=430, col="mute", anim="type")),
        ("кажется", "FULL", dict(src="photo:IMG_2194.HEIC", text="НЕ ЗАСТАВЛЯЙ СЕБЯ", sub="(есть большими порциями)", until="больш")),
        ("больш", "T", dict(text="БОЛЬШИЕ ПОРЦИИ", st="H", size=88, x=540, y=560, col="ink", anim="pop", until="перекус")),
        ("больш", "IMG", dict(src="icon:plate", h=480, x=540, y=900, anim="pop", until="перекус")),
        ("больш", "IMG", dict(src="icon:cross", h=420, x=540, y=900, anim="pop", shadow=False, until="перекус")),
        ("перекус", "T", dict(text="ПОНЕМНОГУ", st="H", size=112, x=540, y=560, col="acc", anim="pop")),
        ("перекус", "IMG", dict(src="icon:snack", h=440, x=540, y=900, anim="pop", motion="float")),
        ("дня", "IMG", dict(src="icon:clock", h=240, x=290, y=1200, anim="pop")),
        ("дня", "IMG", dict(src="icon:hands", h=240, x=290, y=1200, anim="pop", motion="spin", shadow=False)),
        ("дня", "T", dict(text="в течение дня", st="I", size=58, x=290, y=1400, col="ink", anim="type")),
        ("обед", "CARD", dict(src="86573FEB-025C-4508-97B3-CD96A1DBF3A8.jpg", x=790, y=1240, w=330, h=430, rot=6, anim="slide_r"))]),
    dict(bg="ink", els=[
        ("существует", "T", dict(text="НЕТ ОДНОГО\nСРЕДСТВА", st="H", size=96, x=540, y=520, col="ink", anim="pop")),
        ("всем", "T", dict(text="(для всех)", st="I", size=58, x=540, y=700, col="acc", anim="type")),
        (0.3, "ORBIT", dict(names=("lemon", "mineral", "cola", "fruits", "blister", "snack"), r=330, h=160, cy=1060)),
        ("пробова", "T", dict(text="пробуй", st="L", size=70, x=540, y=1060, col="ink", anim="type", until="тебе")),
        ("тебе", "IMG", dict(src="cut:IMG_9141", h=640, x=540, y=1080, anim="up")),
        ("тебе", "T", dict(text="ИМЕННО ТЕБЕ", st="H", size=100, x=540, y=1480, col="acc", anim="pop"))]),
    dict(bg="cream", els=[
        ("спасало", "T", dict(text="А ЧТО СПАСАЛО\nВАС?", st="H", size=96, x=540, y=520, col="ink", anim="pop")),
        (0.0, "CIRCLE", dict(src="vid:IMG_0755", d=380, x=540, y=830, fx=0.55, fy=0.24, zoom=1.8, anim="pop", motion="float")),
        ("кажется", "BUB", dict(text="а у меня…", x=260, y=1090, side="l")),
        ("каждой", "BUB", dict(text="мне помогало…", x=790, y=1120, side="r")),
        ("список", "BUB", dict(text="у меня свой список!", x=400, y=1240, side="l")),
        ("список", "IMG", dict(src="icon:list", h=190, x=850, y=1270, anim="pop", motion="float")),
        ("неожидан", "T", dict(text="(самые неожиданные средства)", st="I", size=46, x=540, y=1420, col="acc", anim="type"))]),
]

# ---------- pictures ----------
def shadowed(im, k=0.35):
    """a soft drop shadow under a cut-out, so it sits on the page like an object"""
    pad = int(im.height * 0.08); out = Image.new("RGBA", (im.width + 2 * pad, im.height + 2 * pad), (0, 0, 0, 0))
    a = Image.new("L", out.size, 0); a.paste(im.getchannel("A"), (pad + pad // 3, pad + pad // 2))
    sh = Image.new("RGBA", out.size, (30, 20, 15, 0)); sh.putalpha(a.filter(ImageFilter.GaussianBlur(pad / 2.2)).point(lambda v: int(v * k)))
    out.alpha_composite(sh); out.alpha_composite(im, (pad, pad)); return out

STICKERS = {"lemon": "lemon", "slice": "slice", "water": "water", "mineral": "mineral", "cola": "cola", "colabottle": "cola",
            "fruits": "fruits", "pom": "fruits", "blister": "blister", "snack": "snack"}   # her photo stickers replace the flat icons

class Pics:
    def __init__(self, cuts, mats, stickers=None): self.cuts, self.mats, self.st, self.c = cuts, mats, stickers, {}

    def get(self, src, h, crop=None, shadow=True):
        key = (src, h, crop, shadow)
        if key in self.c: return self.c[key]
        kind, name = src.split(":", 1)
        if kind == "icon" and self.st and name in STICKERS:
            im = Image.open(os.path.join(self.st, STICKERS[name] + "_c.png")).convert("RGBA")
            im = im.resize((round(im.width * h / im.height), h), Image.LANCZOS)
        elif kind == "icon": im = T.icon(name); im = im.resize((h, h), Image.LANCZOS)
        elif kind == "cut":
            im = Image.open(os.path.join(self.cuts, name + ".png")).convert("RGBA")
            if crop: im = im.crop((0, 0, im.width, int(im.height * crop)))
            im = im.resize((round(im.width * h / im.height), h), Image.LANCZOS)
        elif kind == "vid":
            fs = sorted(glob.glob(os.path.join(self.cuts, "v0755", name + "_*.png")))
            im = [Image.open(f).convert("RGBA") for f in fs]; im = [f.resize((round(f.width * h / f.height), h), Image.LANCZOS) for f in im]
            im = [shadowed(f) if shadow else f for f in im]; self.c[key] = im; return im
        if shadow: im = shadowed(im)
        self.c[key] = im; return im

    def full(self, src, crop=None):
        """her photo or clip filling the frame (1080x1920): one image, or a list of frames at 24 fps; crop keeps a photo's top part"""
        key = ("full", src, crop)
        if key in self.c: return self.c[key]
        kind, name = src.split(":", 1)
        if kind == "vid":
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                subprocess.run(["ffmpeg", "-v", "error", "-ss", "0.5", "-t", "6", "-i", os.path.join(self.mats, name + ".mov"), "-vf",
                                f"fps=24,scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}", "-q:v", "3",
                                os.path.join(td, "f_%04d.jpg")], check=True)
                im = [Image.open(f).convert("RGB") for f in sorted(glob.glob(os.path.join(td, "f_*.jpg")))]
        else:
            import pillow_heif; pillow_heif.register_heif_opener()
            ph = ImageOps.exif_transpose(Image.open(os.path.join(self.mats, name))).convert("RGB")
            if crop: ph = ph.crop((0, 0, ph.width, int(ph.height * crop)))
            im = ImageOps.fit(ph, (W, H), Image.LANCZOS, centering=(0.5, 0.35))
        self.c[key] = im; return im

    def circle(self, src, d, fx=0.5, fy=0.4, zoom=1.0, crop=None):
        """a round «кружок» of her (photo or clip): circle crop, cream ring, soft shadow"""
        key = ("circle", src, d, fx, fy, zoom, crop)
        if key in self.c: return self.c[key]
        if src.startswith("vid:"): fr = self.full(src); frames = fr
        else:                                                              # photos: straight from the original, square around her face
            import pillow_heif; pillow_heif.register_heif_opener()
            fr = ImageOps.exif_transpose(Image.open(os.path.join(self.mats, src.split(":", 1)[1]))).convert("RGB")
            if crop: fr = fr.crop((0, 0, fr.width, int(fr.height * crop)))
            frames = [fr]
        fw, fh = frames[0].size; side = int(min(fw, fh) / zoom)
        x0 = int(min(max(fx * fw - side / 2, 0), fw - side)); y0 = int(min(max(fy * fh - side / 2, 0), fh - side))
        m = Image.new("L", (d * 2, d * 2), 0); ImageDraw.Draw(m).ellipse((0, 0, d * 2 - 1, d * 2 - 1), fill=255); m = m.resize((d, d), Image.LANCZOS)
        out = []
        for f in frames:
            c = f.crop((x0, y0, x0 + side, y0 + side)).resize((d, d), Image.LANCZOS).convert("RGBA"); c.putalpha(m)
            ring = Image.new("RGBA", (d + 20, d + 20), (0, 0, 0, 0)); ImageDraw.Draw(ring).ellipse((0, 0, d + 19, d + 19), fill=CREAM + (255,))
            ring.alpha_composite(c, (10, 10)); out.append(shadowed(ring, 0.3))
        self.c[key] = out if isinstance(fr, list) else out[0]; return self.c[key]

    def card(self, name, w, h, rot):
        import pillow_heif; pillow_heif.register_heif_opener()
        ph = ImageOps.exif_transpose(Image.open(os.path.join(self.mats, name))).convert("RGB"); ph = ImageOps.fit(ph, (w, h), Image.LANCZOS)
        b = 14; c = Image.new("RGBA", (w + 2 * b, h + 2 * b), (255, 255, 255, 255)); c.paste(ph, (b, b))
        return shadowed(c.rotate(rot, resample=Image.BICUBIC, expand=True), 0.3)

def text_img(text, st, size, col):
    f = {"H": lambda: font(size, 800), "L": lambda: font(size, 300), "I": lambda: font(size, 300, True), "N": lambda: font(size, 200, True)}[st]()
    lines = text.split("\n"); lh = int(size * 1.08); w = int(max(f.getlength(l) for l in lines)) + 20; h = lh * len(lines) + int(size * 0.35)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for j, l in enumerate(lines): d.text((w / 2, j * lh + size * 0.92), l, font=f, fill=col + (255,), anchor="ms")
    return im

def outline_img(text, size, col):
    f = font(size, 800); w = int(f.getlength(text)) + 20; h = int(size * 1.3)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); ImageDraw.Draw(im).text((w / 2, size * 0.95), text, font=f, fill=(0, 0, 0, 0), stroke_width=3, stroke_fill=col + (255,), anchor="ms")
    return im

def bubble_img(text, side):
    f = font(40, 500); w = int(f.getlength(text)) + 70; h = 96
    im = Image.new("RGBA", (w + 20, h + 30), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((4, 4, w, h), radius=40, fill=(255, 255, 255), outline=INK, width=3)
    tx = 40 if side == "l" else w - 40; d.polygon([(tx - 12, h - 2), (tx + (-20 if side == "l" else 20), h + 24), (tx + 14, h - 2)], fill=(255, 255, 255), outline=INK)
    d.rectangle((tx - 10, h - 6, tx + 12, h - 1), fill=(255, 255, 255)); d.text((w / 2 + 2, h / 2 + 4), text, font=f, fill=INK, anchor="mm")
    return shadowed(im, 0.25)

# ---------- element runtime ----------
class El:
    def __init__(self, kind, t0, t1, o): self.kind, self.t0, self.t1, self.o = kind, t0, t1, o

def word_time(sc, key):
    if isinstance(key, (int, float)): return sc["t0"] + key
    for w, a, b in sc["words"]:
        if T.norm(w).startswith(key): return a
    return sc["t0"]

def transform(im, t, el):
    """animation in, idle motion and out → (image, dx, dy) or None"""
    o = el.o; u = t - el.t0; s, a, dx, dy, al = 1.0, 0.0, 0.0, 0.0, 1.0
    an = o.get("anim", "pop")
    if an == "pop": s = ease_back(u / 0.3)
    elif an == "up": dy = 90 * (1 - ease(u / 0.35)); al = ease(u / 0.25)
    elif an == "fade": al = ease(u / 0.4)
    elif an == "drop": k = min(u / 0.42, 1); dy = -560 * (1 - k) ** 2
    elif an == "slide_r": dx = 700 * (1 - ease(u / 0.35))
    mo = o.get("motion")
    if mo == "float": dy += 10 * math.sin(t * 2.3 + o.get("x", 0) * .01)
    elif mo == "wiggle": a = 6 * math.sin(t * 5)
    elif mo == "pulse": s *= 1 + 0.06 * math.sin(t * 6)
    elif mo == "spin": a = (t - el.t0) * 150
    if t > el.t1: k = ease((t - el.t1) / 0.2); s *= 1 - k; al *= 1 - k
    if s < 0.02 or al < 0.02: return None
    if abs(s - 1) > 0.01: im = im.resize((max(int(im.width * s), 1), max(int(im.height * s), 1)), Image.BILINEAR)
    if abs(a) > 0.5: im = im.rotate(a, resample=Image.BICUBIC, expand=True)
    if al < 0.99: im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * al)))
    return im, dx, dy

def build(scenes, total, pics):
    els, events, shakes, glitches, keyt = [], [], [], [], []
    for k, (sc, mo) in enumerate(zip(scenes, MONTAGE)):
        end = (scenes[k + 1]["t0"] - 0.22) if k + 1 < len(scenes) else total
        events.append((sc["t0"] - 0.24, "whoosh"))
        if mo.get("num"): events.append((sc["t0"] + 0.05, "ding"))
        for key, kind, o in mo["els"]:
            t = word_time(sc, key) + o.get("delay", 0.0); t1 = word_time(sc, o["until"]) if o.get("until") else end
            if kind in ("SHAKE", "GLITCH"):
                (shakes if kind == "SHAKE" else glitches).append(t); events.append((t, "shake")); continue
            if kind == "SFX": events.append((t, o["name"])); continue
            dark = mo["bg"] == "ink" or (mo["bg"] == "split" and o.get("side") == "r")
            col = (ON_INK.get(o.get("col", "ink"), C.get(o.get("col"))) if dark else C.get(o.get("col", "ink"))) if "col" in o else None
            el = El(kind, t, t1, dict(o, scene=k))
            if kind == "T":
                el.col = col
                if o.get("anim") == "type":
                    el.full = o["text"]; n = len(o["text"].replace(" ", "")); keyt += [t + i * TYPE_STEP for i in range(n)]
                else: el.img = text_img(o["text"], o["st"], o["size"], col); events.append((t, "pop"))
            elif kind == "R":
                el.img = outline_img(o["text"], o["size"], ON_INK["mute"] if dark else MAUVE); el.t0 += o.get("delay", 0)
            elif kind == "IMG":
                el.img = pics.get(o["src"], o["h"], o.get("crop"), o.get("shadow", True)); events.append((t, "pop"))
            elif kind == "CARD": el.img = pics.card(o["src"], o["w"], o["h"], o["rot"]); events.append((t, "whoosh"))
            elif kind in ("FULL", "BGFULL"): el.img = pics.full(o["src"], o.get("crop")); events.append((t, "whoosh"))
            elif kind == "CIRCLE":
                el.kind = "IMG"; el.img = pics.circle(o["src"], o["d"], o.get("fx", 0.5), o.get("fy", 0.4), o.get("zoom", 1.0), o.get("crop")); events.append((t, "pop"))
            elif kind == "BUB": el.img = bubble_img(o["text"], o["side"]); el.o.setdefault("anim", "pop"); events.append((t, "pop"))
            elif kind == "STAMP":
                el.img = T.stamp_img(o["text"]); el.o.update(x=540, anim="stamp"); events.append((t, "thud")); shakes.append(t)
            elif kind == "LINE": el.o["anim"] = "draw"
            elif kind == "ORBIT":
                for i, nm in enumerate(o["names"]):
                    e = El("IMG", t + 0.12 * i, end, dict(src="icon:" + nm, h=o["h"], anim="pop", scene=k, orbit=(o["r"], i * 2 * math.pi / len(o["names"]), o["cy"])))
                    e.img = pics.get("icon:" + nm, o["h"]); els.append(e); events.append((t + 0.12 * i, "pop"))
                continue
            fit_safe(el)
            els.append(el)
    return els, events, shakes, glitches, keyt

_GRAD = {}
def full_frame(fr, el, t, dark=False):
    """full-bleed photo / clip with a slow push-in; FULL fades in and carries a caption on a soft bottom shade"""
    o = el.o; src = el.img; u = t - el.t0
    im = src[int(u * 24) % len(src)] if isinstance(src, list) else src
    s = 1 + 0.06 * min(u / max(el.t1 - el.t0, 0.5), 1); w, h = int(W * s), int(H * s)
    im = im.resize((w, h), Image.BILINEAR).crop(((w - W) // 2, (h - H) // 2, (w - W) // 2 + W, (h - H) // 2 + H)).convert("RGBA")
    al = 1.0 if dark else min(ease(u / 0.15), 1 - ease((t - el.t1) / 0.15) if t > el.t1 else 1)
    if dark:
        im = Image.blend(im, Image.new("RGBA", (W, H), INK + (255,)), 0.72)
    else:
        if "g" not in _GRAD:
            a = (np.clip((np.arange(H) - H * 0.55) / (H * 0.4), 0, 1) ** 1.4 * 200).astype(np.uint8)
            g = Image.new("RGBA", (W, H), INK + (0,)); g.putalpha(Image.fromarray(np.repeat(a[:, None], W, 1))); _GRAD["g"] = g
        im.alpha_composite(_GRAD["g"])
        if o.get("text"): paste(im, text_img(o["text"], "H", 92, CREAM), int(W / 2 - text_img(o["text"], "H", 92, CREAM).width / 2), 1250)
        if o.get("sub"): sp = text_img(o["sub"], "I", 50, BLUSH); paste(im, sp, int(W / 2 - sp.width / 2), 1370)
    if al < 0.99: im.putalpha(int(255 * al))
    fr.alpha_composite(im)

def fit_safe(el):
    """move (and if it must, shrink) a piece so its resting rectangle lies inside SAFE"""
    o = el.o
    if el.kind in ("LINE", "FULL", "BGFULL", "R") or "orbit" in o: return
    if el.kind == "T": im = el.img if hasattr(el, "img") else text_img(el.full + "_", o["st"], o["size"], el.col)
    else: im = el.img[0] if isinstance(el.img, list) else el.img
    w, h = im.size; x, y = o.get("x", 540), o.get("y", 960); al = o.get("al", "c")
    sw, sh = SAFE[2] - SAFE[0], SAFE[3] - SAFE[1]; k = min(1.0, sw / w, sh / h)
    if k < 1 and el.kind != "T":                                           # too big for the safe area: scale the picture down
        sc = lambda f: f.resize((max(int(f.width * k), 1), max(int(f.height * k), 1)), Image.LANCZOS)
        el.img = [sc(f) for f in el.img] if isinstance(el.img, list) else sc(el.img); w, h = int(w * k), int(h * k)
    x0 = x if al == "l" else (x - w / 2 if al == "c" else x - w); top = y - h * (0.72 if el.kind == "T" else 0.5)
    dx = max(SAFE[0] - x0, 0) - max(x0 + w - SAFE[2], 0); dy = max(SAFE[1] - top, 0) - max(top + h - SAFE[3], 0)
    o["x"], o["y"] = x + dx, y + dy

def draw_bg(fr, bg, y0=0):
    d = ImageDraw.Draw(fr)
    if bg == "split": d.rectangle((0, y0, W // 2, H), fill=CREAM); d.rectangle((W // 2, y0, W, H), fill=INK)
    else: d.rectangle((0, y0, W, H), fill=INK if bg == "ink" else CREAM)

def render(scenes, total, pics, work):
    els, events, shakes, glitches, keyt = build(scenes, total, pics)
    starts = [s["t0"] for s in scenes]
    wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", os.path.join(work, "video_v2.mp4")], stdin=subprocess.PIPE)
    type_cache = {}
    for fi in range(int(total * FPS)):
        t = fi / FPS; fr = Image.new("RGBA", (W, H), CREAM + (255,))
        k = max([i for i, s in enumerate(starts) if s - 0.22 <= t] or [0])
        wipe = min(starts[k] - t, 0.22) if starts[k] > t and k > 0 else 0  # the next scene's ground slides up over this one
        cur = k - 1 if wipe > 0 and k > 0 else k
        draw_bg(fr, MONTAGE[cur]["bg"])
        for el in els:                                                     # her clip / photo behind the scene, darkened
            if el.kind == "BGFULL" and el.o["scene"] == cur and el.t0 <= t <= el.t1: full_frame(fr, el, t, dark=True)
        d = ImageDraw.Draw(fr)
        for el in els:
            if el.kind in ("FULL", "BGFULL"): continue
            if el.o["scene"] != cur or t < el.t0 or t > el.t1 + 0.2: continue
            o = el.o
            if el.kind == "LINE":                                          # dotted connector, drawn on
                pts = o["p"]; q = min((t - el.t0) / 0.45, 1); segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
                tot = sum(math.dist(a, b) for a, b in segs); dd = tot * q; col = BLUSH if MONTAGE[cur]["bg"] == "ink" else MAUVE
                for a, b in segs:
                    L_ = math.dist(a, b); n = int(L_ / 16)
                    for i in range(n):
                        if dd <= i * 16: break
                        x = a[0] + (b[0] - a[0]) * i / n; y = a[1] + (b[1] - a[1]) * i / n; d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=col)
                    dd -= L_
                continue
            if el.kind == "T" and o.get("anim") == "type":
                n = int((t - el.t0) / TYPE_STEP) + 1; s_ = el.full[:n]; cursor = n < len(el.full) or int(t * 3) % 2 == 0 and t - el.t0 < 1.2
                key = (id(el), s_, cursor)
                if key not in type_cache:
                    im = text_img(s_ + ("_" if cursor else ""), o["st"], o["size"], el.col); type_cache.clear() if len(type_cache) > 3000 else None; type_cache[key] = im
                im = type_cache[key]; dx = dy = 0; r = (im, 0, 0) if t <= el.t1 else transform(im, t, el)
                if r is None: continue
                im, dx, dy = r
            elif el.kind == "IMG" and isinstance(el.img, list):            # the cut-out video plays and loops
                r = transform(el.img[int((t - el.t0) * 24) % len(el.img)], t, el)
                if r is None: continue
                im, dx, dy = r
            elif o.get("anim") == "stamp":
                u = t - el.t0; s = 1.7 - 0.7 * ease(u / 0.22); im = el.img.resize((int(el.img.width * s), int(el.img.height * s)), Image.BILINEAR); dx = dy = 0
                if t > el.t1: continue
            else:
                r = transform(el.img, t, el)
                if r is None: continue
                im, dx, dy = r
            x, y = o.get("x", 540), o.get("y", 960)
            if "orbit" in o:
                rad, ph, cy = o["orbit"]; x = 540 + rad * math.cos(ph + t * 0.6); y = cy + rad * 0.8 * math.sin(ph + t * 0.6)
            al = o.get("al", "c"); px = x if al == "l" else (x - im.width / 2 if al == "c" else x - im.width)
            if el.kind == "R":                                             # outline copies above and below the word
                for j in range(1, o["n"] + 1):
                    for sgn in (-1, 1): paste(fr, im, int(px + dx - 10), int(y + dy + sgn * j * o["step"] - im.height * 0.72))
                continue
            py = y - im.height * (0.72 if el.kind == "T" else 0.5)
            paste(fr, im, int(px + dx - (10 if el.kind == "T" and al == "l" else 0)), int(py + dy))
        for el in els:                                                     # full-screen moments of her, over everything
            if el.kind == "FULL" and el.o["scene"] == cur and el.t0 <= t <= el.t1 + 0.15: full_frame(fr, el, t)
        if wipe > 0: draw_bg(fr, MONTAGE[k]["bg"], int(H * (wipe / 0.22)))
        d.rounded_rectangle((60, 236, W - 60, 242), radius=3, fill=(200, 190, 180)); d.rounded_rectangle((60, 236, 60 + int((W - 120) * t / total), 242), radius=3, fill=BROWN)
        if os.environ.get("SAFE_GUIDE"): ImageDraw.Draw(fr).rectangle(SAFE, outline=(255, 0, 0), width=3)
        img = np.asarray(fr.convert("RGB"))
        g = [s for s in glitches if 0 <= t - s < 0.35]
        if g:                                                              # colour split + torn slices
            k_ = 1 - (t - g[-1]) / 0.35; o_ = int(14 * k_); img = img.copy(); img[..., 0] = np.roll(img[..., 0], o_, 1); img[..., 2] = np.roll(img[..., 2], -o_, 1)
            rng = np.random.default_rng(fi)
            for _ in range(4):
                y0 = rng.integers(300, 1500); h_ = rng.integers(20, 70); img[y0:y0 + h_] = np.roll(img[y0:y0 + h_], int(rng.integers(-40, 40) * k_), 1)
        sh = [s for s in shakes if 0 <= t - s < 0.3]
        if sh: k_ = 1 - (t - sh[-1]) / 0.3; img = np.roll(img, (int(10 * k_ * math.sin(t * 90)), int(12 * k_ * math.cos(t * 77))), (0, 1))
        wr.stdin.write(np.ascontiguousarray(img).tobytes())
    wr.stdin.close(); wr.wait()
    return events, keyt

def mix(work, events, keyt, out):
    v = T.FX.read(os.path.join(work, "voice.wav")); bank = T.sfx_bank(); N = len(v); bus = np.zeros((N, 2), np.float32)
    def place(x, t):
        i = int(t * SR); n_ = min(len(x), N - i)
        if n_ > 0 and i >= 0: bus[i:i + n_] += x[:n_]
    for t, name in events: place(bank[name](), t)
    for t in keyt: place(bank["key"]() * 0.8, t)
    lv = np.convolve(np.abs(v).mean(1), np.ones(960) / 960, "same"); g = np.where(lv > T.FX.db(-40), T.FX.db(-4), 1.0)
    y = v + bus * g[:, None].astype(np.float32); y *= min(1, T.FX.db(-1) / (np.abs(y).max() + 1e-9))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-i", os.path.join(work, "video_v2.mp4"),
                    "-map", "1:v", "-map", "0:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                   input=y.astype(np.float32).tobytes(), check=True)

if __name__ == "__main__":
    work, cuts, mats, out = sys.argv[1:5]; stickers = sys.argv[5] if len(sys.argv) > 5 else None
    tm = json.load(open(os.path.join(work, "timing.json"))); scenes = [dict(t0=s["t0"], dur=s["dur"], words=s["words"]) for s in tm]
    total = T.FX.read(os.path.join(work, "voice.wav")).shape[0] / SR
    events, keyt = render(scenes, total, Pics(cuts, mats, stickers), work)
    mix(work, events, keyt, out)
