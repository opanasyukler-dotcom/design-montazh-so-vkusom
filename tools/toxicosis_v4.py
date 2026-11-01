"""@mcg.va — «Как справляться с токсикозом?», version 4: mostly animation on her own voice (from her edited talking video),
she appears only now and then — her picture shrinks into a кружок at the start, small кружки later, a zoomed-out card at the end.
Type: thin Montserrat — headings in CAPS, everything else in small letters; her photo stickers + minimal flat icons; subtitles from her speech.
  python3 toxicosis_v4.py BASE_VIDEO WORDS.json STICKERS_DIR WORK_DIR OUT.mp4 [END_SECONDS]
"""
import json, os, re, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import toxicosis_v2 as V
import toxicosis_v3 as V3
from toxicosis_v2 import W, H, CREAM, INK, BROWN, MAUVE, BLUSH, font, ease

def text_thin(text, st, size, col):
    """headings «H»: thin caps; «L» thin small letters; «I» thin italic small letters; «N» the big thin italic number"""
    if st == "H": text = text.upper()
    elif st in ("L", "I"): text = text.lower()
    f = {"H": lambda: font(size, 300), "L": lambda: font(size, 300), "I": lambda: font(size, 300, True), "N": lambda: font(size, 200, True)}[st]()
    lines = text.split("\n"); lh = int(size * 1.12); w = int(max(f.getlength(l) for l in lines)) + 20; h = lh * len(lines) + int(size * 0.35)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for j, l in enumerate(lines): d.text((w / 2, j * lh + size * 0.92), l, font=f, fill=col + (255,), anchor="ms")
    return im

def stamp_thin(text):
    f = font(58, 400); w = int(f.getlength(text)) + 90; h = 120
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((5, 5, w - 6, h - 6), radius=24, outline=(206, 72, 60), width=5); d.text((w / 2, h / 2 + 2), text, font=f, fill=(206, 72, 60), anchor="mm")
    return im.rotate(-5, resample=Image.BICUBIC, expand=True)

def num(n): return (0.0, "T", dict(text=f"{n:02d}.", st="N", size=210, x=80, y=500, al="l", col="acc", anim="pop"))
def ic(key, name, x, y, h, **kw): return (key, "IMG", dict(src="icon:" + name, h=h, x=x, y=y, anim=kw.pop("anim", "pop"), motion=kw.pop("motion", "float"), shadow=False, **kw))
def me(key, mode, **kw): return (key, "BASE", dict(mode=mode, **kw))

# scene starts: (first word, earliest time); the montage per scene (thin caps headings, small-letter text)
STARTS = [("как", 0), ("это", 4), ("и", 7.8), ("поэтому", 10.5), ("первое", 14.5), ("также", 19.3), ("и", 24.2), ("ну", 32), ("также", 35),
          ("также", 41.3), ("и", 45.1), ("может", 48.8)]
# layout: text only in the top band (y 300–820), one or two pictures in the middle band (y 860–1320), subtitles at 1420
FACE = dict(fx=0.52, fy=0.52, zoom=1.1)                                    # her head and shoulders (face centre ≈ 0.52 × 0.46)
M4 = [
    dict(bg="ink", els=[
        me(0.0, "shrink", d=560, x=540, y=1040, **FACE),
        ("как", "T", dict(text="как справляться\nс токсикозом?", st="H", size=76, x=540, y=430, col="ink", anim="up")),
        ("отдельная", "T", dict(text="(отдельная история беременности)", st="I", size=42, x=540, y=650, col="acc", anim="type"))]),
    dict(bg="cream", els=[
        me(0.0, "circle", d=480, x=540, y=1060, **FACE),
        ("тошнит", "T", dict(text="тошнит", st="H", size=140, x=540, y=470, col="ink", anim="pop")), ("тошнит", "GLITCH", {}),
        ("всего", "T", dict(text="от всего", st="L", size=60, x=540, y=600, col="ink", anim="up")),
        ("запаха", "T", dict(text="(от любого запаха)", st="I", size=48, x=540, y=690, col="acc", anim="type")),
        ic("запаха", "smell", 160, 1000, 160, anim="fade")]),
    dict(bg="split", els=[
        ("хочешь", "T", dict(text="хочешь\nесть", st="H", size=72, x=270, y=520, col="ink", anim="up", side="l")),
        ic("хочешь", "plate", 270, 1050, 300),
        ("не", "T", dict(text="но не\nможешь", st="H", size=72, x=810, y=520, col="ink", anim="up", side="r")),
        me("не", "circle", d=340, x=775, y=1050, **FACE),
        ic("можешь", "cross", 270, 1050, 260, motion="none")]),
    dict(bg="cream", els=[
        ("питалась", "T", dict(text="что спасало", st="H", size=70, x=540, y=420, col="ink", anim="up")),
        ("чипсами", "BUB", dict(text="чипсы", x=300, y=600, side="l")),
        ("бармеладками", "BUB", dict(text="кислые мармеладки", x=730, y=640, side="r")),
        me("питалась", "card", w=340, h=604, x=540, y=1030)]),
    dict(bg="ink", num=1, els=[
        num(1), (0.0, "T", dict(text="лимон", st="H", size=110, x=90, y=690, al="l", col="ink", anim="up")),
        ic(0.0, "lemon", 300, 1060, 300, motion="wiggle"),
        ic("воду", "water", 770, 1080, 330, motion="none"), ic("воду", "slice", 770, 980, 120, anim="drop", motion="none"), ("воду", "SFX", dict(name="splash")),
        ("всем", "T", dict(text="(помогает не всем)", st="I", size=48, x=540, y=820, col="acc", anim="type"))]),
    dict(bg="cream", num=2, els=[
        num(2), (0.0, "T", dict(text="боржоми", st="H", size=104, x=90, y=690, al="l", col="ink", anim="up")),
        ic(0.0, "mineral", 300, 1070, 380),
        me("помогал", "card", w=320, h=569, x=760, y=1060),
        ("короткий", "T", dict(text="(но ненадолго)", st="I", size=50, x=540, y=820, col="acc", anim="type"))]),
    dict(bg="ink", num=3, els=[
        num(3), (0.0, "T", dict(text="кока-кола", st="H", size=88, x=90, y=690, al="l", col="ink", anim="up")),
        ic(0.0, "cola", 300, 1070, 330, until="волшебным"),
        me("для", "circle", d=400, x=760, y=1070, **FACE, until="волшебным"),
        ("лучшим", "T", dict(text="лучший вариант", st="L", size=56, x=540, y=820, col="ink", anim="type", until="волшебным")),
        ("волшебным", "T", dict(text="волшебный\nнапиток", st="H", size=84, x=540, y=1000, col="acc", anim="pop")),
        ic("волшебным", "heart", 540, 1250, 150, motion="pulse")]),
    dict(bg="cream", num=4, els=[
        num(4), (0.0, "T", dict(text="гранат\nи цитрусы", st="H", size=84, x=90, y=690, al="l", col="ink", anim="up", until="изжогу")),
        ic(0.0, "fruits", 540, 1090, 340, until="изжогу"),
        ("изжогу", "T", dict(text="изжога", st="H", size=130, x=540, y=700, col="acc", anim="pop")), ("изжогу", "GLITCH", {}),
        ic("изжогу", "flame", 540, 1070, 300, motion="wiggle")]),
    dict(bg="ink", num=5, els=[
        num(5), (0.0, "T", dict(text="дуоника", st="H", size=104, x=90, y=690, al="l", col="ink", anim="up")),
        me(0.0, "circle", d=420, x=300, y=1060, **FACE, until="назначает"),
        ic("препарат", "blister", 770, 1060, 220, until="назначает"),
        ("сильный", "T", dict(text="(если токсикоз сильный)", st="I", size=46, x=540, y=820, col="acc", anim="type")),
        ("назначает", "STAMP", dict(text="ТОЛЬКО С ВРАЧОМ", y=1060))]),
    dict(bg="cream", els=[
        ("большими", "T", dict(text="не большими\nпорциями", st="H", size=78, x=540, y=480, col="ink", anim="up", until="маленькими")),
        ic("большими", "plate", 540, 1060, 360, until="маленькими"), ic("большими", "cross", 540, 1060, 320, motion="none", until="маленькими"),
        ("маленькими", "T", dict(text="понемногу\nи чаще", st="H", size=84, x=540, y=480, col="acc", anim="up")),
        ic("маленькими", "snack", 330, 1070, 320), ic("чаще", "clock", 780, 1070, 220, motion="none"), ic("чаще", "hands", 780, 1070, 220, motion="spin")]),
    dict(bg="ink", els=[
        ("существует", "T", dict(text="нет единого\nсредства", st="H", size=82, x=540, y=470, col="ink", anim="up")),
        ("единого", "ORBIT", dict(names=("lemon", "mineral", "cola", "fruits", "blister", "snack"), r=300, h=140, cy=1070)),
        ("сожалению", "T", dict(text="(к сожалению)", st="I", size=50, x=540, y=1070, col="acc", anim="type"))]),
    dict(bg="cream", els=[
        ("может", "T", dict(text="а что помогало\nвам?", st="H", size=70, x=540, y=420, col="ink", anim="up")),
        me("может", "card", w=360, h=640, x=540, y=1040),
        ("варианты", "BUB", dict(text="а у меня…", x=220, y=820, side="l")),
        ("идеи", "BUB", dict(text="мне помогало…", x=840, y=1290, side="r"))]),
]

class Base:
    """her video, decoded once forward; frame for time t"""
    def __init__(self, path, fps=25, last=None):
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H},fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                  stdout=subprocess.PIPE)
        self.fps, self.i, self.cur, self.last = fps, -1, None, last         # after «last» seconds: hold (her edit ends in a black outro)
    def __call__(self, t):
        want = int(min(t, self.last or t) * self.fps)
        while self.i < want:
            b = self.p.stdout.read(W * H * 3)
            if len(b) < W * H * 3: break
            self.cur = Image.frombuffer("RGB", (W, H), b); self.i += 1
        return self.cur

def subtitles(words):
    phrases, _ = V3.captions(words); f = font(48, 400)
    def draw(fr, t):
        ph = next((p for p in phrases if p[0][1] - 0.05 <= t <= p[-1][2] + 0.2), None)
        if not ph: return
        y = 1420; lum = np.asarray(fr.crop((200, y - 30, 880, y + 30)).convert("L")).mean(); base = INK if lum > 128 else CREAM
        d = ImageDraw.Draw(fr); ws = [(w.lower(), a, b) for w, a, b in ph]; wd = [f.getlength(w) for w, _, _ in ws]
        x = W / 2 - (sum(wd) + 18 * (len(ws) - 1)) / 2
        for (w, a, b), ww in zip(ws, wd):
            if a - 0.03 <= t <= b + 0.05:
                d.rounded_rectangle((x - 16, y - 36, x + ww + 16, y + 36), radius=36, fill=BLUSH); d.text((x + ww / 2, y), w, font=f, fill=INK, anchor="mm")
            else: d.text((x + ww / 2, y), w, font=f, fill=base, anchor="mm")
            x += ww + 18
    return draw

def scenes_from(words, total):
    starts = []; after = 0
    for key, a in STARTS:
        t = V3.word_at(words, key, max(a, after)); starts.append(t); after = t + 0.05
    sc = []
    for i, t in enumerate(starts):
        t0 = 0.0 if i == 0 else t - 0.05; t1 = starts[i + 1] - 0.05 if i + 1 < len(starts) else total
        sc.append(dict(t0=t0, dur=t1 - t0, words=[w for w in words if t0 <= w[1] < t1]))
    return sc

if __name__ == "__main__":
    base, words_json, stickers, work, out = sys.argv[1:6]; end = float(sys.argv[6]) if len(sys.argv) > 6 else None
    os.makedirs(work, exist_ok=True)
    words = [tuple(w) for w in json.load(open(words_json))]
    total = end or float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", base], capture_output=True, text=True).stdout)
    V.MONTAGE[:] = M4; V.text_img = text_thin; V.T.stamp_img = stamp_thin; V.BASE_FRAME = Base(base, last=max(w[2] for w in words)); V.POST = subtitles(words)
    scenes = scenes_from(words, total)
    voice_end = max(w[2] for w in words) + 0.06                            # right after «комментариях», nothing of the next phrase
    events, keyt = V.render(scenes, total, V.Pics(work, work, stickers), work)
    V3.mix(base, total, events, os.path.join(work, "video_v2.mp4"), out, voice_end=voice_end)
