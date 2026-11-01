"""@mcg.va — «Как справляться с токсикозом?», version 4: mostly animation on her own voice (from her edited talking video),
she appears only now and then — her picture shrinks into a кружок at the start, small кружки later, a zoomed-out card at the end.
Type: thin Montserrat — headings in CAPS, everything else in small letters; minimal flat icons; subtitles from her speech.
  python3 toxicosis_v4.py BASE_VIDEO WORDS.json WORK_DIR OUT.mp4 [END_SECONDS]
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
M4 = [
    dict(bg="ink", els=[
        me(0.0, "shrink", d=380, x=540, y=1080, fx=0.5, fy=0.36, zoom=2.2),
        ("как", "T", dict(text="как справляться\nс токсикозом?", st="H", size=78, x=540, y=430, col="ink", anim="up")),
        ("отдельная", "T", dict(text="(отдельная история беременности)", st="I", size=44, x=540, y=680, col="acc", anim="type"))]),
    dict(bg="cream", els=[
        ("буквально", "T", dict(text="буквально", st="I", size=56, x=540, y=440, col="acc", anim="type")),
        ("тошнит", "T", dict(text="тошнит", st="H", size=150, x=540, y=630, col="ink", anim="pop")), ("тошнит", "GLITCH", {}), ("тошнит", "SHAKE", {}),
        ("всего", "T", dict(text="от всего", st="L", size=64, x=540, y=780, col="ink", anim="up")),
        ic("любого", "smell", 540, 1050, 260, anim="fade"),
        ("запаха", "T", dict(text="(от любого запаха)", st="I", size=50, x=540, y=1290, col="acc", anim="type"))]),
    dict(bg="split", els=[
        ("хочешь", "T", dict(text="хочешь\nесть", st="H", size=74, x=270, y=520, col="ink", anim="up", side="l")),
        ic("хочешь", "plate", 270, 880, 300),
        ("не", "T", dict(text="но не\nможешь", st="H", size=74, x=810, y=520, col="ink", anim="up", side="r")),
        ic("можешь", "cross", 270, 880, 260, motion="none"),
        ic("можешь", "sick", 810, 900, 280, motion="wiggle"), ("можешь", "SHAKE", {})]),
    dict(bg="cream", els=[
        ("питалась", "T", dict(text="что спасало", st="H", size=70, x=540, y=420, col="ink", anim="up")),
        ("чипсами", "BUB", dict(text="чипсы", x=280, y=700, side="l")),
        ("бармеладками", "BUB", dict(text="кислые мармеладки", x=700, y=830, side="r")),
        ("помогали", "T", dict(text="(только они и помогали)", st="I", size=48, x=540, y=1010, col="acc", anim="type")),
        me("только", "circle", d=300, x=540, y=1260, fx=0.5, fy=0.36, zoom=2.2)]),
    dict(bg="ink", num=1, els=[
        num(1), (0.0, "T", dict(text="лимон", st="H", size=110, x=90, y=690, al="l", col="ink", anim="up")),
        ic(0.0, "lemon", 780, 560, 300, motion="wiggle"),
        ("добавить", "T", dict(text="пару кусочков\nв воду", st="L", size=58, x=90, y=900, al="l", col="ink", anim="type")),
        ic("воду", "water", 780, 1010, 360, motion="none"), ic("воду", "slice", 780, 900, 130, anim="drop", motion="none"), ("воду", "SFX", dict(name="splash")),
        ("всем", "T", dict(text="(помогает не всем)", st="I", size=50, x=90, y=1250, al="l", col="acc", anim="type"))]),
    dict(bg="cream", num=2, els=[
        num(2), (0.0, "T", dict(text="боржоми", st="H", size=104, x=90, y=690, al="l", col="ink", anim="up")),
        ic(0.0, "mineral", 800, 820, 420),
        ("действительно", "T", dict(text="правда помогал", st="L", size=58, x=90, y=880, al="l", col="ink", anim="type")),
        ic("короткий", "clock", 280, 1120, 220, motion="none"), ic("короткий", "hands", 280, 1120, 220, motion="spin"),
        ("короткий", "T", dict(text="(но ненадолго)", st="I", size=52, x=90, y=1320, al="l", col="acc", anim="type"))]),
    dict(bg="ink", num=3, els=[
        num(3), (0.0, "T", dict(text="кока-кола", st="H", size=100, x=90, y=690, al="l", col="ink", anim="up")),
        ic(0.0, "cola", 800, 800, 420, until="волшебным"), ("кока", "SFX", dict(name="fizz")),
        ("лучшим", "T", dict(text="лучший вариант", st="L", size=60, x=90, y=880, al="l", col="ink", anim="type")),
        me("для", "circle", d=260, x=250, y=1150, fx=0.5, fy=0.36, zoom=2.2, until="волшебным"),
        ("волшебным", "T", dict(text="волшебный\nнапиток", st="H", size=84, x=540, y=1020, col="acc", anim="pop")),
        ic("волшебным", "heart", 540, 1300, 170, motion="pulse")]),
    dict(bg="cream", num=4, els=[
        num(4), (0.0, "T", dict(text="гранат\nи цитрусы", st="H", size=88, x=90, y=690, al="l", col="ink", anim="up", until="изжогу")),
        ic(0.0, "pom", 780, 640, 260, until="изжогу"), ic("цитрусы", "orange", 840, 930, 230, until="изжогу"),
        ("минус", "T", dict(text="но есть минус", st="L", size=58, x=90, y=1000, al="l", col="ink", anim="type", until="изжогу")),
        ("изжогу", "T", dict(text="изжога", st="H", size=130, x=540, y=700, col="acc", anim="pop")), ("изжогу", "GLITCH", {}),
        ic("изжогу", "flame", 540, 1000, 280, motion="wiggle")]),
    dict(bg="ink", num=5, els=[
        num(5), (0.0, "T", dict(text="дуоника", st="H", size=104, x=90, y=690, al="l", col="ink", anim="up")),
        ic(0.0, "blister", 800, 560, 260),
        ("сильный", "T", dict(text="(если токсикоз сильный)", st="I", size=48, x=90, y=860, al="l", col="ink", anim="type")),
        ("специальный", "T", dict(text="для беременных", st="L", size=58, x=90, y=960, al="l", col="ink", anim="type")),
        ic("врач", "med", 800, 1080, 200, motion="pulse"),
        ("назначает", "STAMP", dict(text="ТОЛЬКО С ВРАЧОМ", y=1260))]),
    dict(bg="cream", els=[
        ("большими", "T", dict(text="не большими\nпорциями", st="H", size=80, x=540, y=470, col="ink", anim="up", until="маленькими")),
        ic("большими", "plate", 540, 900, 380, until="маленькими"), ic("большими", "cross", 540, 900, 340, motion="none", until="маленькими"),
        ("маленькими", "T", dict(text="понемногу\nи чаще", st="H", size=88, x=540, y=470, col="acc", anim="pop")),
        *[ic("маленькими", "plate", x, 880, 200, delay=0.12 * i) for i, x in enumerate((240, 540, 840))],
        ic("чаще", "clock", 540, 1200, 200, motion="none"), ic("чаще", "hands", 540, 1200, 200, motion="spin")]),
    dict(bg="ink", els=[
        ("существует", "T", dict(text="нет единого\nсредства", st="H", size=84, x=540, y=470, col="ink", anim="up")),
        ("единого", "ORBIT", dict(names=("lemon", "mineral", "cola", "pom", "orange", "blister"), r=300, h=140, cy=1000)),
        ("сожалению", "T", dict(text="(к сожалению)", st="I", size=52, x=540, y=1000, col="acc", anim="type"))]),
    dict(bg="cream", els=[
        me("может", "card", w=405, h=720, x=540, y=900),
        ("может", "T", dict(text="а что помогало\nвам?", st="H", size=70, x=540, y=380, col="ink", anim="up")),
        ("варианты", "BUB", dict(text="а у меня…", x=230, y=760, side="l")),
        ("идеи", "BUB", dict(text="мне помогало…", x=830, y=980, side="r")),
        ("делитесь", "T", dict(text="(делитесь в комментариях)", st="I", size=46, x=540, y=1330, col="acc", anim="type"))]),
]

class Base:
    """her video, decoded once forward; frame for time t"""
    def __init__(self, path, fps=25):
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H},fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                  stdout=subprocess.PIPE)
        self.fps, self.i, self.cur = fps, -1, None
    def __call__(self, t):
        want = int(t * self.fps)
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
    base, words_json, work, out = sys.argv[1:5]; end = float(sys.argv[5]) if len(sys.argv) > 5 else None
    os.makedirs(work, exist_ok=True)
    words = [tuple(w) for w in json.load(open(words_json))]
    total = end or float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", base], capture_output=True, text=True).stdout)
    V.MONTAGE[:] = M4; V.text_img = text_thin; V.T.stamp_img = stamp_thin; V.BASE_FRAME = Base(base); V.POST = subtitles(words)
    scenes = scenes_from(words, total)
    events, keyt = V.render(scenes, total, V.Pics(work, work), work)
    V3.mix(base, total, events, os.path.join(work, "video_v2.mp4"), out)
