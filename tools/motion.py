import re, subprocess, math, wave, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
SW, SH = 720, 1280
FONTS = "fonts2/M%d.ttf"
PINK = (243, 125, 184); PILL = (244, 156, 198); YEL = (246, 226, 127)
WHITE = (255, 255, 255); DARK = (30, 30, 30); GREY = (90, 90, 90)

# word start times (new timeline) from the subtitle file
WT = []
for l in open("subs.ass"):
    if l.startswith("Dialogue"):
        h, m, s = l.split(",")[1].split(":"); WT.append(int(h)*3600+int(m)*60+float(s))
DUR = 54.15

# scenes: mode, zoom, lines: (anchor, y, [(nwords, text, style)])
SC = [
 ("full",1.0,[("L",330,[(1,"сегодня","r"),(2,"у нас","r")]),("L",405,[(1,"редукционная","r")]),("L",470,[(1,"маммопластика","a")])]),
 ("full",1.18,[("R",330,[(2,"у нас","r")]),("R",400,[(1,"гигантомастия","a")])]),
 ("card",1.0,[("L",200,[(1,"достаточно","r"),(1,"большое","r"),(1,"расстояние","r")]),("L",270,[(1,"от","r"),(2,"яремной вырезки","a")]),("R",430,[(1,"до","r"),(1,"сосочка","p")])]),
 ("full",1.0,[("L",330,[(1,"можно","r"),(1,"даже","r")]),("L",400,[(1,"померить","a")]),("R",560,[(3,"((померяй, пожалуйста, григорий))","b")])]),
 ("full",1.18,[("L",330,[(1,"от","r"),(2,"яремной вырезки","r")]),("L",400,[(1,"до","r"),(1,"соска","a")]),("R",560,[(3,"померим, какое расстояние","p")])]),
 ("full",1.0,[("L",330,[(1,"мы","r"),(1,"будем","r"),(1,"использовать","r")]),("L",400,[(2,"метод рибейро","a")]),("R",560,[(1,"или","r"),(1,"аутоимплант","p")])]),
 ("full",1.18,[("L",330,[(1,"для","r"),(1,"того,","r"),(1,"чтобы","r")]),("L",400,[(1,"верхний","r"),(1,"полюс","a")]),("R",560,[(1,"был","r"),(1,"более","r")]),("R",625,[(1,"наполнен","y")])]),
 ("card",1.0,[("C",200,[(2,"сколько там?","r")]),("C",280,[(3,"32 см","n")])]),
 ("full",1.0,[("L",330,[(1,"уменьшим","r")]),("L",395,[(1,"грудь","a")]),("R",560,[(1,"сделаем","r"),(1,"повыше","r")]),("R",630,[(2,"сосково-ареолярный комплекс","p")])]),
 ("full",1.18,[("L",330,[(2,"и постараемся,","r")]),("L",400,[(4,"чтобы это всё было","r")]),("L",470,[(1,"очень","r")]),("L",535,[(1,"симметрично","a")]),("R",700,[(1,"красиво","y")])]),
 ("card",1.0,[("L",200,[(2,"а какой","r")]),("L",265,[(1,"объём","a")]),("R",415,[(3,"мы хотим оставить","r")]),("R",490,[(1,"груди?","p")])]),
 ("full",1.0,[("L",330,[(5,"((ну вот, вы сами смотрите))","b")]),("L",420,[(1,"по","r"),(1,"моим","r")]),("L",490,[(2,"внешним данным","a")])]),
 ("full",1.18,[("L",330,[(4,"я бы не стала","r")]),("L",400,[(2,"очень много","a")]),("L",540,[(1,"убирать","p")])]),
 ("full",1.0,[("R",330,[(2,"потому что","r")]),("R",400,[(3,"я всегда за","r")]),("R",470,[(1,"сохранность","a")]),("R",610,[(1,"ткани","p")])]),
 ("card",1.0,[("L",210,[(3,"за то, чтобы","r")]),("L",280,[(1,"побольше","a")]),("R",450,[(1,"оставлять","p")])]),
 ("bw",1.1,[("C",470,[(1,"но","r"),(1,"тут","r")]),("C",540,[(3,"по вашему желанию","a")])]),
 ("full",1.0,[("L",330,[(3,"кто-то говорит мне:","r")]),("R",420,[(2,"((очень тяжело))","b")]),("L",520,[(4,"я не хочу там","r")]),("L",590,[(2,"третий размер","a")])]),
 ("full",1.18,[("R",330,[(3,"хочу там или","r")]),("R",400,[(1,"четвёртый","a")]),("R",545,[(1,"троечку","p")])]),
 ("card",1.0,[("C",190,[(1,"хорошо,","r"),(1,"тогда","r")]),("C",260,[(2,"будем оставлять","r")]),("C",330,[(1,"троечку","n")])]),
 ("full",1.0,[("L",330,[(1,"да,","r"),(1,"двоечка","r")]),("L",400,[(1,"маловато","a")]),("R",545,[(1,"((конечно))","b")])]),
]

# ---------- token rendering ----------
SHEAR = 0.18
def font(w, s): return ImageFont.truetype(FONTS % w, s)
def text_img(txt, f, col):
    asc, desc = f.getmetrics()
    w = int(f.getlength(txt)) + 8
    im = Image.new("RGBA", (w, asc + desc + 8), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((4, 4 + asc), txt, font=f, fill=col + (255,), anchor="ls")
    a = np.asarray(im).astype(np.float32) / 255
    return a, 4 + asc
def italic(a, base):
    h, w = a.shape[:2]; ext = int(SHEAR * h) + 2
    M = np.float32([[1, -SHEAR, SHEAR * h], [0, 1, 0]])
    return cv2.warpAffine(a, M, (w + ext, h), flags=cv2.INTER_LINEAR), base
def premul(a):
    a = a.copy(); a[..., :3] *= a[..., 3:4]; return a
def rotate(a, deg):
    h, w = a.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1)
    c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_LINEAR)

def render(txt, style, card):
    """returns premultiplied RGBA float image and baseline row"""
    if style == "r":
        a, b = italic(*text_img(txt, font(500, 58), DARK if card else WHITE))
    elif style == "b":
        a, b = italic(*text_img(txt, font(300, 46), GREY if card else WHITE))
    elif style in ("a", "y"):
        s = 108
        while font(900, s).getlength(txt) > 860: s -= 4
        a, b = text_img(txt, font(900, s), PINK if style == "a" else YEL)
        a = cv2.resize(a, (int(a.shape[1] * 1.06), a.shape[0]))
        a, b = italic(a, b)
    elif style == "n":
        a, b = italic(*text_img(txt, font(900, 210), PINK))
    elif style == "p":
        t, _ = text_img(txt, font(600, 44), WHITE)
        th, tw = t.shape[:2]; ph, pw = th + 6, tw + 44
        im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        ImageDraw.Draw(im).rounded_rectangle((0, 0, pw - 1, ph - 1), radius=ph // 2, fill=PILL + (255,))
        bg = np.asarray(im).astype(np.float32) / 255
        y0, x0 = 3, 22
        ta = t[..., 3:4]
        bg[y0:y0 + th, x0:x0 + tw, :3] = t[..., :3] * ta + bg[y0:y0 + th, x0:x0 + tw, :3] * (1 - ta)
        a = rotate(bg, 3); b = int(a.shape[0] * 0.72)
    return premul(a), b

class Tok:
    pass

scenes = []
wi = 0
for si, (mode, zoom, lines) in enumerate(SC):
    card = mode == "card"; toks = []
    for anc, y, items in lines:
        imgs = []
        for n, txt, st in items:
            tk = Tok(); tk.t0 = WT[wi]; tk.style = st; tk.txt = txt; wi += n
            if st == "b":   # typewriter: prefix images
                tk.frames = [render(txt[:k] + ("_" if k < len(txt) else ""), "r" if False else "b", card) for k in range(len(txt) + 1)]
                tk.img, tk.base = tk.frames[-1]
            else:
                tk.img, tk.base = render(txt, st, card)
            imgs.append(tk)
        gap = 4
        total = sum(t.img.shape[1] for t in imgs) + gap * (len(imgs) - 1)
        if total > W - 150:
            f = (W - 150) / total
            for t in imgs:
                t.img = cv2.resize(t.img, (int(t.img.shape[1] * f), int(t.img.shape[0] * f))); t.base = int(t.base * f)
                if t.style == "b": t.frames = [(cv2.resize(a, (max(1, int(a.shape[1] * f)), int(a.shape[0] * f))), b) for a, b in t.frames]
            total = sum(t.img.shape[1] for t in imgs) + gap * (len(imgs) - 1)
        x = 80 if anc == "L" else (W - 80 - total if anc == "R" else (W - total) // 2)
        bl = y + max((t.base if t.style != "p" else 62) for t in imgs)
        for t in imgs:
            t.x = x; x += t.img.shape[1] + gap
            t.y = (bl - t.base) if t.style != "p" else (bl - 18 - t.img.shape[0] // 2)
            if t.style != "p" and card is False:
                sh = cv2.GaussianBlur(t.img[..., 3], (0, 0), 6)
                t.shadow = np.minimum(1, sh * 0.6)
            else:
                t.shadow = None
            toks.append(t)
    scenes.append(dict(mode=mode, zoom=zoom, toks=toks, start=max(0, toks[0].t0 - 0.08)))
assert wi == len(WT), (wi, len(WT))
for i, s in enumerate(scenes):
    s["end"] = scenes[i + 1]["start"] if i + 1 < len(scenes) else DUR + 1

def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def back(p):
    p = min(max(p, 0), 1); c = 1.9; return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2

def blend(dst, img, x, y, alpha):
    if alpha <= 0.003: return
    h, w = img.shape[:2]
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4] * alpha
    d += s[..., :3] * alpha

def draw_tok(dst, t, tt, sc, card):
    age = tt - t.t0
    if age < 0: return
    out = max(0, (tt - (sc["end"] - 0.16)) / 0.16)
    st = t.style; img = t.img; dx = dy = 0; al = 1; blur = 0; scl = 1
    if st == "r":
        e = ease(age / 0.2); al = e; dx = 40 * (1 - e); blur = 34 * (1 - e)
    elif st in ("a", "y"):
        e = ease(age / 0.26); al = min(1, age / 0.1); dx = -110 * (1 - e); scl = 1 + 0.25 * (1 - e); blur = 60 * (1 - e)
    elif st in ("p", "n"):
        d = 0.3 if st == "p" else 0.4
        scl = max(back(age / d), 0.01); al = min(1, age / 0.08)
    elif st == "b":
        k = min(len(t.frames) - 1, int(age / 0.035))
        img = t.frames[k][0]
        if k < len(t.frames) - 1 and int(age * 6) % 2: img = t.frames[k][0]
    if out > 0:
        al *= 1 - out; dy -= 30 * out; blur += 25 * out
    h, w = img.shape[:2]
    if scl != 1:
        nw, nh = max(1, int(w * scl)), max(1, int(h * scl))
        img = cv2.resize(img, (nw, nh)); dx -= (nw - w) / 2; dy -= (nh - h) / 2
    if blur >= 2:
        k = int(blur) | 1
        img = cv2.blur(img, (k, 1))
    if t.shadow is not None and not card:
        sh = t.shadow
        if scl != 1: sh = cv2.resize(sh, (img.shape[1], img.shape[0]))
        if blur >= 2: sh = cv2.blur(sh, (int(blur) | 1, 1))
        shi = np.zeros(sh.shape + (4,), np.float32); shi[..., 3] = sh
        blend(dst, shi, int(t.x + dx), int(t.y + dy + 3), al)
    blend(dst, img, int(t.x + dx), int(t.y + dy), al)

FULL = (0.0, 0.0, float(W), float(H)); CARD = (165.0, 520.0, 750.0, 1333.0)
def rect_of(s): return CARD if s["mode"] == "card" else FULL
def zoom_of(s, tt):
    if s["mode"] == "card": return 1.0
    p = (tt - s["start"]) / max(0.1, s["end"] - s["start"])
    return s["zoom"] * (1 + 0.035 * min(max(p, 0), 1))

def base_frame(src, tt, si):
    s = scenes[si]
    r, z = rect_of(s), zoom_of(s, tt)
    gray = s["mode"] == "bw"
    if si > 0:
        pv = scenes[si - 1]
        mode_change = (pv["mode"] == "card") != (s["mode"] == "card")
        p = (tt - s["start"]) / 0.32
        if mode_change and p < 1:
            e = ease(p); pr = rect_of(pv); pz = zoom_of(pv, s["start"])
            r = tuple(a + (b - a) * e for a, b in zip(pr, r)); z = pz + (z - pz) * e
    frame = np.full((H, W, 3), 1.0, np.float32)
    x, y, w, h = r
    cw, ch = SW / z, SH / z
    cx0, cy0 = (SW - cw) / 2, (SH - ch) / 2
    crop = src[int(cy0):int(cy0 + ch), int(cx0):int(cx0 + cw)]
    vid = cv2.resize(crop, (int(round(w)), int(round(h))), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    if gray:
        g = vid @ np.float32([0.299, 0.587, 0.114])
        q = min(1, (tt - s["start"]) / 0.2)
        vid = vid * (1 - q) + g[..., None] * q
    xi, yi = int(round(x)), int(round(y))
    hh, ww = vid.shape[:2]
    x0, y0 = max(xi, 0), max(yi, 0); x1, y1 = min(xi + ww, W), min(yi + hh, H)
    frame[y0:y1, x0:x1] = vid[y0 - yi:y1 - yi, x0 - xi:x1 - xi]
    return frame, s["mode"] == "card" and r == CARD

dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "clean.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "video_only.mp4"], stdin=subprocess.PIPE)
fi = 0; si = 0
LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
while True:
    buf = dec.stdout.read(SW * SH * 3)
    if len(buf) < SW * SH * 3: break
    tt = fi / FPS
    if tt > LIMIT: break
    while si + 1 < len(scenes) and tt >= scenes[si + 1]["start"]: si += 1
    src = np.frombuffer(buf, np.uint8).reshape(SH, SW, 3)
    frame, _ = base_frame(src, tt, si)
    card = scenes[si]["mode"] == "card"
    for t in scenes[si]["toks"]: draw_tok(frame, t, tt, scenes[si], card)
    enc.stdin.write((np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    fi += 1
enc.stdin.close(); enc.wait(); dec.kill()

# ---------- sound effects ----------
SR = 48000
rng = np.random.default_rng(7)
def lp_sweep(x, cut):
    y = np.zeros_like(x); s = 0.0
    a = 1 - np.exp(-2 * np.pi * cut / SR)
    for i in range(len(x)):
        s += a[i] * (x[i] - s); y[i] = s
    return y
def whoosh(d=0.5, lo=300, hi=5000, peak=0.6):
    n = int(d * SR); t = np.linspace(0, 1, n)
    env = np.where(t < peak, (t / peak) ** 2, ((1 - t) / (1 - peak)) ** 1.5)
    cut = lo + (hi - lo) * np.sin(np.pi * np.clip(t / (peak * 2), 0, 1)) ** 2
    x = rng.standard_normal(n)
    y = lp_sweep(x, cut) - lp_sweep(x, cut * 0.25)
    y = y * env; return y / np.abs(y).max()
def pop():
    n = int(0.09 * SR); t = np.arange(n) / SR
    f = 260 + 700 * np.exp(-t * 60); ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t * 45); return y / np.abs(y).max()
def click():
    n = int(0.012 * SR); x = rng.standard_normal(n); x = np.diff(x, prepend=0)
    y = x * np.exp(-np.arange(n) / SR * 500); return y / np.abs(y).max()
def ding():
    n = int(1.4 * SR); t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * k) for f, a, k in [(1318.5, 1, 3), (1975.5, .5, 4), (2637, .3, 6), (659.25, .4, 2.5)])
    y *= np.minimum(1, t / 0.004); return y / np.abs(y).max()

trk = np.zeros(int((DUR + 2) * SR))
def put(sig, t, g):
    i = int(max(0, t) * SR); seg = trk[i:i + len(sig)]; seg += sig[:len(seg)] * g
W_BIG, W_SM, SWISH, POP, CLK, DING = whoosh(0.55), whoosh(0.3, 600, 7000, 0.5), whoosh(0.22, 1500, 9000, 0.4), pop(), click(), ding()
for i, s in enumerate(scenes):
    if i > 0:
        pv = scenes[i - 1]
        if (pv["mode"] == "card") != (s["mode"] == "card") or s["mode"] == "bw" or pv["mode"] == "bw":
            put(W_BIG, s["start"] - 0.33, 0.30)
        else:
            put(W_SM, s["start"] - 0.15, 0.12)
    for t in s["toks"]:
        if t.style in ("a", "y"): put(SWISH, t.t0 - 0.06, 0.12)
        elif t.style == "p": put(POP, t.t0, 0.28)
        elif t.style == "n": put(POP, t.t0, 0.3); put(DING, t.t0 + 0.05, 0.16)
        elif t.style == "b":
            for k in range(len(t.txt)):
                if t.txt[k] != " " and t.t0 + k * 0.035 < s["end"]: put(CLK, t.t0 + k * 0.035, 0.05 * rng.uniform(0.6, 1))
trk = trk[:int(DUR * SR)]
with wave.open("sfx.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(trk, -1, 1) * 32767).astype(np.int16).tobytes())
print("scenes", [(round(s["start"], 2), s["mode"]) for s in scenes])
