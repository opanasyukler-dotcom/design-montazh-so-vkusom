import subprocess, json, wave, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

W, H, FPS, SR = 1080, 1920, 30, 48000
FONTS = "../fonts2/M%d.ttf"
POWDER = (228, 210, 210); NAVY = (16, 46, 70); MAUVE = (103, 96, 104); WHITE = (255, 255, 255)

# ---------------- blocks: audio segments (src, a, b) / holds ----------------
B = [
 dict(name="after1", aud=[("surgeon", 15.35, 21.10)]),
 dict(name="before", aud=[("interview", 43.30, 45.57)], hold=1.1),
 dict(name="after_photo", aud=[("surgeon", 21.15, 22.45), ("surgeon", 24.05, 29.05)]),
 dict(name="edema", aud=[("surgeon", 50.20, 55.45)], hold=0.3),
 dict(name="months", aud=[("sil", 0, 0.30), ("surgeon", 46.80, 49.89), ("sil", 0, 0.32), ("surgeon", 80.82, 84.98), ("sil", 0, 0.28), ("surgeon", 85.02, 87.30)], hold=0.17),
 dict(name="feel", aud=[("interview", 0.0, 2.56), ("interview", 4.35, 7.85)]),
 dict(name="final", aud=[("interview", 45.70, 50.30)], hold=1.6),
]
AUD = {}
for n in ("surgeon", "interview"):
    with wave.open(f"a_{n}.wav") as w: AUD[n] = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
WJ = {n: json.load(open(f"{n}.json")) for n in ("surgeon", "interview")}
FIX = {"объемный": "объёмный", "нее": "неё", "отечная": "отёчная", "объем": "объём", "объема": "объёма", "Естественно": "естественно", "еще": "ещё", "все": "всё", "Ане": "Анне", "Романе": "Романовне", "Чем": "о чём"}
REPL_42 = ["мы", "нашей", "сегодняшней", "пациентке", "показали", "как", "раз", "ту,", "чтобы", "она", "не", "волновалась"]

voice = []; t = 0.0; SUBS = []
for b in B:
    b["t0"] = t
    for (src, a, e) in b["aud"]:
        if src == "sil":
            voice.append(np.zeros(int(e * SR), np.float32)); t += e; continue
        seg = AUD[src][int(a * SR):int(e * SR)].copy(); fd = int(0.012 * SR)
        seg[:fd] *= np.linspace(0, 1, fd); fo = int(0.07 * SR); seg[-fo:] *= np.linspace(1, 0, fo) ** 1.5
        voice.append(seg)
        ws = [w for w in WJ[src] if w[0] >= a - 0.05 and w[0] < e - 0.05]
        if src == "surgeon" and abs(a - 42.75) < 0.01:
            ws = [[w[0], w[1], REPL_42[i]] for i, w in enumerate(ws)]
        for w in ws:
            txt = w[2].strip(",.?!:")
            txt = FIX.get(txt, txt)
            SUBS.append((t + max(0, w[0] - a), txt))
        t += e - a
    if b.get("hold"):
        voice.append(np.zeros(int(b["hold"] * SR), np.float32)); t += b["hold"]
    b["t1"] = t
DUR = t
voice = np.concatenate(voice)
with wave.open("voice_cat3.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(voice, -1, 1) * 32767).astype(np.int16).tobytes())
SUBS.sort()
def bt(name): return next(b for b in B if b["name"] == name)
print("dur", DUR, [(b["name"], round(b["t0"], 2), round(b["t1"], 2)) for b in B])

# ---------------- drawing helpers ----------------
def font(w, s): return ImageFont.truetype(FONTS % w, s)
def premul(a): a = a.copy(); a[..., :3] *= a[..., 3:4]; return a
def to_np(im): return premul(np.asarray(im).astype(np.float32) / 255)
def pill(txt, size=44, bg=NAVY, fg=POWDER, weight=600):
    f = font(weight, size); asc, desc = f.getmetrics(); tw = int(f.getlength(txt))
    ph, pw = asc + desc + 18, tw + 52
    im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, pw - 1, ph - 1), radius=ph // 2, fill=bg + (255,))
    d.text((pw // 2, ph // 2 + 2), txt, font=f, fill=fg + (255,), anchor="mm")
    return to_np(im)
def headline_lines(lines, size=86, accent=()):
    """list of per-line images: navy bar + white/powder Black text"""
    out = []
    while max(font(900, size).getlength(ln) for ln in lines) > 930: size -= 2
    for ln in lines:
        f = font(900, size); asc, desc = f.getmetrics()
        words = ln.split(" "); sp = f.getlength(" ")
        tw = int(sum(f.getlength(x) for x in words) + sp * (len(words) - 1))
        ph, pw = asc + desc + 14, tw + 48
        im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, pw - 1, ph - 1), radius=18, fill=NAVY + (255,))
        x = 24
        for wd in words:
            d.text((x, 7 + asc), wd, font=f, fill=(POWDER if wd.strip("?«»,") in accent else WHITE) + (255,), anchor="ls")
            x += f.getlength(wd) + sp
        out.append(to_np(im))
    return out
def hl_h(imgs, gap=10): return sum(i.shape[0] + gap for i in imgs)
def blend(dst, img, x, y, al):
    if al <= 0.003: return
    h, w = img.shape[:2]; x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]; d = dst[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4] * al; d += s[..., :3] * al
def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def back(p): p = min(max(p, 0), 1); c = 1.9; return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2
def shadow_of(img, s=10, a=0.45):
    sh = cv2.GaussianBlur(img[..., 3], (0, 0), s) * a; o = np.zeros(sh.shape + (4,), np.float32); o[..., 3] = sh; return o
def pop_img(dst, img, cx, cy, t, t0, t1=1e9, sh=True):
    if t < t0 or t > t1 + 0.25: return
    s = max(back((t - t0) / 0.3), 0.02); al = min(1, (t - t0) / 0.08) * (1 - min(1, max(0, (t - t1) / 0.25)))
    im = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s)))) if abs(s - 1) > 0.01 else img
    if sh: blend(dst, shadow_of(im), cx - im.shape[1] / 2, cy - im.shape[0] / 2 + 8, al)
    blend(dst, im, cx - im.shape[1] / 2, cy - im.shape[0] / 2, al)
def slide_lines(dst, imgs, cx, y, t, t0, t1=1e9, gap=10, stagger=0.12):
    for i, im in enumerate(imgs):
        a = t - t0 - i * stagger
        if a < 0 or t > t1 + 0.25: y += im.shape[0] + gap; continue
        e = ease(a / 0.3); al = min(1, a / 0.15) * (1 - min(1, max(0, (t - t1) / 0.25)))
        k = int(30 * (1 - e)) | 1
        im2 = cv2.blur(im, (1, k)) if k > 2 else im
        blend(dst, shadow_of(im2, 12, 0.5), cx - im.shape[1] / 2, y + 50 * (1 - e) + 8, al)
        blend(dst, im2, cx - im.shape[1] / 2, y + 50 * (1 - e), al)
        y += im.shape[0] + gap

# vector strokes (arrows / arcs / glows)
def partial(pts, p):
    pts = np.asarray(pts, np.float64)
    if p >= 1: return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
    L_ = cum[-1] * max(p, 0); k = np.searchsorted(cum, L_)
    if k == 0: return pts[:1]
    q = pts[k - 1] + (pts[k] - pts[k - 1]) * ((L_ - cum[k - 1]) / max(seg[k - 1], 1e-6))
    return np.vstack([pts[:k], q])
def bez(a, b, bend=0.25, n=40):
    a, b = np.asarray(a, float), np.asarray(b, float); m = (a + b) / 2; d = b - a
    c = m + np.array([-d[1], d[0]]) * bend; t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b
def stroke(frame, pts, al, col=WHITE, th=7, head=False, dash=False):
    if al <= 0 or len(pts) < 2: return
    mk = np.zeros((H, W), np.uint8)
    if dash:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
        for d0 in np.arange(0, cum[-1], 28):
            cv2.circle(mk, (int(np.interp(d0, cum, pts[:, 0])), int(np.interp(d0, cum, pts[:, 1]))), th // 2 + 1, 255, -1, cv2.LINE_AA)
    else:
        cv2.polylines(mk, [np.round(pts).astype(np.int32)], False, 255, th, cv2.LINE_AA)
    if head and len(pts) > 3:
        e = pts[-1]; d = pts[-1] - pts[-4]; d /= max(np.linalg.norm(d), 1e-6); n_ = np.array([-d[1], d[0]]); hl = 32
        cv2.fillPoly(mk, [np.array([e + d * 6, e - d * hl + n_ * hl * 0.55, e - d * hl - n_ * hl * 0.55]).astype(np.int32)], 255, cv2.LINE_AA)
    sm = cv2.resize(mk, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    sh = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 5), (W, H)).astype(np.float32)[..., None] / 255 * 0.5 * al
    frame *= 1 - sh
    a = mk.astype(np.float32)[..., None] / 255 * al
    frame *= 1 - a; frame += np.float32(col) / 255 * a
def glow(frame, c, ax, ang, al, col=WHITE):
    if al <= 0: return
    m = np.zeros((H // 2, W // 2), np.uint8)
    cv2.ellipse(m, (int(c[0] / 2), int(c[1] / 2)), (int(ax[0] / 2), int(ax[1] / 2)), ang, 0, 360, 255, -1, cv2.LINE_AA)
    a = cv2.resize(cv2.GaussianBlur(m, (0, 0), max(ax) / 5).astype(np.float32) / 255, (W, H))[..., None] * al
    frame *= 1 - a; frame += np.float32(col) / 255 * a
def arc_pts(c, rx, ry, a0, a1, n=50):
    t = np.radians(np.linspace(a0, a1, n)); return np.stack([c[0] + rx * np.cos(t), c[1] + ry * np.sin(t)], 1)
def env(t, t0, t1, dur):
    if t < t0 or t > t1 + 0.3: return 0, 0
    return ease((t - t0) / dur), 1 - min(1, max(0, (t - t1) / 0.3))

# ---------------- sources ----------------
def photo(n):
    im = ImageOps.exif_transpose(Image.open("dl/" + n)).convert("RGB"); w, h = im.size; s = max(W / w, H / h)
    im = im.resize((round(w * s), round(h * s)), Image.LANCZOS); x = (im.width - W) // 2; y = (im.height - H) // 2
    return np.asarray(im.crop((x, y, x + W, y + H))).astype(np.float32) / 255
def kenburns(img, p, z0=1.0, z1=1.08, fx=540, fy=800):
    z = z0 + (z1 - z0) * p
    M = np.float32([[z, 0, fx - z * fx], [0, z, fy - z * fy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
PH = {n: photo(n) for n in ["before_a", "before_b", "d6_b", "m1_b", "m2_a", "m2_b", "m2_c"]}
def blur_ell(img, cx, cy, rx, ry, k=24, sg=18):
    h, w = img.shape[:2]; m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), max(4, rx / 8))[..., None]
    sm = cv2.resize(img, (w // k, h // k), interpolation=cv2.INTER_AREA)
    bl = cv2.GaussianBlur(cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), sg)
    return img * (1 - m) + bl * m
PH["before_a"] = blur_ell(PH["before_a"], 640, 0, 230, 150)
PH["before_b"] = blur_ell(PH["before_b"], 545, 0, 170, 90)
PH["m1_b"] = blur_ell(PH["m1_b"], 545, 0, 190, 100)
import sys as _s; _s.path.insert(0, "../v4"); from retouch import retouch
PH["d6_b"] = retouch(PH["d6_b"])
PI = [np.load("pose_i1.npy"), np.load("pose_i2.npy"), np.load("pose_i3.npy")]
def blur_pose(img, P, f):
    a = P[min(f, len(P) - 1)]; w = abs(a[7, 0] - a[8, 0]); cx = (a[7, 0] + a[8, 0]) / 2; cy = a[0, 1] - 0.25 * w
    return blur_ell(img, cx, cy, w * 0.85, w * 1.15, 40, 30)
FB1 = np.load("fb_b1.npy"); FB4 = np.load("fb_b4.npy")
def blur_box(img, bx, pad=1.35):
    x0, y0, x1, y1 = bx; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - (y1 - y0) * 0.05
    return blur_ell(img, cx, cy, (x1 - x0) / 2 * pad, (y1 - y0) / 2 * pad)

class Vid:
    def __init__(s, segs): s.segs = list(segs); s.p = None; s.last = None
    def _open(s):
        src, a, e = s.segs.pop(0)
        s.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(a), "-to", str(e), "-i", "dl/" + src, "-vf", "fps=30,scale=1080:1920",
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        s.cut = True
    def next(s):
        s.cut = False
        while True:
            if s.p is None:
                if not s.segs: return s.last
                s._open()
            b = s.p.stdout.read(W * H * 3)
            if len(b) == W * H * 3:
                s.last = np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32) / 255; return s.last
            s.p.kill(); s.p = None
VID = {"final": [("interview", 49.0, 50.48)], "months": [("m1_a", 9.0, 12.6)], "after1": [("surgeon", 15.35, 21.10)], "edema": [("surgeon", 50.20, 55.45)], "feel": [("interview", 0.0, 2.56), ("interview", 4.35, 7.85)]}

POSE1 = np.load("pose_b1.npy")
def b1_poles(f):
    f = min(f, len(POSE1) - 1); a = POSE1[max(0, f - 2):f + 3].mean(0); l, r = a[11], a[12]; c = (l + r) / 2; w = np.linalg.norm(l - r)
    return [s + (c - s) * 0.5 + np.array([0, 0.2 * w]) for s in (r, l)], w
KF4 = np.array([[0.2, 120, 860, 270, 850], [2.5, 175, 860, 320, 850], [4.8, 140, 875, 300, 860]], float)
CR4 = (0.0, 255.0, 635.0, 1129.0)
def b4_poles(t):
    p = [np.interp(t, KF4[:, 0], KF4[:, i]) for i in range(1, 5)]
    x0, y0, cw, ch = CR4
    return [np.array([(p[0] - x0) * W / cw, (p[1] - y0) * H / ch]), np.array([(p[2] - x0) * W / cw, (p[3] - y0) * H / ch])]

# ---------------- texts (Anna Tvauri style) ----------------
FD = "/tmp/claude-0/-home-user-design-montazh-so-vkusom/b26b20a4-6ac5-50f8-8404-dcbad88c88d3/scratchpad/v4/fonts/"
BLUE = (16, 46, 70)
def osw(w, s): return ImageFont.truetype(FD + f"Oswald-{w}.ttf", s)
def mont(w, s): return ImageFont.truetype(FONTS % w, s)
class Txt:
    """block of lines; each line: (text, kind) kind in H (headline), HB (headline on blue box), S (small), SB (small bold)"""
    def __init__(s, lines, x=80, align="L", size=78, maxw=920, bar=False):
        s.items = []; s.bar = bar
        for txt, kind in lines:
            if kind in ("H", "HB"):
                sz = size
                while osw(600, sz).getlength(txt) > maxw: sz -= 2
                f = osw(600, sz)
            else:
                f = mont(500 if kind in ("SB", "SL") else 300, 42 if kind == "SL" else (36 if kind != "SS" else 30))
            asc, desc = f.getmetrics()
            s.items.append(dict(txt=txt, kind=kind, f=f, asc=asc, desc=desc, w=f.getlength(txt)))
        s.x = x; s.align = align
    def height(s): return sum((it["asc"] + it["desc"]) * (0.92 if it["kind"] in ("H", "HB") else 1.15) + (14 if it["kind"] == "HB" else 0) for it in s.items)
    def draw(s, frame, y, t, t0, t1=1e9, cps=28):
        if t < t0 or t > t1 + 0.25: return
        fade = 1 - min(1, max(0, (t - t1) / 0.25))
        el = (t - t0)
        nchar = el * cps
        im = Image.new("RGBA", (W, int(s.height()) + 60), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        yy = 10; used = 0
        for it in s.items:
            lh = (it["asc"] + it["desc"]) * (0.92 if it["kind"] in ("H", "HB") else 1.15)
            n = int(min(len(it["txt"]), max(0, nchar - used))); used += len(it["txt"])
            if n <= 0: break
            txt = it["txt"][:n]
            x = s.x if s.align == "L" else (W - it["w"]) / 2
            if it["kind"] == "HB":
                wv = it["f"].getlength(txt)
                d.rectangle((x - 14, yy + it["asc"] * 0.12, x + wv + 14, yy + lh + 14), fill=BLUE + (255,))
                d.text((x, yy + 7 + it["asc"]), txt, font=it["f"], fill=(255, 255, 255, 255), anchor="ls"); yy += lh + 14
            else:
                d.text((x, yy + it["asc"]), txt, font=it["f"], fill=(255, 255, 255, 255), anchor="ls"); yy += lh
        a = to_np(im)
        if s.bar:
            hb = min(1, el / 0.35) * yy
            cv2.rectangle(a, (s.x - 30, 14), (s.x - 25, int(10 + hb)), (1, 1, 1, 1), -1)
        blend(frame, shadow_of(a, 10, 0.55), 0, y + 4, fade)
        blend(frame, a, 0, y, fade)
def osw_tag(txt, size=52, bg=BLUE):
    f = osw(600, size); asc, desc = f.getmetrics(); w = int(f.getlength(txt))
    im = Image.new("RGBA", (w + 32, asc + desc + 12), bg + (255,)); ImageDraw.Draw(im).text((16, 6 + asc), txt, font=f, fill=(255, 255, 255, 255), anchor="ls")
    return to_np(im)

T1 = Txt([("«А ВДРУГ ГРУДЬ ПОЛУЧИЛАСЬ", "H"), ("СЛИШКОМ БОЛЬШАЯ?»", "H"), ("6 дней после операции", "S")], size=50)
LBL_TOP = osw_tag("ОБЪЁМНЫЙ ВЕРХ", 30)
T2 = Txt([("ВОТ КАКОЙ БЫЛ ЗАПРОС", "H")], size=48)
REQ = [Txt([(s, "SB")]) for s in ["— круглая", "— наполненная", "— но естественная"]]
TAG_BEFORE = osw_tag("ДО", 32, (40, 40, 40)); TAG_AFTER = osw_tag("ПОСЛЕ", 32)
T3 = Txt([("СЕЙЧАС ЭТО ЕЩЁ", "H"), ("НЕ ФИНАЛЬНАЯ ФОРМА", "HB"), ("6 дней после операции", "S")], size=50, bar=True)
LBL_SLOPE = Txt([("верхний склон будет ровным", "SB")], align="C")
T4a = Txt([("ОТЁК", "HB")], size=90)
T4b = Txt([("ТКАНИ ЕЩЁ", "H"), ("НЕ АДАПТИРОВАЛИСЬ", "H"), ("поэтому верх груди сейчас", "S"), ("выглядит объёмнее", "S")], size=48, bar=True)
T5 = Txt([("ЧЕРЕЗ НЕСКОЛЬКО МЕСЯЦЕВ", "H"), ("ГРУДЬ БУДЕТ ВЫГЛЯДЕТЬ", "H"), ("СОВСЕМ ИНАЧЕ", "HB")], size=46, align="C")
CAP_M2 = Txt([("2 месяца после операции", "SL")], align="C"); CAP_M1 = Txt([("месяц после операции", "SL")], align="C")
T6 = Txt([("А САМОЧУВСТВИЕ?", "H")], size=54)
f_ = osw(600, 46); qt = "«КАК НА КУРОРТЕ»"; tw = int(f_.getlength(qt)); asc, desc = f_.getmetrics()
em = Image.open("emoji.png").crop((19, 17, 136, 129)).resize((56, 54), Image.LANCZOS)
qi = Image.new("RGBA", (tw + 31, asc + desc + 20), (0, 0, 0, 0)); d_ = ImageDraw.Draw(qi)
d_.rectangle((0, 0, tw + 30, qi.height - 1), fill=BLUE + (255,)); d_.text((15, 10 + asc), qt, font=f_, fill=(255, 255, 255, 255), anchor="ls")
QUOTE = to_np(qi)
T7a = Txt([("«С НЕЙ ВСЁ СОШЛОСЬ»", "HB")], size=40, align="C")
T7 = Txt([("«ПОЧЕМУ Я НЕ СДЕЛАЛА", "H"), ("ЭТО РАНЬШЕ?»", "H")], size=60, align="C")
def word_img(txt, col=(255, 255, 255)):
    f = osw(600, 66); asc, desc = f.getmetrics(); w = int(f.getlength(txt))
    im = Image.new("RGBA", (w + 28, asc + desc + 14), BLUE + (255,)); ImageDraw.Draw(im).text((14, 7 + asc), txt, font=f, fill=col + (255,), anchor="ls")
    return to_np(im)
WORDS7 = []
for li, (line, times) in enumerate([(["«ПОЧЕМУ", "Я", "НЕ", "СДЕЛАЛА"], [3.4, 3.5, 3.6, 3.7]), (["ЭТО", "РАНЬШЕ?»"], [4.0, 4.1])]):
    ims = [word_img(w_, POWDER if w_.startswith("РАНЬШЕ") else (255, 255, 255)) for w_ in line]
    tot = sum(i.shape[1] for i in ims) + 10 * (len(ims) - 1); x = (W - tot) / 2
    for i_, (im_, wt) in enumerate(zip(ims, times)):
        WORDS7.append((im_, x, 1150 + li * 100, wt, False)); x += im_.shape[1] + 10
LOGO = np.asarray(Image.open("../v4/logo.png").resize((320, int(115 * 320 / 620)), Image.LANCZOS)).astype(np.float32) / 255
LOGO = premul(LOGO)
SUBC = {}
def sub_img(w):
    if w not in SUBC:
        f = mont(500, 44); asc, desc = f.getmetrics(); tw = int(f.getlength(w.lower()))
        im = Image.new("RGBA", (tw + 20, asc + desc + 10), (0, 0, 0, 0)); ImageDraw.Draw(im).text((10, 5 + asc), w.lower(), font=f, fill=(255, 255, 255, 255), anchor="ls")
        a = to_np(im); SUBC[w] = (a, shadow_of(a, 6, 0.7))
    return SUBC[w]
def subtitles(frame, t):
    for i, (t0, w) in enumerate(SUBS):
        t1 = SUBS[i + 1][0] if i + 1 < len(SUBS) else DUR
        t1 = min(t1, t0 + 1.0)
        if t0 <= t < t1:
            im, sh = sub_img(w); al = min(1, (t - t0) / 0.06)
            blend(frame, sh, 540 - im.shape[1] / 2, 1500 - im.shape[0] / 2 + 3, al)
            blend(frame, im, 540 - im.shape[1] / 2, 1500 - im.shape[0] / 2, al)
def grade(fr):
    g = fr @ np.float32([0.299, 0.587, 0.114])
    fr = fr * 0.82 + g[..., None] * 0.18
    fr = np.clip((fr - 0.5) * 1.08 + 0.47, 0, 1)
    return fr * VIG
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.38 * (((xx - 540) / 760) ** 2 + ((yy - 960) / 1250) ** 2))[..., None].clip(0.55, 1)
END = False

# ---------------- per-block rendering ----------------
def render_block(b, lt, fi, vid):
    global END
    n = b["name"]; d = b["t1"] - b["t0"]; p = lt / d; END = False
    if n == "after1":
        fr = grade(retouch(blur_box(vid.next(), FB1[min(fi, len(FB1) - 1)])))
        T1.draw(fr, 1250, lt, 0.1)
        pts, w = b1_poles(fi)
        q, al = env(lt, 2.05, 99, 0.45)
        for c in pts: stroke(fr, partial(arc_pts(c + [0, 0.05 * w], 0.2 * w, 0.12 * w, 200, 340), q), al, WHITE, 6)
        q, al = env(lt, 2.35, 99, 0.4)
        if q > 0:
            for i, c in enumerate(pts):
                st = (440 if i == 0 else 640, 395); stroke(fr, partial(bez(st, c + [0, -0.1 * w], -0.2 if i == 0 else 0.2), q), al, WHITE, 4, head=q > 0.95)
        pop_img(fr, LBL_TOP, 540, 360, lt, 2.2, sh=False)
        return fr
    if n == "before":
        img = PH["before_b"] if lt < 1.9 else PH["before_a"]
        fr = grade(kenburns(img, p, 1.0, 1.1, 540, 650))
        T2.draw(fr, 1250, lt, 0.05, cps=40)
        blend(fr, TAG_BEFORE, 80, 330, 1)
        for i, tx in enumerate(REQ): tx.draw(fr, 1320 + i * 54, lt, 0.55 + i * 0.55, cps=40)
        return fr
    if n == "after_photo":
        fr = grade(kenburns(PH["d6_b"], p, 1.04, 1.12, 540, 600))
        z = 1.04 + 0.08 * p
        def mp_(x, y): return np.array([540 + (x - 540) * z, 600 + (y - 600) * z])
        T3.draw(fr, 1250, lt, 0.2)
        blend(fr, TAG_AFTER, 80, 1180, 1)
        q, al = env(lt, 4.9, 99, 0.6)
        for (x0, y0, x1, y1, x2, y2) in [(120, 440, 238, 372, 380, 420), (600, 425, 745, 366, 880, 425)]:
            pts = np.array([mp_(x0, y0), mp_(x1, y1), mp_(x2, y2)])
            tt = np.linspace(0, 1, 40)[:, None]; cur = (1 - tt) ** 2 * pts[0] + 2 * (1 - tt) * tt * (2 * pts[1] - (pts[0] + pts[2]) / 2) + tt ** 2 * pts[2]
            stroke(fr, partial(cur, q), al, WHITE, 6, dash=True)
        LBL_SLOPE.draw(fr, 250, lt, 4.95)
        return fr
    if n == "edema":
        src = blur_box(vid.next(), FB4[min(fi, len(FB4) - 1)], 1.15); x0, y0, cw, ch = CR4
        fr = retouch(cv2.resize(src[int(y0):int(y0 + ch), int(x0):int(x0 + cw)], (W, H), interpolation=cv2.INTER_CUBIC))
        fr = grade(fr)
        pls = b4_poles(lt)
        q, al = env(lt, 0.3, 99, 0.4)
        if q > 0:
            pulse = 0.7 + 0.3 * np.sin((lt - 0.3) * 5)
            for i, c in enumerate(pls): glow(fr, c + [0, 10], (150 * q, 85 * q), -10 if i == 0 else 10, 0.3 * al * pulse, (235, 235, 240))
            for i, c in enumerate(pls): stroke(fr, partial(arc_pts(c + [0, 25], 150, 85, 190, 350), q), al, WHITE, 5, dash=True)
        T4a.draw(fr, 250, lt, 0.3, cps=20)
        T4b.draw(fr, 1190, lt, 1.5, cps=30)
        return fr
    if n == "months":
        if lt < 3.6:     # 1 month: video
            fr = grade(blur_ell(vid.next(), 540, 0, 210, 120))
            CAP_M1.draw(fr, 1320, lt, 0.2, cps=40)
            blend(fr, TAG_AFTER, 960 - TAG_AFTER.shape[1], 330, 1)
            return fr
        seq = [("m2_c", 3.6, 7.1), ("m2_b", 7.1, 99)]
        for (k, a, e) in seq:
            if a <= lt < e:
                pp = (lt - a) / max(0.1, min(e, d) - a)
                fr = grade(kenburns(PH[k], pp, 1.0, 1.05, 540, 960))
                blend(fr, TAG_AFTER, 960 - TAG_AFTER.shape[1], 830, 1); blend(fr, TAG_BEFORE, 960 - TAG_BEFORE.shape[1], 1790, 1)
        T5.draw(fr, 820, lt, 3.7, cps=30)
        CAP_M2.draw(fr, 1060, lt, 4.6, cps=40)
        return fr
    if n == "feel":
        fr = vid.next().copy()
        fr = blur_pose(fr, PI[0], fi) if lt < 2.56 else blur_pose(fr, PI[1], fi - 77)
        if lt >= 2.5: fr = kenburns(fr, 0, 1.1, 1.1, 540, 700)
        fr = grade(fr)
        T6.draw(fr, 1130, lt, 0.05, cps=30)
        pop_img(fr, QUOTE, 540, 1270, lt, 1.35, sh=False)
        return fr
    if n == "final":
        if lt < 3.3:   # stacked before / after
            top = PH["d6_b"][300:300 + 960]; bot = PH["before_b"][330:330 + 960]
            fr = np.concatenate([top, bot], 0).copy()
            k = ease(lt / 0.5); fr[960:] = fr[960:] * k + 0 * (1 - k)
            fr = grade(fr)
            blend(fr, TAG_AFTER, 80, 840, 1); blend(fr, TAG_BEFORE, 80, 1800 - TAG_BEFORE.shape[0], min(1, lt / 0.5))
            T7a.draw(fr, 925, lt, 0.9)
            return fr
        END = True
        k = fi - int(round(3.3 * FPS))
        fr = blur_pose(vid.next().copy(), PI[2], k)
        a = PI[2][min(max(k, 0), len(PI[2]) - 1)]
        z = 1.0 + 0.32 * ease((lt - 3.3) / 2.6)
        fr = grade(kenburns(fr, 0, z, z, a[0, 0], a[0, 1] + 150))
        for (im, x, y, wt, last) in WORDS7:
            age = lt - wt
            if age < 0: continue
            s_ = max(back(age / 0.32), 0.05); al = min(1, age / 0.08)
            im2 = cv2.resize(im, (max(1, int(im.shape[1] * s_)), max(1, int(im.shape[0] * s_))))
            blend(fr, shadow_of(im2, 12, 0.6), x + (im.shape[1] - im2.shape[1]) / 2, y + (im.shape[0] - im2.shape[0]) / 2 + 8, al)
            blend(fr, im2, x + (im.shape[1] - im2.shape[1]) / 2, y + (im.shape[0] - im2.shape[0]) / 2, al)
        return fr

dec_rng = [(b, Vid(VID[b["name"]]) if b["name"] in VID else None) for b in B]
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "video_only3.mp4"], stdin=subprocess.PIPE)
NF = int(round(DUR * FPS)); bi = 0; bstart = 0
for f in range(NF):
    t = f / FPS
    while bi + 1 < len(B) and t >= B[bi + 1]["t0"]: bi += 1; bstart = f
    b, vid = dec_rng[bi]; lt = t - b["t0"]
    fr = render_block(b, lt, f - bstart, vid)
    if bi > 0 and lt < 0.2 and not END:
        e = ease(lt / 0.2); z = 1 + 0.05 * (1 - e); fr = kenburns(fr, 0, z, z, 540, 960)
    blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
    if not END: subtitles(fr, t)
    enc.stdin.write((np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
enc.stdin.close(); enc.wait()

# ---------------- SFX ----------------
rng = np.random.default_rng(5)
def lp(x, cut):
    y = np.zeros_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * cut / SR)
    for i in range(len(x)): s += a[i] * (x[i] - s); y[i] = s
    return y
def whoosh(d=0.5, lo=300, hi=5000, pk=0.6):
    n = int(d * SR); tt = np.linspace(0, 1, n)
    e = np.where(tt < pk, (tt / pk) ** 2, ((1 - tt) / (1 - pk)) ** 1.5)
    cut = lo + (hi - lo) * np.sin(np.pi * np.clip(tt / (pk * 2), 0, 1)) ** 2
    x = rng.standard_normal(n); y = (lp(x, cut) - lp(x, cut * 0.25)) * e; return y / np.abs(y).max()
def pop():
    n = int(0.09 * SR); tt = np.arange(n) / SR; y = np.sin(2 * np.pi * np.cumsum(260 + 700 * np.exp(-tt * 60)) / SR) * np.exp(-tt * 45); return y / np.abs(y).max()
def ding():
    n = int(1.4 * SR); tt = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * fq * tt) * np.exp(-tt * k) for fq, a, k in [(1318.5, 1, 3), (1975.5, .5, 4), (2637, .3, 6), (659.25, .4, 2.5)])
    return y * np.minimum(1, tt / 0.004) / np.abs(y).max()
def thump():
    n = int(0.35 * SR); tt = np.arange(n) / SR; y = np.sin(2 * np.pi * np.cumsum(55 + 90 * np.exp(-tt * 25)) / SR) * np.exp(-tt * 9); return y / np.abs(y).max()
trk = np.zeros(int((DUR + 2) * SR))
def put(s, tt, g): i = int(max(0, tt) * SR); seg = trk[i:i + len(s)]; seg += s[:len(seg)] * g
WB, SWS, PP, DG, TH = whoosh(0.6), whoosh(0.25, 1500, 8000, 0.4), pop(), ding(), thump()
for b in B[1:]: put(WB, b["t0"] - 0.35, 0.28)
def at(name, lt): return bt(name)["t0"] + lt
CLK = (lambda n: (lambda x: x / np.abs(x).max())(np.diff(rng.standard_normal(n), prepend=0) * np.exp(-np.arange(n) / SR * 500)))(int(0.012 * SR))
def typing(t0, nch, cps, g=0.05):
    for k in range(nch): put(CLK, t0 + k / cps, g * rng.uniform(0.6, 1))
typing(at("after1", 0.1), 42, 28); typing(at("before", 0.05), 20, 40); typing(at("after_photo", 0.2), 30, 28)
typing(at("edema", 1.5), 25, 30); typing(at("months", 3.7), 50, 30, 0.04); typing(at("feel", 0.05), 15, 30); [put(PP, at("final", wt), 0.2) for wt in (3.4, 3.5, 3.6, 3.7, 4.0, 4.1)]
for nm, lt in [("after1", 2.2), ("feel", 1.35)]: put(PP, at(nm, lt), 0.22)
put(TH, at("after1", 0.1), 0.3); put(TH, at("edema", 0.3), 0.45); put(DG, at("final", 4.15), 0.14); put(WB, at("final", 3.0), 0.2)
trk = trk[:int(DUR * SR)]
with wave.open("sfx3.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(trk, -1, 1) * 32767).astype(np.int16).tobytes())
