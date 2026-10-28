"""Dynamic vertical vlog «День с 4-месячным малышом» — beat-cut edit, brand palette, kinetic captions, SFX + synth music.
Work dir: src.mov (3840x2160 landscape), fonts (FONT_*).  Steps:
  python3 vlog_day.py base            -> clips/*.mp4 + base.mp4  (cropped/graded vertical 2160x3840@50, cut to the beat)
  python3 vlog_day.py render f0 f1 o  -> overlay render of output frames [f0,f1)
  python3 vlog_day.py audio           -> music.wav, sfx.wav, nat.wav, mix.wav
  TEST=1.2,5 python3 vlog_day.py test -> test_*.png previews
All layout numbers are 1080x1920 design units; K scales to output.
"""
import os, subprocess, sys, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, K = 1080, 1920, 2
OW, OH = W * K, H * K
FPS, BPM = 50, 120
BEAT = 60 / BPM
FONT_M, FONT_S = "Montserrat-VariableFont_wght.ttf", "BiroScriptUSPlus-Regular.ttf"
TEST = [float(x) for x in os.environ.get("TEST", "").split(",") if x]

def hx(s): return np.float32([int(s[i:i + 2], 16) for i in (1, 3, 5)]) / 255
INK, COCOA, ROSE, MIST, BLUSH = hx("#2F0600"), hx("#825D4D"), hx("#C5A29C"), hx("#C5B0AD"), hx("#F7DFDD")
WHITE = np.float32([1, 1, 1])

# (src_start, beats, speed, mode, crop_center_x_fraction)  mode: c = vertical crop, f = framed landscape
EDIT = [
    (0.0, 4, 0.9, "f", .5),
    (2.0, 2, 1, "c", .60), (5.0, 2, 1, "c", .60), (7.2, 2, 1, "c", .62), (9.0, 1, 1, "c", .40),
    (14.3, 3, 1, "c", .42), (19.0, 1, 1, "c", .38),
    (24.0, 2, 1, "c", .50), (26.0, 2, 1, "c", .40), (28.0, 4, 1.4, "f", .5),
    (34.0, 3, 1.3, "c", .40), (39.6, 2, 0.9, "c", .35), (41.5, 2, 1, "f", .5),
    (43.4, 2, 0.9, "c", .65), (45.3, 2, 0.9, "c", .60),
    (48.0, 2, 1, "c", .40), (54.6, 2, 1, "c", .42), (61.0, 2, 1, "c", .45), (66.6, 2, 1, "c", .42),
    (85.0, 1, 1, "c", .40), (99.0, 2, 1, "c", .42),
    (105.3, 2, 1, "c", .60), (107.25, 2, 0.9, "c", .30), (108.4, 3, 1.5, "c", .45), (112.0, 2, 2.0, "c", .50),
    (116.88, 2, 1, "c", .55), (118.2, 2, 1, "c", .45), (120.1, 2, 0.9, "c", .42), (122.7, 6, 0.65, "c", .50),
]
STARTS = np.cumsum([0] + [b for _, b, *_ in EDIT]) * BEAT     # output start time of each clip
DUR = STARTS[-1]
NFR = int(round(DUR * FPS))
def clip_at(t): return int(np.searchsorted(STARTS, t, side="right") - 1)
def T(i, beat=0.0): return STARTS[i] + beat * BEAT                 # time of clip i (+beats)

GRADE = "eq=contrast=1.07:saturation=1.07:gamma=0.98,colorbalance=rs=0.035:bs=-0.03:rm=0.02:bm=-0.02:rh=0.01:bh=-0.01"

def build_base():
    os.makedirs("clips", exist_ok=True)
    lst = []
    for i, (s, beats, sp, mode, cx) in enumerate(EDIT):
        n = int(round(beats * BEAT * FPS)); dsrc = beats * BEAT * sp + 0.2
        cw = 1215; x0 = int(np.clip(cx * 3840 - cw / 2, 0, 3840 - cw))
        if mode == "c":
            vf = f"crop={cw}:2160:{x0}:0,scale={OW}:{OH}:flags=lanczos,{GRADE},setpts=(PTS-STARTPTS)/{sp},fps={FPS}"
            args = ["-vf", vf]
        else:   # landscape framed over its own blurred, dimmed copy
            fc = (f"[0:v]{GRADE},setpts=(PTS-STARTPTS)/{sp},fps={FPS},split[a][b];"
                  f"[a]crop={cw}:2160:{(3840 - cw) // 2}:0,scale={OW // 4}:{OH // 4},gblur=sigma=14,scale={OW}:{OH},eq=brightness=-0.10:saturation=0.8[bg];"
                  f"[b]scale={OW - 120}:-2:flags=lanczos,pad=iw+24:ih+24:12:12:color=0xF7DFDD[fg];"
                  f"[bg][fg]overlay=(W-w)/2:{int(1060 * K)}-h/2")
            args = ["-filter_complex", fc]
        out = f"clips/c{i:02d}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.3f}", "-t", f"{dsrc:.3f}", "-i", "src.mov", "-an", *args,
                        "-frames:v", str(n), "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv420p", out], check=True)
        lst.append(f"file 'c{i:02d}.mp4'")
        print("clip", i, n, flush=True)
    open("clips/list.txt", "w").write("\n".join(lst))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "clips/list.txt", "-c", "copy", "base.mp4"], check=True)

# ---------------------------------------------------------------- drawing helpers
def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def ease_io(p): p = min(max(p, 0), 1); return p * p * (3 - 2 * p)
def back(p, c=1.7): p = min(max(p, 0), 1); return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2
def P(v): return int(round(v * K))

def blend(dst, img, x, y, alpha=1.0):
    if alpha <= 0.003: return
    h, w = img.shape[:2]; x, y = P(x), P(y)
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0: return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]; d = dst[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4] * alpha; d += s[..., :3] * alpha

_fc = {}
def font(kind, size, wght=400):
    k = (kind, size, wght)
    if k not in _fc:
        f = ImageFont.truetype(FONT_M if kind == "m" else FONT_S, size)
        if kind == "m": f.set_variation_by_axes([wght])
        _fc[k] = f
    return _fc[k]

def text_img(txt, f, col, thicken=False, shadow=True):
    pad = 14 * K; asc, desc = f.getmetrics(); w = int(f.getlength(txt)) + 2 * pad + 30 * K
    im = Image.new("L", (w, asc + desc + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad, pad + asc), txt, font=f, fill=255, anchor="ls")
    a = np.asarray(im).astype(np.float32) / 255
    if thicken: a = cv2.dilate(a, np.ones((3, 3), np.uint8))
    out = np.zeros(a.shape + (4,), np.float32); out[..., :3] = col * a[..., None]; out[..., 3] = a
    sh = None
    if shadow:
        s_ = np.minimum(1, cv2.GaussianBlur(a, (0, 0), 6 * K) * 0.8)
        sh = np.zeros_like(out); sh[..., :3] = INK * s_[..., None] * 0.6; sh[..., 3] = s_ * 0.6
    return out, sh, (pad + asc) / K, pad / K, f.getlength(txt) / K

class Txt:
    """one line of text. kind: m (Montserrat) / s (Biro script, wipe-in). x anchor per align. y = baseline (design)"""
    def __init__(self, txt, kind, size, col, x, y, t0, t1, align="c", wght=700, glow=None, anim=None):
        f = font("m", size * K, wght) if kind == "m" else font("s", size * K)
        self.img, self.sh, base, pad, adv = text_img(txt, f, col, thicken=(kind == "s"))
        x0 = x - (adv / 2 if align == "c" else adv if align == "r" else 0)
        self.x, self.y = x0 - pad, y - base
        self.t0, self.t1, self.kind, self.anim = t0, t1, kind, anim or ("wipe" if kind == "s" else "pop")
        self.glow = None
        if glow is not None:
            g = cv2.GaussianBlur(self.img[..., 3], (0, 0), 10 * K)
            self.glow = np.zeros_like(self.img); self.glow[..., :3] = glow * g[..., None]; self.glow[..., 3] = g; self.glow *= 0.9
        self.bbox = (x0, y - (base - pad), x0 + adv, y)
    def draw(self, dst, tt):
        if tt < self.t0 or tt > self.t1 + 0.25: return
        age = tt - self.t0; out = ease((tt - self.t1) / 0.22) if tt > self.t1 else 0
        img, sh, glow = self.img, self.sh, self.glow; dx = dy = 0; al = 1 - out
        if self.anim == "wipe":
            p = ease_io(age / 0.5)
            if p < 1:
                ww = img.shape[1]; ramp = np.clip((np.arange(ww) - p * ww * 1.15) / (-0.15 * ww), 0, 1)[None, :, None].astype(np.float32)
                img = img * ramp; sh = sh * ramp if sh is not None else None
        elif self.anim == "pop":
            sc = 0.6 + 0.4 * back(age / 0.32); al *= min(1, age / 0.1)
            if abs(sc - 1) > 0.01:
                hh, ww = img.shape[:2]; nw, nh = max(2, int(ww * sc)), max(2, int(hh * sc))
                img = cv2.resize(img, (nw, nh)); dx, dy = (ww - nw) / 2 / K, (hh - nh) / 2 / K
                sh = cv2.resize(sh, (nw, nh)) if sh is not None else None
                glow = cv2.resize(glow, (nw, nh)) if glow is not None else None
        elif self.anim == "rise":
            e = ease(age / 0.35); dy = 40 * (1 - e); al *= min(1, age / 0.2)
            k = int(20 * K * (1 - e)) | 1
            if k > 2: img = cv2.blur(img, (1, k)); sh = cv2.blur(sh, (1, k)) if sh is not None else None
        dy -= 20 * out
        if sh is not None: blend(dst, sh, self.x + dx, self.y + dy + 4, al)
        if glow is not None: blend(dst, glow, self.x + dx, self.y + dy, al * (0.65 + 0.35 * np.sin(tt * 6) ** 2))
        blend(dst, img, self.x + dx, self.y + dy, al)

class Chip:
    """section label pill (УТРО / ДЕНЬ / ВЕЧЕР) with a small dot, slides in from the left"""
    def __init__(self, txt, t0, t1, x=70, y=250):
        f = font("m", 34 * K, 800); tw = int(f.getlength(txt)); hgt = 74 * K; pw = tw + 110 * K
        im = Image.new("RGBA", (pw, hgt), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, pw - 1, hgt - 1), radius=hgt // 2, fill=(247, 223, 221, 240))
        d.ellipse((26 * K, hgt // 2 - 9 * K, 44 * K, hgt // 2 + 9 * K), fill=(130, 93, 77, 255))
        d.text((62 * K, hgt // 2), txt, font=f, fill=(47, 6, 0, 255), anchor="lm")
        a = np.asarray(im).astype(np.float32) / 255; a[..., :3] *= a[..., 3:4]; self.img = a
        self.x, self.y, self.t0, self.t1 = x, y, t0, t1
    def draw(self, dst, tt):
        if tt < self.t0 or tt > self.t1 + 0.3: return
        e = ease((tt - self.t0) / 0.35); out = ease((tt - self.t1) / 0.25) if tt > self.t1 else 0
        blend(dst, self.img, self.x - 260 * (1 - e) - 260 * out, self.y, min(1, (tt - self.t0) / 0.15) * (1 - out))

class Counter:
    """«бутылочки: N» ticking counter"""
    def __init__(self, t0, t1, n_to, x, y):
        self.t0, self.t1, self.n_to, self.x, self.y = t0, t1, n_to, x, y
        self.digits = {}
    def num(self, n):
        if n not in self.digits:
            img, sh, base, pad, adv = text_img(str(n), font("m", 120 * K, 900), BLUSH)
            self.digits[n] = (img, sh, base, pad, adv)
        return self.digits[n]
    def ticks(self):
        return [self.t0 + 0.15 + k * (self.t1 - self.t0 - 0.6) / self.n_to for k in range(self.n_to)]
    def draw(self, dst, tt):
        if tt < self.t0 or tt > self.t1 + 0.25: return
        tk = self.ticks(); n = max(1, sum(1 for x in tk if tt >= x)); out = ease((tt - self.t1) / 0.22) if tt > self.t1 else 0
        img, sh, base, pad, adv = self.num(n)
        last = max(x for x in tk if tt >= x) if tt >= tk[0] else self.t0
        sc = 1 + 0.25 * np.exp(-((tt - last) / 0.06) ** 2)
        if abs(sc - 1) > 0.01:
            hh, ww = img.shape[:2]; img = cv2.resize(img, (int(ww * sc), int(hh * sc))); sh = cv2.resize(sh, (int(ww * sc), int(hh * sc)))
        x = self.x - img.shape[1] / K / 2; y = self.y - img.shape[0] / K / 2
        blend(dst, sh, x, y + 5, 1 - out); blend(dst, img, x, y, 1 - out)

class Check:
    """checkbox list item: box draws, then ✓ (ok) or ✗ (fail) is scribbled"""
    def __init__(self, label, ok, t0, t_mark, t1, x, y):
        self.txt = Txt(label, "s", 84, WHITE, x + 80, y, t0 + 0.1, t1, align="l")
        self.ok, self.t0, self.tm, self.t1, self.x, self.y = ok, t0, t_mark, t1, x, y
    def draw(self, dst, tt):
        if tt < self.t0 or tt > self.t1 + 0.25: return
        al = 1 - (ease((tt - self.t1) / 0.22) if tt > self.t1 else 0)
        x, y = self.x, self.y - 46; s = 52
        col = tuple(float(c) * al for c in WHITE); th = 4 * K
        p = ease_io((tt - self.t0) / 0.3)
        pts = [(x, y), (x + s, y), (x + s, y + s), (x, y + s), (x, y)]
        seg = p * 4
        for k in range(4):
            if seg <= k: break
            f = min(1, seg - k); a, b = pts[k], pts[k + 1]
            cv2.line(dst, (P(a[0]), P(a[1])), (P(a[0] + (b[0] - a[0]) * f), P(a[1] + (b[1] - a[1]) * f)), col, th, cv2.LINE_AA)
        q = ease_io((tt - self.tm) / 0.3)
        if q > 0:
            mc = tuple(float(c) * al for c in (BLUSH if self.ok else ROSE))
            if self.ok: path = [(x + 8, y + 26), (x + 22, y + 44), (x + 62, y - 10)]
            else: path = [(x - 4, y - 4), (x + s + 6, y + s + 6)]
            n = len(path) - 1; seg = q * n
            for k in range(n):
                if seg <= k: break
                f = min(1, seg - k); a, b = path[k], path[k + 1]
                cv2.line(dst, (P(a[0]), P(a[1])), (P(a[0] + (b[0] - a[0]) * f), P(a[1] + (b[1] - a[1]) * f)), mc, 7 * K, cv2.LINE_AA)
            if not self.ok and q >= 1:
                q2 = ease_io((tt - self.tm - 0.25) / 0.25)
                a, b = (x + s + 6, y - 4), (x - 4, y + s + 6)
                if q2 > 0: cv2.line(dst, (P(a[0]), P(a[1])), (P(a[0] + (b[0] - a[0]) * q2), P(a[1] + (b[1] - a[1]) * q2)), mc, 7 * K, cv2.LINE_AA)
        self.txt.draw(dst, tt)

class Heart:
    """hand-drawn heart outline"""
    def __init__(self, t0, t1, cx, cy, s):
        u = np.linspace(0, 2 * np.pi, 160)
        self.pts = np.c_[16 * np.sin(u) ** 3, -(13 * np.cos(u) - 5 * np.cos(2 * u) - 2 * np.cos(3 * u) - np.cos(4 * u))] * s / 16 + [cx, cy]
        self.t0, self.t1 = t0, t1
    def draw(self, dst, tt):
        if tt < self.t0 or tt > self.t1 + 0.25: return
        p = ease_io((tt - self.t0) / 0.6); al = 1 - (ease((tt - self.t1) / 0.22) if tt > self.t1 else 0)
        n = max(2, int(p * len(self.pts)))
        cv2.polylines(dst, [np.int32(self.pts[:n] * K)], False, tuple(float(c) * al for c in BLUSH), 6 * K, cv2.LINE_AA)

# ---------------------------------------------------------------- overlay timeline
EL = []
def add(*e): EL.extend(e)
# hook title over framed kitchen shot
add(Txt("день с", "m", 64, WHITE, 540, 470, T(0, 0.15), T(1, 1.5), wght=600, anim="rise"),
    Txt("4-месячным", "m", 104, BLUSH, 540, 600, T(0, 0.6), T(1, 1.5), wght=900, glow=ROSE),
    Txt("малышом", "s", 160, WHITE, 540, 735, T(0, 1.4), T(1, 1.5)))
add(Chip("УТРО", T(1, 0), T(6, 0.9)))
add(Txt("утро начинается", "s", 102, WHITE, 540, 1180, T(1, 0.2), T(3, 1.8)),
    Txt("с кофе", "s", 122, BLUSH, 600, 1290, T(2, 0.0), T(3, 1.8)))
add(Txt("без него — никак", "s", 96, WHITE, 540, 1300, T(5, 0.4), T(6, 0.9)))
add(Chip("ДЕНЬ", T(7, 0), T(14, 1.9)))
add(Txt("а вот и главный", "s", 96, WHITE, 540, 1200, T(7, 0.2), T(8, 1.8)),
    Txt("человек", "m", 80, BLUSH, 540, 1310, T(7, 1.0), T(8, 1.8), wght=900, glow=ROSE))
add(Txt("бутылочки", "s", 102, WHITE, 540, 1060, T(10, 0.1), T(10, 2.9)))
CNT = Counter(T(10, 0.1), T(10, 2.9), 8, 540, 1240); add(CNT)
add(Txt("завтрак", "m", 64, WHITE, 540, 1180, T(11, 0.1), T(11, 1.9), wght=800),
    Txt("(наконец-то)", "s", 90, BLUSH, 560, 1280, T(11, 0.6), T(11, 1.9)))
add(Txt("работаю,", "s", 93, WHITE, 540, 560, T(12, 0.1), T(12, 1.9)),
    Txt("пока он рядом", "s", 93, WHITE, 560, 650, T(12, 0.6), T(12, 1.9)))
add(Txt("время игр", "m", 76, BLUSH, 540, 1250, T(13, 0.1), T(14, 1.9), wght=900, glow=ROSE))
# honest minute: checklist over the talking-head montage
add(Txt("план на день:", "m", 60, WHITE, 120, 330, T(15, 0.1), T(20, 1.9), align="l", wght=800, anim="rise"))
add(Check("любить его", True, T(15, 0.6), T(16, 0.3), T(20, 1.9), 120, 470))
add(Heart(T(16, 0.2), T(16, 1.9), 540, 1180, 120))
add(Check("всё успеть", False, T(17, 0.2), T(18, 0.25), T(20, 1.9), 120, 580))
add(Txt("ага, конечно", "s", 102, BLUSH, 540, 1250, T(18, 0.3), T(18, 1.9)))
add(Check("полистать ленту", True, T(19, 0.0), T(20, 0.5), T(20, 1.9), 120, 690))
add(Chip("ВЕЧЕР", T(21, 0), T(22, 1.9)))
add(Txt("уснул", "s", 130, WHITE, 540, 1250, T(22, 0.2), T(22, 1.9)))
add(Txt("5 минут", "m", 84, BLUSH, 540, 520, T(23, 0.2), T(23, 2.9), wght=900, glow=ROSE),
    Txt("для себя", "s", 110, WHITE, 560, 650, T(23, 0.8), T(23, 2.9)))
add(Txt("жидкое", "s", 110, WHITE, 540, 1150, T(25, 0.3), T(27, 1.9)),
    Txt("золото", "m", 92, BLUSH, 540, 1270, T(26, 0.0), T(27, 1.9), wght=900, glow=ROSE))
add(Txt("и так", "s", 110, WHITE, 540, 1080, T(28, 0.6), DUR + 1),
    Txt("каждый день", "m", 86, BLUSH, 540, 1200, T(28, 1.3), DUR + 1, wght=900, glow=ROSE),
    Heart(T(28, 3.0), DUR + 1, 540, 1340, 48))

# cut transitions: alternate zoom-punch / whip
TRANS = {i: ("whip" if i in (1, 7, 15, 21, 28) else "punch") for i in range(1, len(EDIT))}
yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32) / K
r_ = np.sqrt(((xx - W / 2) / (W * 0.65)) ** 2 + ((yy - H * 0.45) / (H * 0.62)) ** 2)
VIG = (np.clip((r_ - 0.6) / 0.6, 0, 1) ** 1.6 * 0.45)[..., None].astype(np.float32)
VIG_KEEP, VIG_INK = 1 - VIG, VIG * INK
del yy, xx, r_
rng = np.random.default_rng(3)
GRAIN = [cv2.resize(rng.standard_normal((OH // 2, OW // 2)).astype(np.float32) * 0.012, (OW, OH))[..., None] for _ in range(4)]

def camera(src, tt):
    i = clip_at(tt); d = tt - STARTS[i]
    z = 1.0 + 0.04 * d / max(0.5, STARTS[i + 1] - STARTS[i])          # slow push-in per clip
    shift = 0.0; blurk = 0
    if i > 0:
        kind = TRANS[i]
        if kind == "punch" and d < 0.25: z *= 1 + 0.10 * (1 - ease(d / 0.25))
        if kind == "whip" and d < 0.16: shift = -160 * (1 - ease(d / 0.16)); blurk = int(90 * K * (1 - d / 0.16))
    nxt = STARTS[i + 1] - tt
    if i + 1 < len(EDIT) and TRANS[i + 1] == "whip" and nxt < 0.12:
        shift = 160 * ease(1 - nxt / 0.12); blurk = int(90 * K * (1 - nxt / 0.12))
    if abs(z - 1) < 1e-3 and shift == 0: fr = src
    else:
        M = np.float32([[z, 0, (1 - z) * OW / 2 + shift * K], [0, z, (1 - z) * OH / 2]])
        fr = cv2.warpAffine(src, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if blurk > 2: fr = cv2.blur(fr, (blurk | 1, 1))
    return fr

def render_frame(src, tt, fo):
    fr = camera(src, tt).astype(np.float32) * np.float32(1 / 255)
    fr *= VIG_KEEP; fr += VIG_INK
    i = clip_at(tt); d = tt - STARTS[i]
    if i > 0 and d < 0.18: fr += BLUSH * np.float32(0.22 * (1 - d / 0.18))
    for e in EL: e.draw(fr, tt)
    fr += GRAIN[fo % 4]
    if tt < 0.25: fr *= np.float32(tt / 0.25)
    tail = DUR - tt
    if tail < 0.6: fr = fr * np.float32(tail / 0.6) + INK * np.float32(1 - tail / 0.6)
    np.clip(fr, 0, 1, out=fr)
    return (fr * 255 + 0.5).astype(np.uint8)

def decoder(f0):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{f0 / FPS:.3f}", "-i", "base.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE, bufsize=OW * OH * 3)

# ---------------------------------------------------------------- audio
SR = 48000
def audio():
    rng = np.random.default_rng(11)
    n = int((DUR + 0.05) * SR); t_all = np.arange(n) / SR
    def onepole(x, cut):
        y = np.empty_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * np.asarray(cut) / SR) * np.ones(len(x))
        for i in range(len(x)): s += a[i] * (x[i] - s); y[i] = s
        return y
    def norm(y): return y / (np.abs(y).max() + 1e-9)
    def put(trk, sig, t, g):
        i = int(max(0, t) * SR); seg = trk[i:i + len(sig)]; seg += sig[:len(seg)] * g
    # --- music: 120 bpm lo-fi pop, Fmaj7 Em7 Dm7 Cmaj7
    mus = np.zeros(n)
    def note(f, d, kind):
        m = int(d * SR); t = np.arange(m) / SR
        if kind == "keys":
            y = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) * np.exp(-t * 6) + 0.12 * np.sin(6 * np.pi * f * t) * np.exp(-t * 9))
            y *= np.exp(-t * 1.6) * np.minimum(1, t / 0.006) * (1 + 0.15 * np.sin(2 * np.pi * 5 * t))
        elif kind == "bass":
            y = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t); y *= np.minimum(1, t / 0.01) * np.exp(-t * 2.2)
        else:   # pluck
            y = np.sin(2 * np.pi * f * t) * np.exp(-t * 7) + 0.4 * np.sin(4 * np.pi * f * t) * np.exp(-t * 12)
        return y
    def hz(m): return 440 * 2 ** ((m - 69) / 12)
    CH = [(65, [53, 57, 60, 64]), (64, [52, 55, 59, 62]), (62, [50, 53, 57, 60]), (60, [48, 52, 55, 59])]
    nb = int(DUR / BEAT) + 1
    end_drums = DUR - 6 * BEAT
    for b in range(nb):
        tb = b * BEAT; bar = b // 4; root, ch = CH[(bar // 1) % 4]
        if b % 4 == 0:
            for k, m in enumerate(ch): put(mus, note(hz(m + 12), 2.6, "keys"), tb + 0.012 * k, 0.07)
        if b % 2 == 0: put(mus, note(hz(root - 24), BEAT * 1.8, "bass"), tb, 0.22)
        else: put(mus, note(hz(root - 24 + 7), BEAT * 0.9, "bass"), tb + BEAT * 0.5, 0.12)
        if b >= 2 and (b % 8) in (3, 6, 7):    # little pluck motif
            mel = [ch[-1] + 12, ch[2] + 12, ch[1] + 12][(b % 8) % 3]
            put(mus, note(hz(mel), 0.5, "pluck"), tb + BEAT * 0.5, 0.06)
        if 2 * BEAT <= tb < end_drums:
            t = np.arange(int(0.35 * SR)) / SR
            kick = np.sin(2 * np.pi * np.cumsum(48 + 110 * np.exp(-t * 30)) / SR) * np.exp(-t * 9)
            if b % 2 == 0: put(mus, kick, tb, 0.42)
            if b % 4 == 3: put(mus, kick, tb + BEAT * 0.5, 0.25)
            if b % 2 == 1:
                tn = np.arange(int(0.25 * SR)) / SR
                sn = onepole(rng.standard_normal(len(tn)), 3800) * np.exp(-tn * 18) + 0.4 * np.sin(2 * np.pi * 190 * tn) * np.exp(-tn * 25)
                put(mus, norm(sn), tb, 0.16)
            for h in (0, 0.5):
                th = np.arange(int(0.05 * SR)) / SR; hat = rng.standard_normal(len(th)); hat -= onepole(hat, 7000)
                put(mus, norm(hat) * np.exp(-th * 90), tb + h * BEAT + 0.01 * rng.random(), 0.05 if h else 0.035)
    # sidechain-ish pump + gentle lowpass on intro
    pump = 1 - 0.25 * np.exp(-((t_all % (2 * BEAT)) / 0.09))
    mus *= pump
    mus = norm(mus) * 0.5
    # --- sfx
    sfx = np.zeros(n)
    def whoosh(d=0.4, lo=400, hi=6000, peak=0.6):
        m = int(d * SR); t = np.linspace(0, 1, m)
        env = np.where(t < peak, (t / peak) ** 2, ((1 - t) / (1 - peak)) ** 1.5)
        cut = lo + (hi - lo) * np.sin(np.pi * np.clip(t / (peak * 2), 0, 1)) ** 2
        x = rng.standard_normal(m); return norm((onepole(x, cut) - onepole(x, cut * 0.25)) * env)
    def pop():
        m = int(0.09 * SR); t = np.arange(m) / SR; f = 300 + 700 * np.exp(-t * 55)
        return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 40))
    def tick():
        m = int(0.03 * SR); t = np.arange(m) / SR
        return norm(np.sin(2 * np.pi * 2600 * t) * np.exp(-t * 250) + 0.2 * rng.standard_normal(m) * np.exp(-t * 600))
    def scribble(d=0.45):
        m = int(d * SR); t = np.arange(m) / SR
        x = onepole(rng.standard_normal(m), 3500) - onepole(rng.standard_normal(m), 600)
        return norm(x * (0.55 + 0.45 * np.sin(2 * np.pi * 11 * t) ** 2) * np.sin(np.pi * t / d) ** 0.6)
    def ding(f0=1318.5):
        m = int(1.4 * SR); t = np.arange(m) / SR
        return norm(sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * k) for f, a, k in [(f0, 1, 3), (f0 * 1.5, .5, 4), (f0 * 2, .3, 6)]) * np.minimum(1, t / 0.004))
    def buzz():
        m = int(0.3 * SR); t = np.arange(m) / SR
        return norm(np.sign(np.sin(2 * np.pi * 140 * t)) * np.exp(-t * 6) * np.minimum(1, t / 0.01))
    def shutter():
        m = int(0.22 * SR); t = np.arange(m) / SR; x = rng.standard_normal(m)
        return norm(onepole(x * (np.exp(-t * 80) + 0.8 * np.exp(-np.maximum(0, t - 0.09) * 70) * (t > 0.09)), 6000))
    WB, WS, POP, TICK, SCR = whoosh(0.45), whoosh(0.25, 700, 8000, 0.5), pop(), tick(), scribble()
    put(sfx, shutter(), 0.05, 0.25)
    for i in range(1, len(EDIT)):
        if TRANS[i] == "whip": put(sfx, WB, STARTS[i] - 0.3, 0.22)
        else: put(sfx, WS, STARTS[i] - 0.12, 0.08)
    for e in EL:
        if isinstance(e, Txt):
            if e.kind == "s": put(sfx, SCR, e.t0, 0.06)
            elif e.anim == "pop": put(sfx, POP, e.t0, 0.15)
        elif isinstance(e, Chip): put(sfx, POP, e.t0 + 0.05, 0.14)
        elif isinstance(e, Counter):
            for x in e.ticks(): put(sfx, TICK, x, 0.12)
            put(sfx, ding(1760), e.ticks()[-1] + 0.05, 0.06)
        elif isinstance(e, Check):
            put(sfx, SCR, e.t0, 0.06)
            put(sfx, ding() if e.ok else buzz(), e.tm, 0.08 if e.ok else 0.05)
        elif isinstance(e, Heart): put(sfx, ding(1046.5), e.t0 + 0.4, 0.06)
    # --- natural sound from the footage on the coffee / pouring / pump shots
    src = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "src.mov", "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                                       capture_output=True).stdout, np.int16).astype(np.float32) / 32768
    nat = np.zeros(n)
    for i, (s, beats, sp, mode, cx) in enumerate(EDIT):
        if i not in (1, 2, 3, 4, 10, 25, 26): continue
        d = beats * BEAT; seg = src[int(s * SR):int((s + d * sp) * SR)]
        if sp != 1: seg = np.interp(np.arange(int(d * SR)) * sp, np.arange(len(seg)), seg)
        f = int(0.03 * SR); seg = seg.copy(); seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
        put(nat, seg, STARTS[i], 1.0)
    nat = nat / (np.abs(nat).max() + 1e-9) * 0.5
    # duck music under natural sound
    env = np.convolve(np.abs(nat), np.ones(4800) / 4800, "same"); duck = 1 - 0.45 * np.clip(env / (env.max() + 1e-9) * 3, 0, 1)
    mix = mus * duck + nat * 0.8 + sfx
    fo = int(0.8 * SR); mix[-fo:] *= np.linspace(1, 0, fo)
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.89
    for name, sig in (("music.wav", mus), ("sfx.wav", sfx), ("nat.wav", nat), ("mix.wav", mix)):
        w = wave.open(name, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(sig, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "base": build_base(); sys.exit()
    if mode == "audio": audio(); sys.exit()
    if mode == "test":
        for t in TEST:
            fo = int(round(t * FPS)); dec = decoder(fo)
            src = np.frombuffer(dec.stdout.read(OW * OH * 3), np.uint8).reshape(OH, OW, 3); dec.kill()
            o = render_frame(src, t, fo)
            cv2.imwrite(f"test_{t:.2f}.png", cv2.cvtColor(cv2.resize(o, (W // 2, H // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        sys.exit()
    f0, f1, outp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    dec = decoder(f0)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-g", "100", "-pix_fmt", "yuv420p", outp], stdin=subprocess.PIPE)
    for fo in range(f0, f1):
        buf = dec.stdout.read(OW * OH * 3)
        if len(buf) < OW * OH * 3: break
        enc.stdin.write(render_frame(np.frombuffer(buf, np.uint8).reshape(OH, OW, 3), fo / FPS, fo).tobytes())
    dec.kill(); enc.stdin.close(); enc.wait(); print("done", outp)
