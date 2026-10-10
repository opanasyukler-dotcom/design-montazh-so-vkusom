"""«мини влог с 4.5 месячным малышом» — HORIZONTAL 16:9 version (3840x2160, native source pixels, no crop).
Voice-over part (0-21 s) is time-locked to the source audio; the rest is a beat-cut montage with collages.
Lossless FFV1 tile intermediates (crop/scale done once with lanczos), overlays composited in numpy, x264 crf 12 out.
Work dir: in/src.mov, words.json, emoji/<code>.png, fonts.
  python3 -I mini_vlog.py tiles               -> tiles/*.mkv
  python3 -I mini_vlog.py test 1,22,30         -> test_*.png
  python3 -I mini_vlog.py render out.mp4 [s0 s1]   (segment index range)
  python3 -I mini_vlog.py audio voice.wav mix.wav  (voice.wav = cleaned VO, 48 kHz mono)
"""
import json, os, subprocess, sys, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, K, FPS = 1920, 1080, 2, 50
OW, OH = W * K, H * K
SRC = "in/src.mov"
FONT_M, FONT_S = "Montserrat-VariableFont_wght.ttf", "BiroScriptUSPlus-Regular.ttf"
def hx(s): return np.float32([int(s[i:i + 2], 16) for i in (1, 3, 5)]) / 255
INK, COCOA, ROSE, BLUSH = hx("#2F0600"), hx("#825D4D"), hx("#C5A29C"), hx("#F7DFDD")
WHITE, BLACK = np.float32([1, 1, 1]), np.float32([0, 0, 0])
BEAT = 0.5

# ---------------------------------------------------------------- edit decision list
# tile: (src_start, speed, kind, cx, zoom, delay)  kind c = vertical crop, l = landscape tile
def C(src, cx=.5, sp=1.0, z=1.0): return (src, sp, "c", cx, z, 0.0)
def Lt(src, sp=1.0, delay=0.0): return (src, sp, "l", .5, 1.0, delay)
A = [  # time-locked to the voice-over (output time == source time)
    (0.00, 3.64, "crop", [C(0.00, .47)]),
    (3.64, 4.92, "crop", [C(3.64, .50)]),
    (4.92, 5.88, "crop", [C(4.92, .62)]),
    (5.88, 6.88, "crop", [C(5.88, .50)]),
    (6.88, 7.88, "crop", [C(6.88, .60)]),
    (7.88, 8.68, "crop", [C(7.88, .60)]),
    (8.68, 10.48, "crop", [C(8.68)]),
    (10.48, 11.20, "crop", [C(10.48, .62)]),
    (11.20, 12.24, "crop", [C(11.20, .40)]),
    (12.24, 13.92, "crop", [C(12.24, .42)]),
    (13.92, 14.76, "crop", [C(13.92, .33)]),
    (14.76, 16.00, "crop", [C(14.76, .45)]),
    (16.00, 16.92, "crop", [C(24.08, .30)]),            # «укачиваний» → rocking cut-away
    (16.92, 19.00, "crop", [C(16.92, .45)]),
    (19.00, 21.32, "crop", [C(19.00, .47, z=1.12)]),    # punch-in on «я не понимаю»
]
B_ = [  # beat-cut montage: (beats, layout, tiles)
    (4, "col3", [Lt(21.32, .70), Lt(22.72, .90, 0.5), Lt(24.08, .92, 1.0)]),
    (2, "crop", [C(25.00, .45)]),
    (2, "crop", [C(26.16, .40)]),
    (3, "col2", [Lt(27.16, .67), Lt(28.16, .96, 0.5)]),
    (2, "crop", [C(29.20, .62)]),
    (2, "crop", [C(30.04, .47)]),
    (2, "crop", [C(30.96, .30)]),
    (3, "crop", [C(31.96, sp=2.0)]),
    (3, "col3", [Lt(35.08, .40), Lt(35.68, .53, 0.5), Lt(36.48, .72, 1.0)]),
    (2, "crop", [C(37.20, .50, sp=.8)]),
    (2, "crop", [C(38.00, .45)]),
    (2, "crop", [C(40.04, .55)]),
    (2, "crop", [C(41.60, .35)]),
    (3, "col2", [Lt(42.60, .74), Lt(43.72, .94, 0.5)]),
    (4, "crop", [C(44.76, .45, sp=1.4)]),
    (3, "crop", [C(47.60, .40, sp=.7)]),
    (2, "crop", [C(48.64, .45)]),
]
SEGS = [dict(t0=a, t1=b, lay=lay, tiles=t) for a, b, lay, t in A]
t = A[-1][1]
for beats, lay, tiles in B_:
    SEGS.append(dict(t0=t, t1=t + beats * BEAT, lay=lay, tiles=tiles)); t += beats * BEAT
DUR = t
NFR = int(round(DUR * FPS))
# keep every tile inside its source shot (shot boundaries from histogram cut detection, 25 fps proxy)
SHOTS = [0.0, 3.64, 4.12, 4.92, 5.88, 6.88, 7.88, 8.68, 10.48, 11.2, 12.24, 13.92, 14.76, 21.32, 22.72, 24.08, 25.0, 26.16, 27.16, 28.16,
         29.2, 30.04, 30.96, 31.96, 35.08, 35.68, 36.48, 37.2, 38.0, 40.04, 41.6, 42.6, 43.72, 44.76, 47.6, 48.64, 49.9]
FIT = []
for s_ in SEGS:
    if s_["t0"] < A[-1][1] and s_["tiles"][0][0] == s_["t0"]: continue          # time-locked VO segments are the source's own cut
    nt = []
    for (src, sp, kind, cx, z, delay) in s_["tiles"]:
        vis = (s_["t1"] - s_["t0"]) - delay
        end = min(b for b in SHOTS if b > src + 0.01) - 0.08
        if src + vis * sp > end:
            nsp = max(0.25, (end - src) / vis); FIT.append((s_["t0"], src, sp, round(nsp, 3))); sp = nsp
        nt.append((src, sp, kind, cx, z, delay))
    s_["tiles"] = nt
for i, s in enumerate(SEGS):
    s["f0"], s["f1"] = int(round(s["t0"] * FPS)), int(round(s["t1"] * FPS)); s["i"] = i
B0 = len(A)          # first montage segment
def seg_at(tt): return next(s for s in SEGS if s["t0"] <= tt < s["t1"] or s is SEGS[-1])

def tsize(s, ti):
    """design size of tile ti in segment s (full frame for single shots; collages: 2 side by side / 1 big + 2 small)"""
    if s["lay"] == "crop": return W, H
    if s["lay"] == "col2": return 880, 495
    return (1120, 630) if ti == 0 else (660, 371)
def tile_path(si, ti): return f"tiles/s{si:02d}_{ti}.mkv"
def build_tiles():
    os.makedirs("tiles", exist_ok=True)
    for s in SEGS:
        for ti, (src, sp, kind, cx, z, delay) in enumerate(s["tiles"]):
            n = s["f1"] - s["f0"] - int(round(delay * FPS))
            if kind == "c" and z == 1.0:
                geo = "null"                                         # native 3840x2160: no crop, no scale
            elif kind == "c":
                cw, ch = 3840 / z, 2160 / z; x0 = int(np.clip(cx * 3840 - cw / 2, 0, 3840 - cw)); y0 = int((2160 - ch) / 2)
                geo = f"crop={int(cw)}:{int(ch)}:{x0}:{y0},scale={OW}:{OH}:flags=lanczos"
            else:
                tw, th = tsize(s, ti); geo = f"scale={tw * K}:{th * K}:flags=lanczos"
            vf = f"{geo},setpts=(PTS-STARTPTS)/{sp},fps={FPS}"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{src:.3f}", "-t", f"{n / FPS * sp + 0.3:.3f}", "-i", SRC, "-an", "-vf", vf,
                            "-frames:v", str(n), "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p", tile_path(s["i"], ti)], check=True)
        print("tiles", s["i"], flush=True)

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
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1].astype(np.float32) * (1 / 255)
    dst[y0:y1, x0:x1] = np.clip((d * (1 - s[..., 3:4] * alpha) + s[..., :3] * alpha) * 255 + 0.5, 0, 255).astype(np.uint8)
_fc = {}
def font(path, size, wght=None):
    k = (path, size, wght)
    if k not in _fc:
        f = ImageFont.truetype(path, size)
        if wght: f.set_variation_by_axes([wght])
        _fc[k] = f
    return _fc[k]
def raster(txt, f, col, dil=0, sh_a=0.55, sh_r=7):
    pad = 18 * K; asc, desc = f.getmetrics(); w = int(f.getlength(txt)) + 2 * pad + 24 * K
    im = Image.new("L", (w, asc + desc + 2 * pad), 0); ImageDraw.Draw(im).text((pad, pad + asc), txt, font=f, fill=255, anchor="ls")
    a = np.asarray(im).astype(np.float32) / 255
    if dil: a = cv2.dilate(a, np.ones((dil, dil), np.uint8))
    out = np.zeros(a.shape + (4,), np.float32); out[..., :3] = col * a[..., None]; out[..., 3] = a
    s_ = np.minimum(1, cv2.GaussianBlur(a, (0, 0), sh_r * K) * 1.4) * sh_a
    sh = np.zeros_like(out); sh[..., 3] = s_
    return out, sh, (pad + asc) / K, pad / K, f.getlength(txt) / K
def rounded(w, h, r):
    m = np.zeros((h, w), np.uint8); cv2.rectangle(m, (r, 0), (w - r, h), 255, -1); cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)): cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return (m.astype(np.float32) / 255)[..., None]
_mk = {}
def tile_mask(tw, th):
    if (tw, th) not in _mk:
        m = rounded(tw * K, th * K, 24 * K)
        sh = cv2.GaussianBlur(np.pad(m[..., 0], 40 * K)[::4, ::4], (0, 0), 5)
        S_ = np.zeros((m.shape[0] + 80 * K, m.shape[1] + 80 * K, 4), np.float32); S_[..., 3] = cv2.resize(sh, (S_.shape[1], S_.shape[0])) * 0.35
        _mk[(tw, th)] = (m, S_)
    return _mk[(tw, th)]

# ---------------------------------------------------------------- text & icons
class Word:
    """one-word subtitle: thin Montserrat white, or Biro Script blush for highlighted words"""
    def __init__(self, txt, t0, t1, script=False):
        f = font(FONT_S, int(120 * K)) if script else font(FONT_M, int(64 * K), 300)
        self.img, self.sh, base, pad, adv = raster(txt, f, BLUSH if script else WHITE, dil=(3 if script else 0), sh_a=0.6)
        self.x, self.y = 960 - adv / 2 - pad, 990 - base
        self.t0, self.t1, self.script = t0, t1, script
    def active(self, tt): return self.t0 <= tt < self.t1
    def draw(self, dst, tt):
        age = tt - self.t0; img, sh = self.img, self.sh
        if self.script:
            p = ease_io(age / 0.35)
            if p < 1:
                ww = img.shape[1]; ramp = np.clip((np.arange(ww) - p * ww * 1.15) / (-0.15 * ww), 0, 1)[None, :, None].astype(np.float32)
                img, sh = img * ramp, sh * ramp
            blend(dst, sh, self.x, self.y + 4); blend(dst, img, self.x, self.y); return
        sc = 0.86 + 0.14 * back(age / 0.16); al = min(1, age / 0.06)
        if abs(sc - 1) > 0.005:
            hh, ww = img.shape[:2]; nw, nh = int(ww * sc), int(hh * sc)
            img, sh = cv2.resize(img, (nw, nh)), cv2.resize(sh, (nw, nh)); dx, dy = (ww - nw) / 2 / K, (hh - nh) / 2 / K
        else: dx = dy = 0
        blend(dst, sh, self.x + dx, self.y + dy + 4, al); blend(dst, img, self.x + dx, self.y + dy, al)

class Label:
    """handwritten caption (Biro Script) + optional Montserrat-thin small line, blur-rise in, fade out"""
    def __init__(self, txt, t0, t1, y=960, size=110, sub=None, x=960):
        self.img, self.sh, base, pad, adv = raster(txt, font(FONT_S, int(size * K)), WHITE, dil=3, sh_a=0.6)
        self.x, self.y, self.t0, self.t1 = x - adv / 2 - pad, y - base, t0, t1
        self.right, self.base_y = x + adv / 2, y
        self.sub = None
        if sub:
            si, ss, sb, sp, sa = raster(sub, font(FONT_M, int(46 * K), 300), WHITE, sh_a=0.6)
            self.sub = (si, ss, x - sa / 2 - sp, y + 70 - sb)
    def active(self, tt): return self.t0 <= tt <= self.t1 + 0.3
    def draw(self, dst, tt):
        age = tt - self.t0; out = ease((tt - self.t1) / 0.25) if tt > self.t1 else 0
        p = ease_io(age / 0.5); img, sh = self.img, self.sh
        if p < 1:
            ww = img.shape[1]; ramp = np.clip((np.arange(ww) - p * ww * 1.15) / (-0.15 * ww), 0, 1)[None, :, None].astype(np.float32)
            img, sh = img * ramp, sh * ramp
        dy = -14 * out; al = 1 - out
        blend(dst, sh, self.x, self.y + dy + 4, al); blend(dst, img, self.x, self.y + dy, al)
        if self.sub:
            si, ss, sx, sy = self.sub; a2 = min(1, max(0, age - 0.3) / 0.25) * al
            blend(dst, ss, sx, sy + dy + 3, a2); blend(dst, si, sx, sy + dy, a2)

_emo = {}
def sticker(code, size):
    k = (code, size)
    if k not in _emo:
        im = np.asarray(Image.open(f"emoji/{code}.png").convert("RGBA")).astype(np.float32) / 255
        im = np.pad(im, ((40, 40), (40, 40), (0, 0))); a = im[..., 3]
        border = cv2.GaussianBlur(cv2.dilate((a > 0.08).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))).astype(np.float32) / 255, (0, 0), 1.2)
        out = np.zeros_like(im); out[..., :3] = border[..., None]; out[..., 3] = border
        out[..., :3] = out[..., :3] * (1 - a[..., None]) + im[..., :3] * a[..., None]; out[..., 3] = np.maximum(out[..., 3], a)
        out = cv2.resize(out, (P(size), P(size)), interpolation=cv2.INTER_AREA)
        sh = np.zeros_like(out); sh[..., 3] = cv2.GaussianBlur(out[..., 3], (0, 0), 8 * K) * 0.35
        _emo[k] = (out, sh)
    return _emo[k]
class Icon:
    def __init__(self, code, x, y, t0, t1, size=150, rot=0.0):
        self.code, self.x, self.y, self.t0, self.t1, self.size, self.rot = code, x, y, t0, t1, size, rot
    def active(self, tt): return self.t0 <= tt <= self.t1 + 0.22
    def draw(self, dst, tt):
        img, sh = sticker(self.code, self.size); age = tt - self.t0
        out = ease((tt - self.t1) / 0.2) if tt > self.t1 else 0
        sc = back(age / 0.35, 2.2) * (1 - 0.5 * out)
        if sc < 0.03: return
        ang = self.rot + 6 * np.sin(age * 3.0) + (1 - min(1, age / 0.3)) * -16
        n = max(4, int(img.shape[1] * sc)); M = cv2.getRotationMatrix2D((n / 2, n / 2), ang, 1)
        im2, sh2 = cv2.warpAffine(cv2.resize(img, (n, n)), M, (n, n)), cv2.warpAffine(cv2.resize(sh, (n, n)), M, (n, n))
        dy = -6 * np.sin(age * 2.4); al = min(1, age / 0.08) * (1 - out)
        blend(dst, sh2, self.x - n / K / 2 + 6, self.y - n / K / 2 + 12 + dy, al)
        blend(dst, im2, self.x - n / K / 2, self.y - n / K / 2 + dy, al)

# subtitles: one word at a time (numbers glued to the next word), highlights in handwriting
WORDS = json.load(open("words.json"))
toks = []
for s, e, w in WORDS:
    w = w.strip()
    if toks and toks[-1][2].isdigit(): toks[-1][2] += " " + w; toks[-1][1] = e; continue
    toks.append([s, e, w])
HL = {"умывалась", "тусуемся", "истерику", "отсыпается", "укачиваний", "ноль", "понимаю", "7 утра", "5 утра"}
def clean(w): return w.lower().strip(",.?!«»")
OVER = []
for k, (s, e, w) in enumerate(toks):
    nxt = toks[k + 1][0] if k + 1 < len(toks) else e + 0.6
    OVER.append(Word(w.strip(",.?!"), s - 0.03, min(nxt - 0.02, e + 0.55), script=clean(w) in HL))

def T(si, beat=0.0): return SEGS[si]["t0"] + beat * BEAT
OVER += [
    Label("мини влог", 0.15, 3.45, y=500, size=170, sub="с 4.5 месячным малышом"),
    Icon("1f90d", 1300, 400, 0.9, 3.45, size=110, rot=10),
    Label("готовлю яичницу", 8.75, 10.40, y=160, size=96), Icon("1f373", 1400, 120, 8.9, 10.4, size=130, rot=-6),
    Icon("2615", 1560, 380, 11.25, 12.2, size=150, rot=8),
    Icon("1f62e_200d_1f4a8", 1480, 330, 20.55, 21.30, size=150, rot=6),
    Label("кофе — первым делом", T(B0, 0.3), T(B0, 3.9), x=640, y=1010, size=84), Icon("2615", 1060, 950, T(B0, 0.3), T(B0, 3.9), size=120, rot=8),
    Label("бутылочки", T(B0 + 1, .1), T(B0 + 1, 1.9)), Icon("1f37c", 1290, 890, T(B0 + 1, .1), T(B0 + 1, 1.9), rot=10),
    Label("завтрак", T(B0 + 2, .1), T(B0 + 2, 1.9)), Icon("1f373", 1240, 890, T(B0 + 2, .1), T(B0 + 2, 1.9), rot=-8),
    Label("время игр", T(B0 + 3, .2), T(B0 + 4, 1.9), y=980), Icon("1f9f8", 1270, 910, T(B0 + 3, .2), T(B0 + 4, 1.9), rot=8),
    Label("наконец-то умылась", T(B0 + 5, .1), T(B0 + 5, 1.95), size=100), Icon("1fae7", 1440, 890, T(B0 + 5, .1), T(B0 + 5, 1.95)),
    Label("уснула", T(B0 + 6, .1), T(B0 + 6, 1.9)), Icon("1f634", 1210, 890, T(B0 + 6, .1), T(B0 + 6, 1.9), rot=-6),
    Label("готовлю смесь", T(B0 + 8, .2), T(B0 + 8, 2.9), x=640, y=1010, size=96), Icon("1f37c", 990, 950, T(B0 + 8, .2), T(B0 + 8, 2.9), size=120, rot=8),
    Label("кормление", T(B0 + 9, .1), T(B0 + 9, 1.9)), Icon("1f90d", 1300, 890, T(B0 + 9, .1), T(B0 + 9, 1.9)),
    Label("на прогулку", T(B0 + 10, .1), T(B0 + 11, 1.9)), Icon("2600_fe0f", 1330, 880, T(B0 + 10, .1), T(B0 + 11, 1.9), rot=10),
    Label("кофе с собой", T(B0 + 12, .1), T(B0 + 12, 1.9)), Icon("2615", 1340, 890, T(B0 + 12, .1), T(B0 + 12, 1.9), rot=-8),
    Label("вот такое у нас утро", T(B0 + 15, .2), DUR + 1, y=940, size=100), Icon("1f90d", 960, 790, T(B0 + 15, .5), DUR + 1, size=120),
]

# icons that accompany a handwritten label sit right after its last letter (no overlap)
for k in range(len(OVER) - 1):
    l_, i_ = OVER[k], OVER[k + 1]
    if isinstance(l_, Label) and isinstance(i_, Icon) and abs(l_.t0 - i_.t0) < 0.3 and l_.t1 < DUR and l_.base_y > 300:
        i_.x = l_.right + i_.size * 0.62 + 40; i_.y = l_.base_y - 55

# requested: no stickers/icons, no music (ICONS=1 / MUSIC=1 bring them back)
if os.environ.get("ICONS", "0") != "1": OVER = [o for o in OVER if not isinstance(o, Icon)]
USE_MUSIC = os.environ.get("MUSIC", "0") == "1"

# ---------------------------------------------------------------- compositor
class Tiles:
    def __init__(self): self.si = None; self.decs = []; self.last = []
    def open(self, si):
        for d in self.decs:
            if d: d.kill()
        s = SEGS[si]; self.si = si; self.decs = []; self.last = []
        for ti, (src, sp, kind, cx, z, delay) in enumerate(s["tiles"]):
            tw, th = tsize(s, ti); w, h = tw * K, th * K
            self.decs.append(subprocess.Popen(["ffmpeg", "-v", "error", "-i", tile_path(si, ti), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                              stdout=subprocess.PIPE, bufsize=w * h * 3))
            self.last.append(None)
    def get(self, si, fo):
        if si != self.si: self.open(si)
        s = SEGS[si]; out = []
        for ti, (src, sp, kind, cx, z, delay) in enumerate(s["tiles"]):
            tw, th = tsize(s, ti); w, h = tw * K, th * K
            if fo - s["f0"] >= int(round(delay * FPS)):
                buf = self.decs[ti].stdout.read(w * h * 3)
                if len(buf) == w * h * 3: self.last[ti] = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                out.append(self.last[ti])
            else: out.append(None)
        return out
TILES = Tiles()

def blurred_bg(img, dim=0.18):
    q = cv2.resize(img, (OW // 16, int(OW // 16 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    q = cv2.GaussianBlur(q, (0, 0), 3)
    # cover the full vertical frame
    h = OH; w = int(h * q.shape[1] / q.shape[0]); big = cv2.resize(q, (w, h), interpolation=cv2.INTER_LINEAR)
    x0 = (w - OW) // 2; big = big[:, x0:x0 + OW]
    return (big.astype(np.float32) * (1 - dim)).astype(np.uint8)
PAPER = None
def paper():
    global PAPER
    if PAPER is None:
        rng = np.random.default_rng(2); n = cv2.GaussianBlur(rng.standard_normal((OH // 4, OW // 4)).astype(np.float32), (0, 0), 1.0) * 4
        base = np.empty((OH, OW, 3), np.float32); base[:] = BLUSH * 255
        PAPER = np.clip(base + cv2.resize(n, (OW, OH))[..., None], 0, 255).astype(np.uint8)
    return PAPER.copy()

def place_tile(fr, img, cx, cy, age, slide=True):
    th, tw = img.shape[0] // K, img.shape[1] // K; TMASK, TSHADOW = tile_mask(tw, th)
    e = ease(age / 0.35) if slide else 1.0
    x, y = cx - tw / 2, cy - th / 2 + 50 * (1 - e)
    al = min(1, age / 0.15) if slide else 1.0
    blend(fr, TSHADOW, x - 40, y - 40 + 14, al)
    t4 = np.concatenate([img.astype(np.float32) / 255 * TMASK, TMASK], 2)
    blend(fr, t4, x, y, al)

def render_frame(fo):
    tt = fo / FPS; s = seg_at(tt); si = s["i"]; imgs = TILES.get(si, fo); d = tt - s["t0"]
    lay = s["lay"]
    if lay == "crop":
        fr = imgs[0].copy()
    elif lay == "frame":
        fr = blurred_bg(imgs[0]); place_tile(fr, imgs[0], 540, 1000, d, slide=False)
    else:
        fr = paper(); n = len(imgs)
        xs, ys = ([640, 1500, 1500], [500, 290, 720]) if n == 3 else ([500, 1420], [500, 540])
        for ti, img in enumerate(imgs):
            if img is None: continue
            place_tile(fr, img, xs[ti], ys[ti], d - s["tiles"][ti][5])
    # cut accents: quick punch-in for crops, white flash on layout changes, whip blur into collages
    if lay == "crop" and si > 0 and d < 0.16:
        z = 1 + 0.05 * (1 - ease(d / 0.16)); M = np.float32([[z, 0, (1 - z) * OW / 2], [0, z, (1 - z) * OH / 2]])
        fr = cv2.warpAffine(fr, M, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if si > 0 and SEGS[si - 1]["lay"] != lay and d < 0.14:
        a = 0.35 * (1 - d / 0.14); fr = cv2.addWeighted(fr, 1 - a, np.full_like(fr, 255), a, 0)
    if si >= B0 and lay.startswith("col") and d < 0.1:
        k = int(120 * K * (1 - d / 0.1)) | 1; fr = cv2.blur(fr, (k, 1))
    for o in OVER:
        if o.active(tt): o.draw(fr, tt)
    if DUR - tt < 0.5: fr = (fr.astype(np.float32) * max(0, (DUR - tt) / 0.5)).astype(np.uint8)
    if tt < 0.2: fr = (fr.astype(np.float32) * (tt / 0.2)).astype(np.uint8)
    return fr

# ---------------------------------------------------------------- sound: music bed, foley, UI sfx
SR = 48000
def audio(voice_path, out_path):
    rng = np.random.default_rng(21); n = int((DUR + 0.2) * SR); tA = np.arange(n) / SR
    def onepole(x, cut):
        y = np.empty_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * np.asarray(cut, dtype=np.float64) / SR) * np.ones(len(x))
        for i in range(len(x)): s += a[i] * (x[i] - s); y[i] = s
        return y
    def bp(x, lo, hi): return onepole(x, hi) - onepole(x, lo)
    def norm(y): return y / (np.abs(y).max() + 1e-9)
    def put(trk, sig, t, g):
        i = int(max(0, t) * SR); seg = trk[i:i + len(sig)]; seg += sig[:len(seg)] * g
    def env(m, att, rel): t = np.arange(m) / SR; d = m / SR; return np.minimum(1, t / att) * np.minimum(1, (d - t) / rel)
    # --- foley
    def pour(d=1.0, f0=500, f1=1300):           # water into a mug: bubbly resonance that rises as it fills
        m = int(d * SR); t = np.arange(m) / SR; x = rng.standard_normal(m)
        fc = f0 + (f1 - f0) * (t / d) ** 0.7
        y = np.zeros(m); seg = 1200
        for i in range(0, m, seg):
            j = min(m, i + seg); c = fc[i]; y[i:j] = bp(x[i:j], c * 0.75, c * 1.35)
        gurgle = 1 + 0.6 * np.sin(2 * np.pi * (9 + 6 * np.sin(2 * np.pi * 1.3 * t)) * t)
        drops = np.convolve((rng.random(m) < 0.004) * rng.uniform(0.3, 1, m), np.exp(-np.arange(300) / 40) * np.sin(np.arange(300) * 0.35), "same")
        return norm((y * gurgle + 0.25 * drops) * env(m, 0.06, 0.15))
    def sizzle(d=1.8):
        m = int(d * SR); x = rng.standard_normal(m); h = x - onepole(x, 3000)
        crack = np.convolve((rng.random(m) < 0.003) * rng.uniform(0.3, 1.2, m), np.exp(-np.arange(120) / 12), "same")
        return norm((0.5 * h * (0.8 + 0.2 * np.sin(2 * np.pi * 4 * np.arange(m) / SR)) + crack) * env(m, 0.1, 0.3))
    def clink(f=2400):
        m = int(0.35 * SR); t = np.arange(m) / SR
        return norm(sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * k) for r, a, k in [(1, 1, 18), (2.76, .5, 25), (5.4, .25, 35)]))
    def sip():
        m = int(0.45 * SR); x = rng.standard_normal(m); t = np.arange(m) / SR
        return norm(bp(x, 900, 3500) * np.sin(np.pi * t / 0.45) ** 2 * (0.6 + 0.4 * np.sin(2 * np.pi * 14 * t)))
    def sigh(d=1.3):                            # breathy exhale with a falling formant
        m = int(d * SR); t = np.arange(m) / SR; x = rng.standard_normal(m)
        y = np.zeros(m); seg = 1200
        for i in range(0, m, seg):
            j = min(m, i + seg); c = 1300 - 700 * (t[i] / d); y[i:j] = bp(x[i:j], c * 0.5, c * 1.6)
        e = np.minimum(1, t / 0.12) * np.exp(-np.maximum(0, t - 0.25) * 2.4)
        return norm(y * e + 0.3 * bp(x, 150, 400) * e)
    def tap(d=1.0):
        m = int(d * SR); x = rng.standard_normal(m); t = np.arange(m) / SR
        return norm((bp(x, 700, 5000) * (0.8 + 0.2 * np.sin(2 * np.pi * 23 * t)) + 0.4 * bp(x, 200, 600)) * env(m, 0.05, 0.2))
    def splash():
        m = int(0.5 * SR); x = rng.standard_normal(m); t = np.arange(m) / SR
        return norm((bp(x, 300, 4000) * np.exp(-t * 9) + 0.5 * bp(x, 1500, 7000) * np.exp(-t * 14)) * np.minimum(1, t / 0.005))
    def rattle(d=1.0):
        m = int(d * SR); y = np.zeros(m)
        for k in range(int(d * 7)):
            st = int((k / 7 + rng.uniform(-0.01, 0.01)) * SR); bead = bp(rng.standard_normal(int(0.06 * SR)), 2500, 9000) * np.exp(-np.arange(int(0.06 * SR)) / SR * 60)
            st = max(0, min(m - 1, st)); L = min(len(bead), m - st); y[st:st + L] += bead[:L]
        return norm(y)
    def pump(d=1.5):
        m = int(d * SR); y = np.zeros(m); t = np.arange(m) / SR
        motor = bp(rng.standard_normal(m), 120, 600) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.6 * t) ** 2)
        for k in np.arange(0, d, 0.62):
            st = int(k * SR); c = clink(900)[:int(0.08 * SR)] * 0.5; y[st:st + len(c)] += c[:max(0, m - st)]
        return norm(motor * 0.6 + y)
    def birds(d=3.0):
        m = int(d * SR); y = np.zeros(m)
        for _ in range(int(d * 2.2)):
            st = rng.uniform(0, d - 0.3); f0 = rng.uniform(2800, 4800); L = rng.uniform(0.06, 0.16); mm = int(L * SR); t = np.arange(mm) / SR
            ch = np.sin(2 * np.pi * np.cumsum(f0 * (1 + 0.25 * np.sin(2 * np.pi * rng.uniform(25, 45) * t))) / SR) * np.sin(np.pi * t / L) ** 2
            for r in range(rng.integers(1, 4)):
                s0 = int((st + r * (L + 0.04)) * SR); y[s0:s0 + mm] += ch[:max(0, m - s0)] * 0.8 ** r
        return norm(y)
    def outdoor(d):
        m = int(d * SR); x = rng.standard_normal(m); t = np.arange(m) / SR
        wind = onepole(x, 400) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.23 * t)); wheels = bp(rng.standard_normal(m), 60, 220) * 0.6
        return norm(wind + wheels) * env(m, 0.4, 0.5)
    # --- ui sfx
    def whoosh(d=0.35, lo=600, hi=6000):
        m = int(d * SR); t = np.linspace(0, 1, m); x = rng.standard_normal(m); return norm(bp(x, lo, hi) * np.sin(np.pi * t) ** 2)
    def pop(f0=520):
        m = int(0.11 * SR); t = np.arange(m) / SR; f = f0 + 900 * np.exp(-t * 60); return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 38))
    def scribble(d=0.45):
        m = int(d * SR); t = np.arange(m) / SR; x = bp(rng.standard_normal(m), 700, 4000)
        return norm(x * (0.55 + 0.45 * np.sin(2 * np.pi * 11 * t) ** 2) * np.sin(np.pi * t / d) ** 0.6)
    def chime(f0=1046.5):
        m = int(1.5 * SR); t = np.arange(m) / SR
        return norm(sum(a * np.sin(2 * np.pi * f0 * r * t) * np.exp(-t * k) for r, a, k in [(1, 1, 2.8), (2, .3, 4), (3, .12, 6)]) * np.minimum(1, t / 0.004))
    # --- music: soft lo-fi bed, 120 bpm, louder in the montage
    mus = np.zeros(n)
    def hz(mm): return 440 * 2 ** ((mm - 69) / 12)
    def keys(f, d):
        m = int(d * SR); t = np.arange(m) / SR
        y = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) * np.exp(-t * 6)
        return y * np.exp(-t * 1.4) * np.minimum(1, t / 0.008) * (1 + 0.12 * np.sin(2 * np.pi * 5 * t))
    CH = [(65, [53, 57, 60, 64]), (64, [52, 55, 59, 62]), (62, [50, 53, 57, 60]), (60, [48, 52, 55, 59])]
    for b in range(int(DUR / BEAT) + 1):
        tb = b * BEAT; root, ch = CH[(b // 4) % 4]
        if b % 4 == 0:
            for k, mm in enumerate(ch): put(mus, keys(hz(mm + 12), 2.4), tb + 0.01 * k, 0.06)
        if b % 2 == 0:
            m = int(BEAT * 1.6 * SR); t = np.arange(m) / SR
            put(mus, (np.sin(2 * np.pi * hz(root - 24) * t)) * np.exp(-t * 2.2) * np.minimum(1, t / 0.01), tb, 0.18)
        if tb >= 21.0 and tb < DUR - 1.5:
            t2 = np.arange(int(0.3 * SR)) / SR; kick = np.sin(2 * np.pi * np.cumsum(48 + 100 * np.exp(-t2 * 30)) / SR) * np.exp(-t2 * 10)
            if b % 2 == 0: put(mus, kick, tb, 0.35)
            if b % 2 == 1: put(mus, norm(bp(rng.standard_normal(int(0.2 * SR)), 1500, 6000) * np.exp(-np.arange(int(0.2 * SR)) / SR * 18)), tb, 0.12)
            for hh in (0, 0.5):
                th = int(0.04 * SR); put(mus, norm(bp(rng.standard_normal(th), 6000, 14000)) * np.exp(-np.arange(th) / SR * 90), tb + hh * BEAT, 0.035)
    mus = norm(mus) * np.where(tA < 21.0, 0.10, 0.30)
    if not USE_MUSIC: mus[:] = 0
    fol = np.zeros(n); ui = np.zeros(n)
    put(fol, pour(1.0, 450, 1250), 6.9, 0.30); put(fol, pour(0.8, 350, 900), 7.9, 0.22)
    put(fol, sizzle(1.8), 8.7, 0.18); put(fol, clink(), 10.55, 0.18); put(fol, clink(2600), 10.85, 0.14); put(fol, clink(2300), 11.05, 0.12)
    put(fol, sip(), 12.7, 0.10); put(fol, sizzle(0.85), 13.92, 0.18)
    put(fol, sigh(1.3), 20.95, 0.32)
    for tb in (T(B0, 0), T(B0, 1), T(B0, 2)): put(fol, clink(2200), tb + 0.1, 0.08)
    put(fol, tap(1.0), T(B0 + 1), 0.22); put(fol, clink(3000), T(B0 + 2, 0.5), 0.15)
    put(fol, rattle(1.0), T(B0 + 4), 0.18)
    for k in range(3): put(fol, splash(), T(B0 + 5, 0.1) + k * 0.33, 0.24)
    put(fol, sigh(1.1), T(B0 + 6, 0.2), 0.22)
    m_ = int(1.5 * SR); tt_ = np.arange(m_) / SR
    machine = norm(bp(rng.standard_normal(m_), 140, 520) * (0.7 + 0.3 * np.sin(2 * np.pi * 50 * tt_))) * env(m_, 0.15, 0.3)
    put(fol, machine, T(B0 + 8), 0.10); put(fol, pour(1.1, 700, 1600), T(B0 + 8, 0.4), 0.18)
    out_t = T(B0 + 10); put(fol, outdoor(DUR - out_t), out_t, 0.10); put(fol, birds(DUR - out_t), out_t, 0.08)
    for o in OVER:
        if isinstance(o, Icon): put(ui, pop(np.random.default_rng(int(o.t0 * 10)).uniform(450, 650)), o.t0, 0.10)
        if isinstance(o, Label): put(ui, scribble(), o.t0, 0.06)
    for s in SEGS[1:]:
        if s["i"] >= B0 or s["lay"] != "crop": put(ui, whoosh(), s["t0"] - 0.18, 0.08 if s["i"] >= B0 else 0.06)
    put(ui, chime(), T(B0 + 15, 0.5), 0.07); put(ui, chime(784), 0.2, 0.05)
    # voice + duck
    w = wave.open(voice_path); v = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    v = np.pad(v, (0, max(0, n - len(v))))[:n]
    envv = np.convolve(np.abs(v), np.ones(2400) / 2400, "same"); duck = 1 - 0.5 * np.clip(envv / 0.06, 0, 1)
    mix = v + (mus + fol * 0.9 + ui) * duck
    fo = int(0.5 * SR); mix[-fo:] *= np.linspace(1, 0, fo)
    mix = np.clip(mix, -0.95, 0.95)
    w = wave.open(out_path, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "tiles": build_tiles(); sys.exit()
    if mode == "audio": audio(sys.argv[2], sys.argv[3]); sys.exit()
    if mode == "info":
        for s in SEGS: print(s["i"], round(s["t0"], 2), round(s["t1"], 2), s["lay"])
        print("DUR", DUR, "frames", NFR); sys.exit()
    if mode == "test":
        want = sorted(float(x) for x in sys.argv[2].split(","))
        for t in want:
            fo = int(round(t * FPS)); s = seg_at(t)
            TILES.open(s["i"])
            for f in range(s["f0"], fo): TILES.get(s["i"], f)      # advance decoders to the frame
            o = render_frame(fo)
            cv2.imwrite(f"test_{t:.2f}.png", cv2.cvtColor(cv2.resize(o, (540, 960), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        sys.exit()
    out = sys.argv[2]; s0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0; s1 = int(sys.argv[4]) if len(sys.argv) > 4 else len(SEGS)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-g", "100", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "5.2",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", out], stdin=subprocess.PIPE)
    f_end = NFR if s1 >= len(SEGS) else SEGS[s1]["f0"]
    for fo in range(SEGS[s0]["f0"], f_end): enc.stdin.write(render_frame(fo).tobytes())
    enc.stdin.close(); enc.wait(); print("done", out)
