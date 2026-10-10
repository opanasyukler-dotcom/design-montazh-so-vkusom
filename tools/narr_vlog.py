"""«мини влог с 4.5 месячным малышом» — NARRATED 16:9 version (3840x2160, native source pixels).
Montage is cut to the narration («Каждое утро она надеялась…»): each phrase gets its own shots, short shots are slowed down
with frame blending (framerate filter) and freeze on their last frame rather than spill into the next shot.
Music bed: the «Desperate Housewives» theme from the user's screen recording, looped on its own 23.55 s phrase repeat, ducked.
Work dir: in/src.mov, in/vo0..4.mp3, in/screen.mp4, nwords.json, nar_plan.json, fonts.
  python3 -I narr_vlog.py tiles | test 1,22,30 | info
  python3 -I narr_vlog.py render out.mp4 [s0 s1]   (segment index range)
  python3 -I narr_vlog.py audio mix.wav
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


# ---------------------------------------------------------------- edit decision list (cut to the narration)
# tile: (src_start, speed, kind, cx, zoom, delay)  kind c = full frame, l = collage tile; speed None = fit to the shot
def C(src, sp=None, z=1.0, cx=.5): return (src, sp, "c", cx, z, 0.0)
def Lt(src, delay=0.0, sp=None): return (src, sp, "l", .5, 1.0, delay)
EDL = [  # (t0, layout, tiles); a segment lasts until the next one starts
    (0.00, "crop", [C(4.92)]),                                   # window — title, «Каждое утро она надеялась»
    (3.50, "crop", [C(4.12)]),                                   # «что сегодня всё пойдёт
    (4.80, "crop", [C(5.88)]),                                   #   по плану»
    (6.20, "crop", [C(31.96)]),                                  # «Но, как известно, в доме, где есть маленький ребёнок»
    (9.50, "crop", [C(28.16)]),                                  # «планы строит
    (10.60, "crop", [C(29.20)]),                                 #   только один человек»
    (11.80, "crop", [C(19.30, 1.0)]),                            # «И это не мама»
    (13.30, "crop", [C(6.88)]),                                  # «Поэтому утро начиналось
    (14.60, "crop", [C(7.88)]),                                  #   с кофе»
    (15.80, "crop", [C(11.20)]),                                 # «Единственного, что ещё связывало её
    (17.20, "crop", [C(12.24)]),                                 #   с прежней жизнью»
    (19.60, "crop", [C(25.00)]),                                 # «Затем наступало время бутылочек»
    (21.40, "col2", [Lt(25.30), Lt(36.48, .5)]),                 # «Иногда ей казалось, что она моет их гораздо чаще,
    (24.00, "crop", [C(0.30, 1.0)]),                             #   чем моется сама»
    (26.80, "crop", [C(8.68)]),                                  # «Дальше был завтрак»
    (29.00, "crop", [C(21.32)]),                                 # «Она уже давно перестала надеяться,
    (31.00, "crop", [C(13.92)]),                                 #   что когда-нибудь
    (32.30, "crop", [C(26.16)]),                                 #   съест его горячим»
    (33.90, "col2", [Lt(28.16), Lt(29.20, .5)]),                 # «А у маленькой хозяйки дома
    (35.90, "crop", [C(27.16)]),                                 #   тем временем было полно дел»
    (37.80, "crop", [C(5.88)]),                                  # «Игрушки,
    (39.00, "crop", [C(29.30)]),                                 #   погремушки»
    (40.40, "crop", [C(28.16)]),                                 # «и первые попытки исследовать этот огромный мир»
    (42.60, "crop", [C(24.08)]),                                 # «И пока малышка была занята,
    (44.20, "crop", [C(14.76, 1.0)]),                            #   её мама совершила практически невозможное»
    (47.60, "crop", [C(30.04)]),                                 # «Она умылась»
    (49.35, "crop", [C(30.96)]),                                 # «Вскоре ребёнок уснул»
    (51.00, "crop", [C(4.92)]),                                  # «В квартире наконец наступила тишина»
    (53.60, "crop", [C(16.00, 1.0)]),                            # «Но опыт подсказывал ей,
    (56.40, "crop", [C(22.72)]),                                 #   что радоваться раньше времени не стоит»
    (58.80, "col3", [Lt(6.88), Lt(25.00, .8), Lt(27.16, 1.6)]),  # «Потому что совсем скоро всё начнётся заново»
    (62.15, "crop", [C(35.68)]),                                 # «Бутылочки,
    (63.00, "crop", [C(36.48)]),                                 #   смесь,
    (63.65, "crop", [C(37.20)]),                                 #   кормление»
    (64.95, "crop", [C(38.00)]),                                 # «И вот наконец они вышли на прогулку»
    (68.10, "crop", [C(41.60)]),                                 # «Она купила себе второй кофе
    (69.50, "crop", [C(40.04)]),                                 #   и на несколько минут почувствовала,
    (71.60, "crop", [C(44.76)]),                                 #   что снова контролирует свою жизнь»
    (75.50, "col2", [Lt(42.60), Lt(43.72, .6)]),                 # «Разумеется, это была лишь иллюзия»
    (77.90, "crop", [C(47.60)]),                                 # «Ведь настоящей хозяйке этого дома
    (80.20, "col3", [Lt(28.16), Lt(24.08, .6), Lt(30.96, 1.2)]), #   ещё даже не исполнился год»
]
DUR = 84.6
SEGS = [dict(t0=a, t1=(EDL[k + 1][0] if k + 1 < len(EDL) else DUR), lay=lay, tiles=t) for k, (a, lay, t) in enumerate(EDL)]
NFR = int(round(DUR * FPS))
# every tile stays inside its source shot (shot boundaries from histogram cut detection, 25 fps proxy):
# speed = what the shot can give (≤1, ≥0.3); anything still missing is a freeze on the last frame
SHOTS = [0.0, 3.64, 4.12, 4.92, 5.88, 6.88, 7.88, 8.68, 10.48, 11.2, 12.24, 13.92, 14.76, 21.32, 22.72, 24.08, 25.0, 26.16, 27.16, 28.16,
         29.2, 30.04, 30.96, 31.96, 35.08, 35.68, 36.48, 37.2, 38.0, 40.04, 41.6, 42.6, 43.72, 44.76, 47.6, 48.64, 49.9]
for s_ in SEGS:
    nt = []
    for (src, sp, kind, cx, z, delay) in s_["tiles"]:
        vis = (s_["t1"] - s_["t0"]) - delay
        end = min(b for b in SHOTS if b > src + 0.01) - 0.08
        fit = min(1.0, max(0.3, (end - src) / vis))
        sp = fit if sp is None else min(sp, max(0.3, (end - src) / vis))
        nt.append((src, round(sp, 3), kind, cx, z, delay, end))
    s_["tiles"] = nt
for i, s in enumerate(SEGS):
    s["f0"], s["f1"] = int(round(s["t0"] * FPS)), int(round(s["t1"] * FPS)); s["i"] = i
def seg_at(tt): return next(s for s in SEGS if s["t0"] <= tt < s["t1"] or s is SEGS[-1])

def tsize(s, ti):
    """design size of tile ti in segment s (full frame for single shots; collages: 2 side by side / 1 big + 2 small)"""
    if s["lay"] == "crop": return W, H
    if s["lay"] == "col2": return 880, 495
    return (1120, 630) if ti == 0 else (660, 371)
def tile_path(si, ti): return f"tiles/s{si:02d}_{ti}.mkv"
def build_tiles(only=None):
    os.makedirs("tiles", exist_ok=True)
    for s in SEGS:
        if only is not None and s["i"] not in only: continue
        for ti, (src, sp, kind, cx, z, delay, end) in enumerate(s["tiles"]):
            n = s["f1"] - s["f0"] - int(round(delay * FPS))
            if kind == "c" and z == 1.0:
                geo = "null"                                         # native 3840x2160: no crop, no scale
            elif kind == "c":
                cw, ch = 3840 / z, 2160 / z; x0 = int(np.clip(cx * 3840 - cw / 2, 0, 3840 - cw)); y0 = int((2160 - ch) / 2)
                geo = f"crop={int(cw)}:{int(ch)}:{x0}:{y0},scale={OW}:{OH}:flags=lanczos"
            else:
                tw, th = tsize(s, ti); geo = f"scale={tw * K}:{th * K}:flags=lanczos"
            # slow motion: blend neighbouring source frames instead of duplicating them (smooth, no stutter)
            rate = f"framerate=fps={FPS}:interp_start=0:interp_end=255:scene=100" if sp < 0.95 else f"fps={FPS}"
            vf = f"{geo},setpts=(PTS-STARTPTS)/{sp},{rate}"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{src:.3f}", "-t", f"{min(n / FPS * sp + 0.3, end - src):.3f}", "-i", SRC, "-an",
                            "-vf", vf, "-frames:v", str(n), "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p", tile_path(s["i"], ti)], check=True)
        print("tiles", s["i"], [t[1] for t in s["tiles"]], flush=True)

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
        self.dark = self.img.copy(); self.dark[..., :3] = (COCOA if script else INK) * self.img[..., 3:4]   # on blush paper
    def active(self, tt): return self.t0 <= tt < self.t1
    def draw(self, dst, tt):
        age = tt - self.t0; img, sh = (self.dark, self.sh * 0.25) if ON_PAPER else (self.img, self.sh)
        if self.script:
            p = ease_io(age / 0.35)
            if p < 1:
                ww = img.shape[1]; ramp = np.clip((np.arange(ww) - p * ww * 1.15) / (-0.15 * ww), 0, 1)[None, :, None].astype(np.float32)
                img, sh = img * ramp, sh * ramp
            blend(dst, sh, self.x, self.y + 4); blend(dst, img, self.x, self.y); return
        sc = 1.0; al = min(1, age / 0.08)              # plain fade-in, no scale pulse
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


# subtitles: one word at a time from the narration's word timings, spelled exactly as in the script
SCRIPT = ("Каждое утро она надеялась, что сегодня всё пойдёт по плану. Но, как известно, в доме, где есть маленький ребёнок, планы строит "
          "только один человек. И это не мама. Поэтому утро начиналось с кофе. Единственного, что ещё связывало её с прежней жизнью. "
          "Затем наступало время бутылочек. Иногда ей казалось, что она моет их гораздо чаще, чем моется сама. Дальше был завтрак. "
          "Она уже давно перестала надеяться, что когда-нибудь съест его горячим. А у маленькой хозяйки дома тем временем было полно дел. "
          "Игрушки, погремушки и первые попытки исследовать этот огромный мир. И пока малышка была занята, её мама совершила практически "
          "невозможное. Она умылась. Вскоре ребёнок уснул. В квартире наконец наступила тишина. Но опыт подсказывал ей, что радоваться "
          "раньше времени не стоит. Потому что совсем скоро всё начнётся заново. Бутылочки, смесь, кормление. И вот наконец они вышли на "
          "прогулку. Она купила себе второй кофе и на несколько минут почувствовала, что снова контролирует свою жизнь. Разумеется, это "
          "была лишь иллюзия. Ведь настоящей хозяйке этого дома ещё даже не исполнился год.").split()
WORDS = []
for s, e, w in json.load(open("nwords.json")):
    w = w.strip()
    if w.startswith("-") and WORDS: WORDS[-1][2] += w; WORDS[-1][1] = e; continue      # «когда» + «-нибудь»
    WORDS.append([s, e, w])
def clean(w): return w.lower().strip(",.?!«»…").replace("ё", "е")
assert len(WORDS) == len(SCRIPT) and all(clean(a[2]) == clean(b) for a, b in zip(WORDS, SCRIPT)), "narration ≠ script"
HL = {"плану", "мама", "кофе", "бутылочек", "сама", "горячим", "хозяйки", "мир", "невозможное", "умылась", "тишина", "заново",
      "прогулку", "второй", "иллюзия", "год"}
OVER = []
for k, ((s, e, _), w) in enumerate(zip(WORDS, SCRIPT)):
    nxt = WORDS[k + 1][0] if k + 1 < len(WORDS) else e + 0.6
    OVER.append(Word(w.strip(",.?!…"), s - 0.03, min(nxt - 0.02, e + 0.55), script=clean(w) in HL))
OVER += [Label("мини влог", 0.15, 3.30, y=500, size=170, sub="с 4.5 месячным малышом")]

# ---------------------------------------------------------------- compositor
class Tiles:
    def __init__(self): self.si = None; self.decs = []; self.last = []
    def open(self, si):
        for d in self.decs:
            if d: d.kill()
        s = SEGS[si]; self.si = si; self.decs = []; self.last = []
        for ti in range(len(s["tiles"])):
            tw, th = tsize(s, ti); w, h = tw * K, th * K
            self.decs.append(subprocess.Popen(["ffmpeg", "-v", "error", "-i", tile_path(si, ti), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                              stdout=subprocess.PIPE, bufsize=w * h * 3))
            self.last.append(None)
    def get(self, si, fo):
        if si != self.si: self.open(si)
        s = SEGS[si]; out = []
        for ti, tl in enumerate(s["tiles"]):
            delay = tl[5]; tw, th = tsize(s, ti); w, h = tw * K, th * K
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

ON_PAPER = False
def render_frame(fo):
    global ON_PAPER
    tt = fo / FPS; s = seg_at(tt); si = s["i"]; imgs = TILES.get(si, fo); d = tt - s["t0"]
    lay = s["lay"]; ON_PAPER = lay.startswith("col")
    if lay == "crop":
        fr = imgs[0].copy()
    elif lay == "frame":
        fr = blurred_bg(imgs[0]); place_tile(fr, imgs[0], 540, 1000, d, slide=False)
    else:
        fr = paper(); n = len(imgs)
        xs, ys = ([610, 1540, 1540], [500, 290, 720]) if n == 3 else ([500, 1420], [500, 540])
        for ti, img in enumerate(imgs):
            if img is None: continue
            place_tile(fr, img, xs[ti], ys[ti], d - s["tiles"][ti][5])
    # cut accents: quick punch-in for crops, white flash on layout changes, whip blur into collages
    # (no punch-in zoom on cuts — removed on request)
    if si > 0 and SEGS[si - 1]["lay"] != lay and d < 0.14:
        a = 0.35 * (1 - d / 0.14); fr = cv2.addWeighted(fr, 1 - a, np.full_like(fr, 255), a, 0)
    if lay.startswith("col") and d < 0.1:
        k = int(120 * K * (1 - d / 0.1)) | 1; fr = cv2.blur(fr, (k, 1))
    for o in OVER:
        if o.active(tt): o.draw(fr, tt)
    if DUR - tt < 1.0: fr = (fr.astype(np.float32) * max(0, (DUR - tt) / 1.0)).astype(np.uint8)
    if tt < 0.2: fr = (fr.astype(np.float32) * (tt / 0.2)).astype(np.uint8)
    return fr

# ---------------------------------------------------------------- sound: narration, music bed, quiet foley
SR = 48000
def decode(path, extra=(), ss=None, t=None):
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", str(ss)] if ss is not None else []) + (["-t", str(t)] if t is not None else []) + \
          ["-i", path, "-vn", *extra, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).astype(np.float64)
def lufs(x):
    p = subprocess.run(["ffmpeg", "-nostats", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-af", "ebur128", "-f", "null", "-"],
                       input=x.astype(np.float32).tobytes(), capture_output=True).stderr.decode()
    return float(p.split("Summary:")[1].split("I:")[1].split("LUFS")[0])
def music(n):
    """theme looped on its own phrase repeat: 5.0→34.23, back to 10.68→34.23, back to 10.68→end (natural ending lands after «год»)"""
    x = decode("in/screen.mp4")
    J0, J1 = 34.23, 10.68
    pieces = [(5.0, J0), (J1, J0), (J1, 44.0)]
    out = np.zeros(0); xf = int(0.08 * SR)
    for a, b in pieces:
        p = x[int(a * SR):int(b * SR)].copy()
        if len(out):
            # align the splice to the best-matching phase (±15 ms) and crossfade equal-power
            ref = out[-xf:]; best, bo = -1e9, 0
            for o in range(-720, 721, 8):
                i0 = int(a * SR) + o; c = np.dot(ref, x[i0 - xf:i0]) if i0 - xf > 0 else -1e9
                if c > best: best, bo = c, o
            p = x[int(a * SR) + bo - xf:int(b * SR)].copy()
            g = np.sin(np.linspace(0, np.pi / 2, xf)); out[-xf:] = out[-xf:] * g[::-1] + p[:xf] * g; p = p[xf:]
        out = np.concatenate([out, p])
    out[:int(0.6 * SR)] *= np.linspace(0, 1, int(0.6 * SR))
    return np.pad(out, (0, max(0, n - len(out))))[:n]

def audio(out_path):
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
    # --- foley (synthesized, kept quiet under the story)
    def pour(d=1.0, f0=500, f1=1300):
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
    def whoosh(d=0.35, lo=600, hi=6000):
        m = int(d * SR); t = np.linspace(0, 1, m); x = rng.standard_normal(m); return norm(bp(x, lo, hi) * np.sin(np.pi * t) ** 2)
    fol = np.zeros(n)
    put(fol, pour(1.1, 450, 1250), 13.35, 0.30); put(fol, pour(1.0, 350, 900), 14.65, 0.22)
    put(fol, clink(), 16.9, 0.12); put(fol, sip(), 18.2, 0.10)
    put(fol, tap(1.6), 19.65, 0.20)
    put(fol, sizzle(2.1), 26.85, 0.16); put(fol, clink(3000), 32.6, 0.13)
    put(fol, rattle(1.3), 39.05, 0.16)
    for k in range(3): put(fol, splash(), 47.7 + k * 0.42, 0.22)
    m_ = int(1.3 * SR); tt_ = np.arange(m_) / SR
    machine = norm(bp(rng.standard_normal(m_), 140, 520) * (0.7 + 0.3 * np.sin(2 * np.pi * 50 * tt_))) * env(m_, 0.15, 0.3)
    put(fol, machine, 62.2, 0.09); put(fol, pour(0.6, 700, 1600), 63.0, 0.16)
    put(fol, outdoor(DUR - 64.95), 64.95, 0.09); put(fol, birds(DUR - 64.95), 64.95, 0.07)
    for s in SEGS[1:]:
        if s["lay"] != "crop" or SEGS[s["i"] - 1]["lay"] != "crop": put(fol, whoosh(), s["t0"] - 0.18, 0.05)
    # narration: the user's clips in story order, light corrective EQ only (they are already clean)
    plan = json.load(open("nar_plan.json")); v = np.zeros(n)
    for k in plan["order"]:
        c = decode(f"in/vo{k}.mp3", ["-af", "highpass=f=70,equalizer=f=220:t=q:w=1.0:g=-1.5"])
        put(v, c, plan["starts"][str(k)], 1.0)
    v *= 10 ** ((-16.0 - lufs(v)) / 20)
    mus = music(n); mus *= 10 ** ((-24.0 - lufs(mus)) / 20)
    # duck the music under the voice (smooth, ~6 dB), let it breathe in the gaps
    envv = np.convolve(np.abs(v), np.ones(4800) / 4800, "same"); envv = np.convolve(envv, np.ones(9600) / 9600, "same")
    duck = 1 - 0.45 * np.clip(envv / (0.5 * envv.max()), 0, 1)
    mix = v + mus * duck * 1.0 + fol * 0.9 * (1 - 0.3 * np.clip(envv / (0.5 * envv.max()), 0, 1))
    fo = int(1.2 * SR); mix[-fo:] *= np.linspace(1, 0, fo) ** 1.5
    g = 10 ** ((-15.0 - lufs(mix)) / 20); mix *= g
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-af", "alimiter=limit=0.89:level=false:attack=3:release=60",
                    "-c:a", "pcm_s16le", out_path], input=mix.astype(np.float32).tobytes(), check=True)
    print("mix", out_path, "peak before limiter", round(float(np.abs(mix).max()), 3), "LUFS", round(lufs(mix), 2))

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "tiles": build_tiles({int(x) for x in sys.argv[2].split(",")} if len(sys.argv) > 2 else None); sys.exit()
    if mode == "audio": audio(sys.argv[2]); sys.exit()
    if mode == "info":
        for s in SEGS: print(s["i"], round(s["t0"], 2), round(s["t1"], 2), s["lay"], [(t[0], t[1]) for t in s["tiles"]])
        print("DUR", DUR, "frames", NFR); sys.exit()
    if mode == "test":
        want = sorted(float(x) for x in sys.argv[2].split(","))
        for t in want:
            fo = int(round(t * FPS)); s = seg_at(t)
            TILES.open(s["i"])
            for f in range(s["f0"], fo): TILES.get(s["i"], f)      # advance decoders to the frame
            o = render_frame(fo)
            cv2.imwrite(f"test_{t:.2f}.png", cv2.cvtColor(cv2.resize(o, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        sys.exit()
    out = sys.argv[2]; s0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0; s1 = int(sys.argv[4]) if len(sys.argv) > 4 else len(SEGS)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-g", "100", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "5.2",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", out], stdin=subprocess.PIPE)
    f_end = NFR if s1 >= len(SEGS) else SEGS[s1]["f0"]
    for fo in range(SEGS[s0]["f0"], f_end): enc.stdin.write(render_frame(fo).tobytes())
    enc.stdin.close(); enc.wait(); print("done", out)
