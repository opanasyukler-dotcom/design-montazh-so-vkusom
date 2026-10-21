"""Reels 1080x1920 (~43 s): «Не хочу импланты. Не хочу подтяжку. Что тогда можно сделать?» — the same patient as
stories_breast_fat_only: marking (before), the operating room, the first result.  Plus a cover (JPG).

Voice pieces rA..rG are cut from the client's videos with tools/clean_speech.py (pauses, «ааа», «ага» removed, the
patient's voice changed), the subtitle text is the exact phrase put on their word timings (<piece>.subs.json,
tools/subs_retext.py).

Design: headlines Bebas Neue (bold, caps), everything else thin Montserrat (300); brand navy + white.
Text never lies on the body: the marking / surgeon shots sit in a rounded video card under the text band (blurred
copy of the frame behind), the result shots (body low in the frame) keep the full frame with the text on top.
Each scene has its own text entrance (drop, pop, typewriter, wipe, slide, blur-in, tracking-in) and an animation that
confirms the words (strike-through «импланты»/«подтяжку», marker on «липофилинг», жир → грудь, donor-site chips
ticked as they are named, asymmetry bars, a drawn check mark).  Sound effects are synthesised (whoosh, pop, tick,
swipe, ding) and mixed under the voice.

usage: python3 reels_no_implants_no_lift.py SRC_DIR OUT_DIR [cover|reel]
"""
import json, os, subprocess, sys, zlib
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, SR = 1080, 1920, 30, 48000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
BEBAS = os.path.join(ROOT, "skills/montage-svg-typography/Шрифты/Коллекция шрифтов/BebasNeueCyrillic.ttf")
WHITE, NAVY, GRAPHITE = (255, 255, 255), (2, 31, 71), (12, 14, 20)
SRC, OUT = sys.argv[1], sys.argv[2]
WHAT = sys.argv[3:] or ["cover", "reel"]
DRIVE = os.path.join(SRC, "drive")
os.makedirs(OUT, exist_ok=True)
BAND_TOP = 200            # Instagram's top bar above; the caption and buttons cover the bottom ~300 px


def mont(size, wght=300):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([wght]); return f


def bebas(size): return ImageFont.truetype(BEBAS, size)


def clamp(x): return min(max(x, 0.0), 1.0)
def ease(x): x = clamp(x); return 1 - (1 - x) ** 3
def back(x, s=1.7): x = clamp(x); x -= 1; return 1 + (s + 1) * x ** 3 + s * x ** 2      # overshoot


def wrap(text, f, width):
    out, line = [], ""
    for w in text.split(" "):
        t = (line + " " + w).strip()
        if f.getlength(t) <= width or not line: line = t
        else: out.append(line); line = w
    return out + [line]


def rgba(im): return np.asarray(im, np.float32) / 255.0


def blit(fr, a, x, y, k=1.0):
    """alpha-composite float RGBA `a` onto float RGB frame at (x, y), extra opacity k"""
    if k <= 0.003: return
    h, w = a.shape[:2]; x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    s = a[y0 - y:y1 - y, x0 - x:x1 - x]; al = s[..., 3:4] * k
    fr[y0:y1, x0:x1] = fr[y0:y1, x0:x1] * (1 - al) + s[..., :3] * al


def scaled(a, sc):
    if abs(sc - 1) < 0.01: return a
    return cv2.resize(a, None, fx=max(sc, 0.05), fy=max(sc, 0.05), interpolation=cv2.INTER_LINEAR)


def shadowed(im, r=8, a=140):
    sh = Image.new("RGBA", im.size, GRAPHITE + (0,))
    sh.putalpha(im.getchannel("A").point(lambda v: v * a // 255))
    return Image.alpha_composite(sh.filter(ImageFilter.GaussianBlur(r)), im)


def text_sprite(text, f, col=WHITE, pad=20, shadow=True):
    w = int(f.getlength(text)) + 2 * pad; asc, desc = f.getmetrics(); h = asc + desc + 2 * pad
    im = Image.new("RGBA", (max(w, 1), h), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((pad, pad), text, font=f, fill=col + (255,))
    return rgba(shadowed(im) if shadow else im), pad


def pill(text, f, fill=NAVY, padx=26, h=None, check=False, alpha=235):
    h = h or int(f.size * 1.8); cw = int(h * 0.55) if check else 0
    w = int(f.getlength(text)) + 2 * padx + cw
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), h // 2, fill=fill + (alpha,))
    d.text((padx + cw + (w - 2 * padx - cw) / 2, h / 2 + 1), text, font=f, fill=WHITE + (255,), anchor="mm")
    if check:                                                # thin white tick drawn as lines
        x0, y0, s = padx - 4, h / 2, h * 0.17
        d.line([(x0, y0), (x0 + s * 0.8, y0 + s), (x0 + s * 2.4, y0 - s * 1.1)], fill=WHITE + (255,), width=4, joint="curve")
    return rgba(im)


# ---------------- band elements: each has a height and draw(fr, t, x, y) ----------------
class Headline:
    """Bebas caps, centred lines; per-character sprites so every entrance can move letters / words / lines.
    fx: drop | pop | type | wipe | slide | slide_alt | blur | track.  strike=(word, t): a navy bar crosses the word;
    marker=(word, t): a navy band grows behind the word.  out=t: fades out (a second phase takes the slot)"""
    def __init__(self, lines, size, fx, t0=0.0, out=None, strike=None, marker=None, gap=0.98):
        self.f = bebas(size); self.size = size; self.fx, self.t0, self.out = fx, t0, out
        self.strike, self.marker = strike, marker
        self.step = int(size * gap); self.h = self.step * len(lines) + int(size * 0.12)
        self.lines = []
        for ln in lines:
            lw = self.f.getlength(ln); x0 = (W - lw) / 2; chars = []; word = 0
            for i, ch in enumerate(ln):
                if ch == " ": word += 1; continue
                sp, pad = text_sprite(ch, self.f, pad=18)
                chars.append(dict(a=sp, x=x0 + self.f.getlength(ln[:i]) - pad, pad=pad, word=word, i=i))
            self.lines.append(dict(text=ln, x0=x0, w=lw, chars=chars, words=word + 1))

    def word_span(self, word):
        for li, ln in enumerate(self.lines):
            k = ln["text"].find(word)
            if k >= 0: return li, ln["x0"] + self.f.getlength(ln["text"][:k]), ln["x0"] + self.f.getlength(ln["text"][:k + len(word)])

    def draw(self, fr, t, y):
        if self.out is not None and t > self.out + 0.25: return
        kout = 1 - clamp((t - self.out) / 0.22) if self.out is not None else 1.0
        if self.marker:                                      # grows from the left, behind the letters
            word, tm = self.marker; li, xa, xb = self.word_span(word); p = ease((t - tm) / 0.35)
            if p > 0:
                yy = y + li * self.step + int(self.size * 0.02); hh = int(self.size * 0.86)
                band = np.zeros((hh, int(xb - xa) + 36, 4), np.float32); band[..., :3] = np.array(NAVY) / 255; band[..., 3] = 0.95
                blit(fr, band[:, :max(1, int(band.shape[1] * p))], xa - 18, yy, kout)
        nchar = sum(len(l["chars"]) for l in self.lines); ci = 0
        for li, ln in enumerate(self.lines):
            ly = y + li * self.step - 18 - int(self.size * 0.08)
            for c in ln["chars"]:
                a, x, yy, k = c["a"], c["x"], ly, 1.0
                if self.fx == "drop":
                    tt = self.t0 + 0.028 * ci; p = back((t - tt) / 0.38); k = clamp((t - tt) / 0.15); yy -= 70 * (1 - p)
                elif self.fx == "pop":
                    tt = self.t0 + 0.13 * (c["word"] + 2 * li); p = back((t - tt) / 0.34, 2.2); k = clamp((t - tt) / 0.12)
                    a = scaled(a, 0.55 + 0.45 * p); x += (c["a"].shape[1] - a.shape[1]) / 2; yy += (c["a"].shape[0] - a.shape[0]) / 2
                elif self.fx == "type":
                    k = 1.0 if t >= self.t0 + 0.034 * ci else 0.0
                elif self.fx == "wipe":
                    tt = self.t0 + 0.16 * li; p = ease((t - tt) / 0.45)
                    edge = ln["x0"] + p * (ln["w"] + 60) - 30; cx = c["x"] + c["pad"]
                    k = clamp((edge - cx) / 40)
                elif self.fx == "slide":
                    tt = self.t0 + 0.07 * (c["word"] + 3 * li); p = ease((t - tt) / 0.4); k = p; x -= 90 * (1 - p)
                elif self.fx == "slide_alt":
                    tt = self.t0 + 0.18 * li; p = ease((t - tt) / 0.42); k = p; x += (-1 if li % 2 == 0 else 1) * 260 * (1 - p)
                elif self.fx == "blur":
                    tt = self.t0 + 0.15 * li; p = ease((t - tt) / 0.5); k = p
                    s = 14 * (1 - p)
                    if s > 0.4: a = cv2.GaussianBlur(a, (0, 0), s)
                elif self.fx == "track":
                    p = ease((t - self.t0) / 0.7); k = p
                    x += (c["i"] - len(ln["text"]) / 2) * 26 * (1 - p)
                if self.out is not None: yy -= 30 * (1 - kout)
                blit(fr, a, x, yy, k * kout); ci += 1
        if self.strike:
            word, ts = self.strike; li, xa, xb = self.word_span(word); p = ease((t - ts) / 0.28)
            if p > 0:
                yy = y + li * self.step + int(self.size * 0.36); L = int((xb - xa + 40) * p)
                bar = np.zeros((14, max(L, 1), 4), np.float32); bar[..., 3] = 1
                bar[3:11, :, :3] = np.array(NAVY) / 255; bar[:3, :, :3] = 1; bar[11:, :, :3] = 1
                blit(fr, bar, xa - 20, yy, kout)


class Line:
    """thin Montserrat, centred, fade-up (optionally fade-out for a second phase)"""
    def __init__(self, text, size=40, t0=0.0, out=None, width=900):
        f = mont(size, 300); self.items = [text_sprite(l, f, pad=16) for l in wrap(text, f, width)]
        self.step = int(size * 1.32); self.h = self.step * len(self.items); self.t0, self.out = t0, out

    def draw(self, fr, t, y):
        p = ease((t - self.t0) / 0.4)
        if self.out is not None: p *= 1 - clamp((t - self.out) / 0.22)
        for i, (a, pad) in enumerate(self.items):
            blit(fr, a, (W - a.shape[1]) / 2, y + i * self.step - pad + 22 * (1 - p), p)


class Chip:
    """navy pill; pop with overshoot; pulse=True: a slight heartbeat (ВАЖНО)"""
    def __init__(self, text, f, t0=0.0, pulse=False, check=False, h=None):
        self.a = pill(text, f, check=check, h=h); self.h = self.a.shape[0]; self.t0, self.pulse = t0, pulse

    def draw_at(self, fr, t, cx, y):
        p = back((t - self.t0) / 0.34, 2.4)
        if p <= 0: return
        sc = 0.4 + 0.6 * p
        if self.pulse and t > self.t0 + 0.4: sc *= 1 + 0.06 * max(0.0, np.sin((t - self.t0 - 0.4) * 2 * np.pi / 1.1)) ** 4
        a = scaled(self.a, sc); blit(fr, a, cx - a.shape[1] / 2, y + (self.h - a.shape[0]) / 2, clamp((t - self.t0) / 0.1))

    def draw(self, fr, t, y): self.draw_at(fr, t, W / 2, y)


class ChipRow:
    """chips in one centred row, each pops (ticked) when the word is said"""
    def __init__(self, items):
        f = mont(38, 300); self.chips = [Chip(txt, f, t0, check=True, h=74) for txt, t0 in items]
        self.h = 74; self.gap = 16
        tot = sum(c.a.shape[1] for c in self.chips) + self.gap * (len(self.chips) - 1); x = (W - tot) / 2; self.cx = []
        for c in self.chips: self.cx.append(x + c.a.shape[1] / 2); x += c.a.shape[1] + self.gap

    def draw(self, fr, t, y):
        for c, cx in zip(self.chips, self.cx): c.draw_at(fr, t, cx, y)


class Flow:
    """[жир] ——→ [грудь]: the two pills pop, the arrow draws itself between them"""
    def __init__(self, a, b, t0):
        f = mont(36, 300); self.A, self.B = Chip(a, f, t0, h=68), Chip(b, f, t0 + 0.55, h=68); self.t0 = t0; self.h = 68

    def draw(self, fr, t, y):
        if t < self.t0: return
        wa, wb, L = self.A.a.shape[1], self.B.a.shape[1], 230
        x0 = (W - (wa + L + wb)) / 2
        self.A.draw_at(fr, t, x0 + wa / 2, y); self.B.draw_at(fr, t, x0 + wa + L + wb / 2, y)
        p = ease((t - self.t0 - 0.2) / 0.4)
        if p > 0:
            ov = np.zeros((self.h, L, 4), np.float32); xe = int(20 + (L - 50) * p); yc = self.h // 2
            cv2.line(ov, (20, yc), (xe, yc), (1, 1, 1, 1), 4, cv2.LINE_AA)
            if p > 0.6: cv2.line(ov, (xe, yc), (xe - 16, yc - 12), (1, 1, 1, 1), 4, cv2.LINE_AA); cv2.line(ov, (xe, yc), (xe - 16, yc + 12), (1, 1, 1, 1), 4, cv2.LINE_AA)
            blit(fr, ov, x0 + wa, y)


class Bars:
    """asymmetry: two thin bars grow to different lengths as «побольше» / «поменьше» are said"""
    def __init__(self, rows):
        self.f = mont(35, 300); self.rows = [(text_sprite(lbl, self.f, pad=10), t0, t1, ln) for lbl, t0, t1, ln in rows]
        self.h = 2 * 92

    def draw(self, fr, t, y):
        x0 = 150
        for r, ((lab, pad), t0, t1, ln) in enumerate(self.rows):
            k = ease((t - t0) / 0.3)
            if k <= 0: continue
            yy = y + r * 92; blit(fr, lab, x0 - pad, yy - pad, k)
            p = 0.08 + 0.92 * back((t - t1) / 0.5, 1.2) if t > t1 else 0.08 * k
            L = int(ln * p); bar = np.zeros((20, max(L, 1), 4), np.float32); bar[..., :3] = 1; bar[..., 3] = 0.95
            cv2.rectangle(bar, (0, 0), (max(L, 1) - 1, 19), (1, 1, 1, 1), -1)
            track = np.zeros((20, 780, 4), np.float32); track[..., :3] = 1; track[..., 3] = 0.18
            blit(fr, track, x0, yy + 46, k); blit(fr, bar, x0, yy + 46, k)


class CheckMark:
    """a circle draws itself, then the tick (strokes, like SVG strokeDashoffset)"""
    def __init__(self, t0, size=120): self.t0, self.s, self.h = t0, size, size

    def draw(self, fr, t, y):
        p = ease((t - self.t0) / 0.5); q = ease((t - self.t0 - 0.45) / 0.3)
        if p <= 0: return
        S = self.s; ov = np.zeros((S, S, 4), np.float32); c = S // 2
        cv2.ellipse(ov, (c, c), (c - 5, c - 5), -90, 0, 360 * p, (1, 1, 1, 1), 5, cv2.LINE_AA)
        if q > 0:
            pts = [(int(S * 0.29), int(S * 0.52)), (int(S * 0.44), int(S * 0.67)), (int(S * 0.72), int(S * 0.36))]
            l1 = min(q / 0.4, 1); a, b = pts[0], pts[1]
            cv2.line(ov, a, (int(a[0] + (b[0] - a[0]) * l1), int(a[1] + (b[1] - a[1]) * l1)), (1, 1, 1, 1), 6, cv2.LINE_AA)
            if q > 0.4:
                l2 = (q - 0.4) / 0.6; a, b = pts[1], pts[2]
                cv2.line(ov, a, (int(a[0] + (b[0] - a[0]) * l2), int(a[1] + (b[1] - a[1]) * l2)), (1, 1, 1, 1), 6, cv2.LINE_AA)
        blit(fr, ov, (W - S) / 2, y)


class Slot:
    """phases sharing one place in the band (the second replaces the first)"""
    def __init__(self, *els): self.els = els; self.h = max(e.h for e in els)
    def draw(self, fr, t, y):
        for e in self.els: e.draw(fr, t, y + (self.h - e.h) / 2)


# ---------------- subtitles ----------------
class PillSubs:
    """one word at a time, lower case, thin Montserrat on a navy pill, small pop-in"""
    def __init__(self, words, size=40):
        f = mont(size, 300); self.items = []
        for i, (a, b, t) in enumerate(words):
            t = t.lower().strip(".,!?«»…")
            e = min(words[i + 1][0] if i + 1 < len(words) else b + 0.5, b + 0.45)
            self.items.append((a, e, pill(t, f, padx=22)))
        self.h = self.items[0][2].shape[0] if self.items else 0

    def draw(self, fr, t, y):
        for a, e, arr in self.items:
            if a - 0.03 <= t < e:
                k = clamp((t - a + 0.03) / 0.12); a2 = scaled(arr, 0.85 + 0.15 * k)
                blit(fr, a2, (W - a2.shape[1]) / 2, y + (arr.shape[0] - a2.shape[0]) / 2, k); return


# ---------------- video ----------------
class Video:
    def __init__(self, path, a, dur):
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(dur + 0.2), "-i", path, "-vf", f"scale={W}:{H}:flags=lanczos,fps={FPS}",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.last = None

    def frame(self):
        buf = self.p.stdout.read(W * H * 3)
        if len(buf) == W * H * 3: self.last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        return self.last

    def close(self): self.p.kill()


def rounded_mask(w, h, r):
    m = np.zeros((h, w), np.uint8); cv2.rectangle(m, (r, 0), (w - r, h), 255, -1); cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in [(r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)]: cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return (m.astype(np.float32) / 255)[..., None]


_CARD = {}


def card_frame(src, y0, lt, zoom):
    """blurred, darkened copy of the frame behind; the frame itself in a rounded card from y0 to the bottom.
    The card eases in (scale 0.94 -> 1) at the start of each scene; a slow push-in keeps it alive."""
    small = cv2.resize(src, (W // 10, H // 10), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 4), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 * 0.42
    ch = H - 30 - y0; cw = int(min(ch * 9 / 16, 940)); ch = int(cw * 16 / 9)
    s = 0.94 + 0.06 * ease(lt / 0.35); cw2, ch2 = int(cw * s), int(ch * s)
    z = zoom
    M = cv2.getRotationMatrix2D((W / 2, H * 0.45), 0, z)
    src = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    card = cv2.resize(src, (cw2, ch2), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x, y = (W - cw2) // 2, y0 + (ch - ch2) // 2
    key = (x, y, cw2, ch2)
    if key not in _CARD:                                       # mask + soft drop shadow, per card size
        sh = np.zeros((H // 4, W // 4), np.float32)
        cv2.rectangle(sh, ((x + 6) // 4, (y + 18) // 4), ((x + cw2 - 6) // 4, min(y + ch2 + 18, H - 1) // 4), 1, -1)
        sh = cv2.resize(cv2.GaussianBlur(sh, (0, 0), 5.5), (W, H))[..., None] * 0.55
        _CARD[key] = (rounded_mask(cw2, ch2, 34), sh)
    m, sh = _CARD[key]
    bg = bg * (1 - sh)
    reg = bg[y:y + ch2, x:x + cw2]; bg[y:y + ch2, x:x + cw2] = reg * (1 - m) + card[:reg.shape[0]] * m
    return bg


def full_frame(src, shift, lt, zoom):
    """the whole frame, full width, moved down by `shift` so the body sits under the text band; above it a blurred
    stretch of the frame's own top that melts into it.  Each scene starts with a short punch-in (1.06 -> 1)"""
    z = zoom + 0.06 * (1 - ease(lt / 0.3))
    M = cv2.getRotationMatrix2D((W / 2, H * 0.5), 0, z); M[1, 2] += shift
    fr = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32) / 255
    top = int(M[1, 2] + 0 * M[1, 1])                          # first row of the picture
    if top > 0:
        strip = cv2.resize(src[:max(8, H // 6)], (W // 10, max(2, top // 10 + 1)), interpolation=cv2.INTER_AREA)
        fill = cv2.resize(cv2.GaussianBlur(strip, (0, 0), 3), (W, top + 120)).astype(np.float32) / 255 * 0.6
        g = np.clip((np.arange(top + 120) - top) / 120.0, 0, 1)[:, None, None]
        fr[:top + 120] = fill * (1 - g) + fr[:top + 120] * g
    return fr


def top_fade(bottom, a=0.85):
    ys = np.arange(H, dtype=np.float32); p = np.clip((bottom + 160 - ys) / 360, 0, 1); p = p * p * (3 - 2 * p)
    return (a * p)[:, None, None]


# ---------------- sound effects (synthesised) ----------------
def _env(n, att, dec):
    e = np.ones(n, np.float32); a = max(1, int(att * SR)); e[:a] = np.linspace(0, 1, a)
    e[a:] = np.exp(-np.arange(n - a) / (dec * SR)); return e


def _lp(x, a):
    """one-pole low-pass, a: per-sample coefficient (array or scalar)"""
    a = np.broadcast_to(np.asarray(a, np.float32), x.shape); out = np.empty_like(x); y = 0.0
    for k in range(len(x)): y += a[k] * (x[k] - y); out[k] = y
    return out


def _tone(f, n, dec, harm=((1, 1.0),)):
    tt = np.arange(n) / SR
    return sum(g * np.sin(2 * np.pi * f * h * tt) * np.exp(-tt * h / dec) for h, g in harm).astype(np.float32)


def sfx(kind):
    """riser: airy swell that ends on the cut; hit: soft sub thump + muffled body (scene change); tap: soft UI click;
    swoosh: short paper-like air (strike, marker, arrow); shimmer: bright sparkle (typing); marimba: one wooden note
    (ВАЖНО); success: two rising marimba notes (final check)"""
    rng = np.random.default_rng(zlib.crc32(kind.encode()))
    if kind == "riser":
        n = int(0.5 * SR); x = rng.standard_normal(n).astype(np.float32); r = np.linspace(0, 1, n)
        y = _lp(x, 0.01 + 0.2 * r ** 2) - _lp(x, 0.004)
        return y * r ** 2.5 * 0.55
    if kind == "hit":
        n = int(0.45 * SR); tt = np.arange(n) / SR; f = 38 + 60 * np.exp(-tt * 30)
        sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 9)
        body = _lp(rng.standard_normal(n).astype(np.float32), 0.05) * np.exp(-tt * 40)
        return (sub * 0.7 + body * 0.5).astype(np.float32)
    if kind == "tap":
        n = int(0.06 * SR); tt = np.arange(n) / SR
        return (np.sin(2 * np.pi * 1450 * tt) * np.exp(-tt * 110) * 0.28 + _lp(rng.standard_normal(n).astype(np.float32), 0.3) * np.exp(-tt * 400) * 0.15).astype(np.float32)
    if kind == "swoosh":
        n = int(0.22 * SR); x = rng.standard_normal(n).astype(np.float32); r = np.linspace(0, 1, n)
        y = _lp(x, 0.35) - _lp(x, 0.08)
        return y * np.sin(np.pi * r) ** 1.5 * 0.5
    if kind == "shimmer":
        n = int(0.6 * SR); out = np.zeros(n, np.float32)
        for k, f in enumerate([2093, 2637, 3136, 3951, 3136, 4186]):
            j = int(k * 0.07 * SR); b = _tone(f, n - j, 0.09); out[j:] += b * 0.07
        return out
    if kind == "marimba":
        return _tone(392, int(0.9 * SR), 0.22, ((1, 1.0), (4, 0.25), (10, 0.06))) * 0.4
    if kind == "success":
        a = _tone(659.3, int(0.9 * SR), 0.2, ((1, 1.0), (4, 0.25))); b = _tone(987.8, int(0.9 * SR), 0.25, ((1, 1.0), (4, 0.25)))
        out = np.zeros(int(1.05 * SR), np.float32); out[:len(a)] += a * 0.35; k = int(0.13 * SR); out[k:k + len(b)] += b * 0.35
        return out


# ---------------- scenes ----------------
# piece: voice clip (its own picture unless `picture`); card=True: picture in the card under the band;
# hold: frozen last frame (silence); extra: silent footage after the voice; tail: picture continues after the voice
def scenes():
    M = mont
    return [
        dict(piece="rA.mp4", card=False,
             band=[Chip("ДО", bebas(46), 0.15, h=64),
                   Headline(["ИМПЛАНТЫ НЕ ХОЧУ"], 124, "drop", strike=("ИМПЛАНТЫ", 0.75))],
             sfx=[(0, "hit"), (0.15, "tap"), (0.75, "swoosh")]),
        dict(piece="rB.mp4", card=False, hold=0.7,
             band=[Slot(Headline(["ПОДТЯЖКУ", "ТОЖЕ НЕ ХОЧУ"], 112, "pop", out=4.0, strike=("ПОДТЯЖКУ", 3.15)),
                        Headline(["ТОГДА ЧТО", "МОЖНО СДЕЛАТЬ?"], 112, "type", t0=4.2))],
             sfx=[(0, "hit"), (0.05, "tap"), (0.3, "tap"), (3.15, "swoosh"), (4.2, "shimmer")]),
        dict(piece="rC.mp4", card=False, shift=180,
             band=[Headline(["ВЫБРАЛИ", "ЛИПОФИЛИНГ"], 116, "wipe", marker=("ЛИПОФИЛИНГ", 0.85)),
                   Slot(Line("собственная жировая ткань", 40, t0=1.5, out=5.6), Flow("жир", "грудь", 5.85))],
             sfx=[(0, "hit"), (0.85, "swoosh"), (5.85, "tap"), (6.05, "swoosh"), (6.4, "tap")]),
        dict(piece="rD.mp4", card=True, extra=("v1.mp4", 48.35, 1.8),
             band=[Headline(["ДАЖЕ У СТРОЙНОЙ ПАЦИЕНТКИ", "МОЖНО НАЙТИ ДОНОРСКИЕ ЗОНЫ"], 84, "slide"),
                   ChipRow([("внутренние бёдра", 0.1), ("колени", 1.45), ("галифе", 2.5)])],
             sfx=[(0, "hit"), (0.1, "tap"), (1.45, "tap"), (2.5, "tap")]),
        dict(piece="rE.mp4", card=False, shift=170, picture=("v5.mp4", 1.0),
             band=[Headline(["ОБЪЁМ ДОБАВЛЯЕМ", "НЕ ВСЕГДА ОДИНАКОВО"], 104, "blur"),
                   Line("можно учитывать асимметрию", 38, t0=0.9),
                   Bars([("с одной стороны · побольше", 2.8, 4.3, 700), ("с другой · поменьше", 4.8, 5.13, 380)])],
             sfx=[(0, "hit"), (0.05, "swoosh"), (4.3, "tap"), (5.13, "tap")]),
        dict(piece="rF.mp4", card=True,
             band=[Chip("ВАЖНО", bebas(50), 0.0, pulse=True, h=70),
                   Headline(["ЛИПОФИЛИНГ", "НЕ ЗАМЕНЯЕТ ПОДТЯЖКУ"], 100, "slide_alt", t0=0.25),
                   Line("но может помочь добавить наполненность, если пациентка не готова к более объёмной операции", 34, t0=4.4, width=880)],
             sfx=[(0, "hit"), (0.02, "marimba"), (0.25, "swoosh"), (4.4, "swoosh")]),
        dict(piece="rG.mp4", card=False, shift=170, picture=("v5.mp4", 7.4), tail=2.0,
             band=[CheckMark(0.15), Headline(["ИМЕННО ЭТУ ЗАДАЧУ", "И РЕШАЛИ"], 104, "track", t0=0.1),
                   Chip("Подробнее в актуальном «Липофилинг»", M(36, 300), 3.6, h=72)],
             sfx=[(0, "hit"), (0.75, "success"), (3.6, "tap")]),
    ]


def dur_of(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True).stdout)


def pcm(path):
    return np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                                        capture_output=True, check=True).stdout, np.float32)


def layout(s):
    """stack the band from BAND_TOP; subtitles under it; the card (or the free full frame) below"""
    y = BAND_TOP; s["ys"] = []
    for e in s["band"]: s["ys"].append(y); y += e.h + 26
    s["sub_y"] = y + 4; y += s["subs"].h + 30
    s["card_y"] = y


def reel():
    sc = scenes(); t = 0.0
    for s in sc:
        s["voice"] = dur_of(os.path.join(DRIVE, s["piece"]))
        s["dur"] = s["voice"] + s.get("hold", 0) + (s["extra"][2] if "extra" in s else 0) + s.get("tail", 0)
        s["t0"] = t; t += s["dur"]
        words = json.load(open(os.path.join(DRIVE, s["piece"] + ".subs.json")))[0]["w"]
        s["subs"] = PillSubs(words); layout(s)
        if not s["card"]: s["shade"] = top_fade(s["card_y"] - 30)
    total = t; print(f"reel {total:.1f}s", flush=True)
    tmp = os.path.join(OUT, "reel.v.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for si, s in enumerate(sc):
        pic = s.get("picture")
        main = Video(os.path.join(DRIVE, pic[0]), pic[1], s["voice"] + s.get("tail", 0)) if pic else Video(os.path.join(DRIVE, s["piece"]), 0.0, s["voice"])
        ex = Video(os.path.join(DRIVE, s["extra"][0]), s["extra"][1], s["extra"][2]) if "extra" in s else None
        nv = int(round((s["voice"] + s.get("tail", 0)) * FPS)); nh = int(round(s.get("hold", 0) * FPS)); n = int(round(s["dur"] * FPS))
        last = None
        for i in range(n):
            lt = i / FPS
            if i < nv: src = main.frame(); last = src
            elif i < nv + nh: src = last
            else: src = ex.frame()
            zoom = 1.0 + 0.04 * i / max(n, 1)
            if s["card"]: fr = card_frame(src, s["card_y"], lt, zoom)
            else: fr = full_frame(src, s.get("shift", s["card_y"] - 50), lt, zoom) * (1 - s["shade"])
            for e, y in zip(s["band"], s["ys"]): e.draw(fr, lt, y)
            s["subs"].draw(fr, lt, s["sub_y"])
            o = (np.clip(fr, 0, 1) * 255).astype(np.uint8); enc.stdin.write(o.tobytes())
            if i == min(n - 1, int(s.get("preview", min(s["voice"] - 0.3, 5.5)) * FPS)):
                Image.fromarray(o).save(os.path.join(OUT, f"kadr_{si + 1}.jpg"), quality=90)
        main.close()
        if ex: ex.close()
    enc.stdin.close(); enc.wait()
    # voice (silence under held / extra / tail footage) + sound effects under it
    voice = np.zeros(int(total * SR) + SR, np.float32); fx = np.zeros_like(voice)
    for s in sc:
        a = pcm(os.path.join(DRIVE, s["piece"]))[:int(s["voice"] * SR)]; i0 = int(s["t0"] * SR); voice[i0:i0 + len(a)] = a
        for tt, kind in s["sfx"] + ([(-0.5, "riser")] if s["t0"] > 0 else []):
            x = sfx(kind); j = int((s["t0"] + tt) * SR); fx[j:j + len(x)] += x[:len(fx) - j]
    voice /= max(1e-6, np.abs(voice).max()); mix = voice * 0.9 + fx * 0.55
    mix = mix[:int(total * SR)] / max(1.0, np.abs(mix).max() / 0.98)
    wav = os.path.join(OUT, "reel.f32"); mix.astype(np.float32).tofile(wav)
    out = os.path.join(OUT, "reels.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", wav,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5", "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", "-shortest", "-movflags", "+faststart", out], check=True)
    os.remove(tmp); os.remove(wav); print("wrote", out, flush=True)


def cover():
    """same card layout as the reel: Bebas title on top (inside the central 3:4 zone y 240..1680), the before body in
    the card below, nothing written over the body"""
    fr = subprocess.run(["ffmpeg", "-v", "error", "-ss", "20.0", "-i", os.path.join(DRIVE, "v2.mp4"), "-frames:v", "1",
                         "-vf", f"scale={W}:{H}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    src = np.frombuffer(fr, np.uint8).reshape(H, W, 3)
    y = 255; band = [Headline(["НЕ ХОЧУ ИМПЛАНТЫ.", "НЕ ХОЧУ ПОДТЯЖКУ"], 128, "type", t0=-1), Line("что можно сделать с грудью?", 44)]
    ys = []
    for e in band: ys.append(y); y += e.h + 22
    img = card_frame(src, y + 14, 1.0, 1.0)
    for e, yy in zip(band, ys): e.draw(img, 5.0, yy)
    out = os.path.join(OUT, "oblozhka.jpg"); Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out, quality=95)
    print("wrote", out)


if "cover" in WHAT: cover()
if "reel" in WHAT: reel()
