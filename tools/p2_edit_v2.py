"""«Я не люблю материнство. Часть 2» v2 — plain white text, visual inserts instead of cards/badges,
light skin retouch, no final card. Reuses the cut + timing from p2_edit.py (same work dir: cut60.mp4, cuts.json, words.json).
  python3 -I p2_edit_v2.py test 3,33,60      -> test_*.png
  python3 -I p2_edit_v2.py render f0 f1 out.mp4
  python3 -I p2_edit_v2.py sfx sfx.wav
"""
import subprocess, sys, wave
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # p2_edit.py sits next to this script
import numpy as np, cv2
sys.argv_backup = sys.argv; sys.argv = [sys.argv[0], "lib"]
import p2_edit as B
sys.argv = sys.argv_backup
from PIL import Image, ImageDraw

W, H, K, FPS, OW, OH = B.W, B.H, B.K, B.FPS, B.OW, B.OH
ease, ease_io, back, P, blend, font, raster = B.ease, B.ease_io, B.back, B.P, B.blend, B.font, B.raster
INK, WHITE = B.INK, B.WHITE
DUR = B.SPEECH_END                     # no freeze / final card any more
NFR = sum(B.NF)
ws, we, sec_end, STARTS, CUTS = B.ws, B.we, B.sec_end, B.STARTS, B.CUTS
FONT_M = B.FONT_M
BLACK = np.float32([0, 0, 0])

# ---------------------------------------------------------------- plain white text
class Line:
    """one white line with a soft dark shadow; anim: rise (blur-rise) or pop"""
    def __init__(self, txt, size, wght, x, y, t0, t1, align="c", anim="rise"):
        f = font(FONT_M, int(size * K), wght)
        self.img, self.sh, base, pad, adv = raster(txt, f, WHITE, halo=BLACK, halo_r=7, halo_a=0.55)
        x0 = x - (adv / 2 if align == "c" else adv if align == "r" else 0)
        self.x, self.y, self.t0, self.t1, self.anim = x0 - pad, y - base, t0, t1, anim
    def active(self, tt): return self.t0 <= tt <= self.t1 + 0.3
    def draw(self, dst, tt):
        age = tt - self.t0; out = ease((tt - self.t1) / 0.25) if tt > self.t1 else 0
        img, sh = self.img, self.sh; dy = -16 * out; dx = 0; al = min(1, age / 0.18) * (1 - out)
        if self.anim == "pop":
            sc = 0.7 + 0.3 * back(age / 0.32)
            if abs(sc - 1) > 0.01:
                hh, ww = img.shape[:2]; nw, nh = max(2, int(ww * sc)), max(2, int(hh * sc))
                img, sh = cv2.resize(img, (nw, nh)), cv2.resize(sh, (nw, nh)); dx, dy = (ww - nw) / 2 / K, (hh - nh) / 2 / K + dy
        else:
            e = ease(age / 0.35); dy += 30 * (1 - e); k = int(20 * K * (1 - e)) | 1
            if k > 2: img, sh = cv2.blur(img, (1, k)), cv2.blur(sh, (1, k))
        blend(dst, sh, self.x + dx, self.y + dy, al); blend(dst, img, self.x + dx, self.y + dy, al)

# subtitles: white Montserrat, accents just heavier — no colours, boxes or handles
S = 50
class SubWord:
    def __init__(self, txt, heavy):
        f = font(FONT_M, int((S * 1.12 if heavy else S) * K), 800 if heavy else 600)
        self.img, self.sh, self.base, self.pad, self.adv = raster(txt, f, WHITE, halo=BLACK, halo_r=6, halo_a=0.6)
        self.space = font(FONT_M, S * K, 600).getlength(" ") / K
class Sub:
    def __init__(self, lines, t_in, t_out):
        self.t_in, self.t_out, self.words = t_in, t_out, []
        for ln in lines:
            ws_ = [SubWord(t, h) for t, h in ln["words"]]
            total = sum(w.adv for w in ws_) + sum(w.space for w in ws_[:-1])
            al = ln["align"]; x = ln["x"] - (total / 2 if al == "c" else total if al == "r" else 0)
            for w, t0 in zip(ws_, ln["t"]):
                w.x, w.y, w.t0 = x - w.pad, ln["y"] - w.base, t0; x += w.adv + w.space; self.words.append(w)
    def active(self, tt): return self.t_in - 0.01 <= tt <= self.t_out + 0.3
    def draw(self, dst, tt):
        out = ease((tt - self.t_out) / 0.22) if tt > self.t_out else 0
        for w in self.words:
            age = tt - w.t0
            if age < 0: continue
            e = ease(age / 0.25); al = min(1, age / 0.12) * (1 - out); dy = 14 * (1 - e) - 12 * out
            blend(dst, w.sh, w.x, w.y + dy, al); blend(dst, w.img, w.x, w.y + dy, al)

SPOTS = [(110, 1270, "l"), (970, 1330, "r"), (540, 1390, "c")]
subs = []
for ci, ch in enumerate(B.chunks):
    x, y, al = SPOTS[ci % 3]
    words = [(w.strip("«»").rstrip(",."), B.clean(w) in B.ACC_B, s) for s, e, w in ch]
    txt = " ".join(w for w, _, _ in words); lines = [words]
    if len(txt) > 15 and len(words) > 1:
        k = (len(words) + 1) // 2; lines = [words[:k], words[k:]]
    L = []
    for li, wl in enumerate(lines):
        dx = (60 if al == "l" else -60 if al == "r" else 30) * li
        L.append(dict(words=[(w, h) for w, h, _ in wl], x=x + dx, y=y + li * int(S * 1.3), align=al, t=[t0 - 0.04 for _, _, t0 in wl]))
    nxt = B.chunks[ci + 1][0][0] - 0.08 if ci + 1 < len(B.chunks) else DUR
    subs.append(Sub(L, ch[0][0] - 0.06, min(nxt, ch[-1][1] + 0.7)))

# ---------------------------------------------------------------- visual inserts schedule
T_HOOK_END = sec_end("hook") - 0.2
T_WORK = (ws(99) - 0.25, sec_end("work") - 0.1)          # framed video on blurred bg
T_TWO2 = ws(171)                                          # «поехала на вторую»: whip zoom + flash
T_REC = (ws(273) - 0.2, sec_end("trips") - 0.1)          # camcorder viewfinder
T_HUSB = 78.65                                            # freeze-zoom on husband (fully in frame from ~78.6 s)
HUSB_LEN = 1.2
T_BW = (ws(376) - 0.15, ws(400) - 0.1)                   # «раньше» in b&w, colour back on «сейчас»
T_DREAM = (ws(478) - 0.25, sec_end("why") - 0.1)         # soft dreamy glow
SEC_CHANGES = [STARTS[i] for i in range(1, len(CUTS)) if CUTS[i][2] != CUTS[i - 1][2]]

TEXT = [
    Line("ЧАСТЬ 2", 30, 700, 540, 300, 0.1, T_HOOK_END, anim="pop"),
    Line("Я НЕ ЛЮБЛЮ", 70, 900, 540, 395, 0.25, T_HOOK_END),
    Line("МАТЕРИНСТВО", 84, 900, 540, 490, 0.6, T_HOOK_END),
    Line("Что конкретно меня бесит?", 42, 600, 540, 565, 1.3, T_HOOK_END),
    Line("Каждый день хочу уволиться.", 46, 800, 540, 250, ws(99), T_WORK[1]),
    Line("Каждый день передумываю.", 46, 600, 540, 320, ws(106), T_WORK[1]),
    Line("Закончила одну смену.", 50, 700, 540, 330, ws(164), sec_end("two") - 0.15),
    Line("Поехала на вторую.", 62, 900, 540, 420, T_TWO2, sec_end("two") - 0.15, anim="pop"),
    Line("Выйти с ребёнком из дома", 46, 700, 540, 330, ws(273), T_REC[1]),
    Line("= спецоперация", 66, 900, 540, 420, ws(276), T_REC[1], anim="pop"),
    Line("это мой муж", 54, 800, 520, 560, T_HUSB + 0.1, T_HUSB + HUSB_LEN - 0.05, anim="pop"),
    Line("Раньше: стрелки, укладка, макияж.", 42, 700, 540, 330, ws(376), sec_end("self") - 0.15),
    Line("Сейчас: успеть помыться.", 52, 900, 540, 410, ws(400), sec_end("self") - 0.15, anim="pop"),
    Line("Иногда просто хочется снова", 44, 600, 540, 330, ws(478), T_DREAM[1]),
    Line("почувствовать себя собой.", 54, 800, 540, 405, ws(484), T_DREAM[1]),
]

def amount(tt, a, b, fade=0.35):
    return ease_io((tt - a) / fade) * (1 - ease_io((tt - b) / fade))

# ---------------------------------------------------------------- light skin retouch
YUNET = os.environ.get("YUNET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "yunet.onnx"))
DET = cv2.FaceDetectorYN_create(YUNET, "", (OW // 8, OH // 8), 0.6)
class Retouch:
    def __init__(self): self.box = None; self.n = 0
    def __call__(self, fr):
        if self.n % 6 == 0:
            sm = cv2.cvtColor(cv2.resize(fr, (OW // 8, OH // 8), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
            _, f = DET.detect(sm)
            if f is not None and len(f):
                x, y, w, h = max(f[:, :4], key=lambda r: r[2] * r[3]) * 8
                nb = np.float32([x - 0.25 * w, y - 0.3 * h, w * 1.5, h * 1.7])
                self.box = nb if self.box is None else self.box * 0.6 + nb * 0.4
        self.n += 1
        if self.box is None: return fr
        x, y, w, h = self.box.astype(int)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(OW, x + w), min(OH, y + h)
        if x1 - x0 < 64 or y1 - y0 < 64: return fr
        crop = fr[y0:y1, x0:x1]
        q = cv2.resize(crop, ((x1 - x0) // 4, (y1 - y0) // 4), interpolation=cv2.INTER_AREA)
        sm = cv2.bilateralFilter(q, 9, 28, 6)
        ycc = cv2.cvtColor(q, cv2.COLOR_RGB2YCrCb)
        skin = cv2.inRange(ycc, (40, 135, 85), (250, 175, 135)).astype(np.float32) / 255
        ex, ey = (x1 - x0) // 4, (y1 - y0) // 4          # feather the box edges
        fx = np.minimum(1, np.minimum(np.arange(ex), np.arange(ex)[::-1]) / (ex * 0.12 + 1))
        fy = np.minimum(1, np.minimum(np.arange(ey), np.arange(ey)[::-1]) / (ey * 0.12 + 1))
        m = cv2.GaussianBlur(skin, (0, 0), 3) * fy[:, None] * fx[None, :] * 0.55
        smooth = cv2.resize(sm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        m = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)[..., None]
        c = crop.astype(np.float32)
        # keep fine texture: only replace the low-frequency part
        low = cv2.GaussianBlur(c, (0, 0), 2.0)
        out = c + (smooth - low) * m
        fr = fr.copy(); fr[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        return fr

# ---------------------------------------------------------------- frame composition
FR_SCALE, FR_CY = 0.70, 1120
_mask_cache = {}
def rounded_mask(w, h, r):
    k = (w, h, r)
    if k not in _mask_cache:
        m = np.zeros((h, w), np.uint8); cv2.rectangle(m, (r, 0), (w - r, h), 255, -1); cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
        for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)): cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
        _mask_cache[k] = (m.astype(np.float32) / 255)[..., None]
    return _mask_cache[k]

def blurred_bg(fr, dim=0.25):
    q = cv2.resize(fr, (OW // 8, OH // 8), interpolation=cv2.INTER_AREA)
    q = cv2.GaussianBlur(q, (0, 0), 6)
    return (cv2.resize(q, (OW, OH)).astype(np.float32) * (1 - dim)).astype(np.uint8)

def viewfinder(dst, tt, a):
    if a <= 0.01: return
    col = (255, 255, 255); th = 5 * K; L = 90 * K; m = 70 * K
    ov = dst.copy()
    for (x, y, dx, dy) in ((m, m + 120 * K, 1, 1), (OW - m, m + 120 * K, -1, 1), (m, OH - m - 380 * K, 1, -1), (OW - m, OH - m - 380 * K, -1, -1)):
        cv2.line(ov, (x, y), (x + dx * L, y), col, th, cv2.LINE_AA); cv2.line(ov, (x, y), (x, y + dy * L), col, th, cv2.LINE_AA)
    if int(tt * 2) % 2 == 0: cv2.circle(ov, (m + 40 * K, m + 175 * K), 14 * K, (235, 40, 40), -1, cv2.LINE_AA)
    el = max(0, tt - T_REC[0]); stamp = f"REC  00:{int(el) // 60:02d}:{int(el) % 60:02d}"
    cv2.putText(ov, stamp, (m + 70 * K, m + 188 * K), cv2.FONT_HERSHEY_SIMPLEX, 1.1 * K, col, 2 * K, cv2.LINE_AA)
    cx, cy = OW // 2, int(OH * 0.47); cv2.line(ov, (cx - 30 * K, cy), (cx + 30 * K, cy), col, 2 * K, cv2.LINE_AA); cv2.line(ov, (cx, cy - 30 * K), (cx, cy + 30 * K), col, 2 * K, cv2.LINE_AA)
    cv2.addWeighted(ov, a, dst, 1 - a, 0, dst)

def zoom_at(tt):
    i = min(int(np.searchsorted(STARTS, tt, side="right") - 1), len(CUTS) - 1); d = tt - STARTS[i]
    z = (1.0 if i % 2 == 0 else 1.06) * (1 + 0.02 * d / max(0.5, STARTS[i + 1] - STARTS[i]))
    if i > 0 and d < 0.2: z *= 1 + 0.035 * (1 - ease(d / 0.2))
    if 0 <= tt - T_TWO2 < 0.35: z *= 1 + 0.12 * (1 - ease((tt - T_TWO2) / 0.35))
    return z

RET = Retouch()
_freeze = None
def render_frame(src, tt):
    global _freeze
    fr = RET(src)
    # husband freeze-zoom: hold the frame at T_HUSB and push in on the right side
    if T_HUSB <= tt < T_HUSB + HUSB_LEN:
        if _freeze is None: _freeze = fr
        p = ease((tt - T_HUSB) / 0.35); z = 1 + 0.6 * p; fx, fy = OW * 0.84, OH * 0.38
        M = np.float32([[z, 0, (1 - z) * fx], [0, z, (1 - z) * fy]])
        out = cv2.warpAffine(_freeze, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        g = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        out = cv2.addWeighted(out, 1 - 0.6 * p, g, 0.6 * p, 0)
    else:
        z = zoom_at(tt); fy = H * 0.42 * K
        M = np.float32([[z, 0, (1 - z) * OW / 2], [0, z, (1 - z) * fy]])
        out = cv2.warpAffine(fr, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # framed video on its own blurred background
    a = amount(tt, *T_WORK)
    if a > 0.005:
        s = 1 - (1 - FR_SCALE) * a; w, h = int(OW * s), int(OH * s)
        small = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
        bg = blurred_bg(out); x0 = (OW - w) // 2; y0 = int(np.clip(FR_CY * K * a + OH / 2 * (1 - a) - h / 2, 0, OH - h))
        m = rounded_mask(w, h, int(40 * K * a) + 1)
        sh = cv2.GaussianBlur(np.pad(m[..., 0], 40 * K)[::4, ::4], (0, 0), 6); sh = cv2.resize(sh, (w + 80 * K, h + 80 * K))[..., None] * 0.45 * a
        bgf = bg.astype(np.float32); ys, xs = y0 - 40 * K + 18 * K, x0 - 40 * K
        ya, yb = max(0, ys), min(OH, ys + sh.shape[0]); xa, xb = max(0, xs), min(OW, xs + sh.shape[1])
        bgf[ya:yb, xa:xb] *= 1 - sh[ya - ys:yb - ys, xa - xs:xb - xs]
        reg = bgf[y0:y0 + h, x0:x0 + w]; reg[:] = reg * (1 - m) + small.astype(np.float32) * m
        out = np.clip(bgf, 0, 255).astype(np.uint8)
    # b&w «раньше»
    a = amount(tt, *T_BW)
    if a > 0.005:
        g = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB); out = cv2.addWeighted(out, 1 - a, g, a, 0)
    # dreamy glow
    a = amount(tt, *T_DREAM, fade=0.5)
    if a > 0.005:
        q = cv2.GaussianBlur(cv2.resize(out, (OW // 6, OH // 6), interpolation=cv2.INTER_AREA), (0, 0), 5)
        glow = cv2.resize(q, (OW, OH)).astype(np.float32)
        o = out.astype(np.float32); o = 255 - (255 - o) * (255 - glow * 0.45 * a) / 255
        out = np.clip(o * (1 - 0.05 * a) + np.float32([255, 236, 230]) * 0.05 * a, 0, 255).astype(np.uint8)
    # camcorder viewfinder
    a = amount(tt, *T_REC, fade=0.2)
    if a > 0.005:
        out = cv2.addWeighted(out, 1 - 0.25 * a, cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB), 0.25 * a, 0)
        viewfinder(out, tt, a)
    # white flash on section changes and on «поехала на вторую»
    for sc in SEC_CHANGES + [T_TWO2, T_HUSB]:
        d = tt - sc
        if 0 <= d < 0.2: out = cv2.addWeighted(out, 1 - 0.35 * (1 - d / 0.2), np.full_like(out, 255), 0.35 * (1 - d / 0.2), 0)
    for t in TEXT:
        if t.active(tt): t.draw(out, tt)
    for g in subs:
        if g.active(tt) and not (T_HUSB <= tt < T_HUSB + HUSB_LEN): g.draw(out, tt)
    tail = DUR - tt
    if tail < 0.45: out = (out.astype(np.float32) * max(0, tail / 0.45)).astype(np.uint8)
    return out

def sfx(path):
    SR = 48000; rng = np.random.default_rng(9); n = int((DUR + 0.3) * SR); trk = np.zeros(n)
    def onepole(x, cut):
        y = np.empty_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * cut / SR)
        for i in range(len(x)): s += a * (x[i] - s); y[i] = s
        return y
    def norm(y): return y / (np.abs(y).max() + 1e-9)
    def put(sig, t, g):
        i = int(max(0, t) * SR); seg = trk[i:i + len(sig)]; seg += sig[:len(seg)] * g
    def tone(f, d, dec):
        t = np.arange(int(d * SR)) / SR; return np.sin(2 * np.pi * f * t) * np.exp(-t * dec) * np.minimum(1, t / 0.004)
    def pop():
        t = np.arange(int(0.09 * SR)) / SR; f = 300 + 700 * np.exp(-t * 55)
        return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 40))
    def whoosh(d=0.45, lo=500, hi=4500):
        m = int(d * SR); t = np.linspace(0, 1, m); x = rng.standard_normal(m)
        return norm((onepole(x, hi) - onepole(x, lo)) * np.sin(np.pi * t) ** 2)
    def boom():
        t = np.arange(int(1.0 * SR)) / SR; f = 42 + 70 * np.exp(-t * 18)
        return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.5) * np.minimum(1, t / 0.004))
    def shutter():
        t = np.arange(int(0.22 * SR)) / SR; x = rng.standard_normal(len(t))
        return norm(onepole(x * (np.exp(-t * 80) + 0.8 * np.exp(-np.maximum(0, t - 0.09) * 70) * (t > 0.09)), 6000))
    def rec_beep(): return norm(np.r_[tone(1760, 0.09, 8), np.zeros(int(0.06 * SR)), tone(1760, 0.09, 8)])
    def record_scratch():
        t = np.arange(int(0.35 * SR)) / SR; f = 900 * np.exp(-t * 6) + 120
        return norm((np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.6 + onepole(rng.standard_normal(len(t)), 2500)) * np.exp(-t * 7))
    def sparkle():
        t = np.arange(int(0.9 * SR)) / SR
        return norm(sum(np.sin(2 * np.pi * f * t) * np.exp(-np.clip(t - k * 0.07, 0, None) * 8) * (t >= k * 0.07) for k, f in enumerate([1568, 1976, 2349, 3136])))
    def riser(d=0.6):
        m = int(d * SR); t = np.linspace(0, 1, m); x = rng.standard_normal(m)
        return norm((onepole(x, 6000) - onepole(x, 800)) * t ** 2)
    def thud(): return norm(tone(70, 0.4, 12) + 0.4 * onepole(rng.standard_normal(int(0.4 * SR)), 300) * np.exp(-np.arange(int(0.4 * SR)) / SR * 15))
    WH, POP = whoosh(), pop()
    put(boom(), 0.12, 0.30); put(whoosh(0.35, 800, 6000), 0.15, 0.12)
    for t in TEXT: put(POP if t.anim == "pop" else whoosh(0.3, 900, 5000), t.t0, 0.16 if t.anim == "pop" else 0.08)
    for sc in SEC_CHANGES: put(WH, sc - 0.25, 0.14)
    put(WH, T_WORK[0] - 0.1, 0.16)
    put(riser(0.5), T_TWO2 - 0.5, 0.12); put(boom(), T_TWO2, 0.18)
    put(rec_beep(), T_REC[0] + 0.05, 0.10)
    put(record_scratch(), T_HUSB - 0.05, 0.16); put(shutter(), T_HUSB + 0.05, 0.22)
    put(whoosh(0.5, 300, 2500), T_BW[0] - 0.2, 0.12); put(sparkle(), T_BW[1], 0.10)
    put(sparkle(), T_DREAM[0] + 0.1, 0.10)
    put(thud(), DUR - 1.2, 0.18)
    for g in subs:
        for w in g.words:
            pass
    w = wave.open(path, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(trk, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sfx": sfx(sys.argv[2]); sys.exit()
    if mode == "test":
        for t in [float(x) for x in sys.argv[2].split(",")]:
            fo = int(round(t * FPS)); dec = B.decoder(fo)
            src = np.frombuffer(dec.stdout.read(OW * OH * 3), np.uint8).reshape(OH, OW, 3); dec.kill()
            RET.n = 0; RET.box = None; globals()["_freeze"] = None
            if T_HUSB <= t < T_HUSB + HUSB_LEN:      # freeze needs the frame at T_HUSB
                d2 = B.decoder(int(round(T_HUSB * FPS))); _f = np.frombuffer(d2.stdout.read(OW * OH * 3), np.uint8).reshape(OH, OW, 3); d2.kill()
                globals()["_freeze"] = RET(_f); RET.n = 0
            o = render_frame(src, t)
            cv2.imwrite(f"test_{t:.2f}.png", cv2.cvtColor(cv2.resize(o, (540, 960), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        sys.exit()
    f0, f1, outp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    if f0 > int(T_HUSB * FPS) and f0 < int((T_HUSB + HUSB_LEN) * FPS) + 1:
        raise SystemExit("chunk must not start inside the husband freeze")
    dec = B.decoder(f0)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-g", "120", "-pix_fmt", "yuv420p",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", outp], stdin=subprocess.PIPE)
    for fo in range(f0, f1):
        buf = dec.stdout.read(OW * OH * 3)
        if len(buf) < OW * OH * 3: break
        enc.stdin.write(render_frame(np.frombuffer(buf, np.uint8).reshape(OH, OW, 3), fo / FPS).tobytes())
    dec.kill(); enc.stdin.close(); enc.wait(); print("done", outp)
