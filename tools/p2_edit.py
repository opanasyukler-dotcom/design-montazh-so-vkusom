"""«Я не люблю материнство. Часть 2» — Reels cut from a 5-min 4K talk, house-style subtitles + text inserts.
Work dir: in/src.mov, words.json (whisper words on the source), db.npy (10 ms loudness), fonts.
  python3 -I p2_edit.py cut            -> cuts.json, cut.wav, seg/*.mp4, cut60.mp4 (4K 60 fps CFR, exact frame counts)
  python3 -I p2_edit.py test 3.2,10    -> test_*.png      (needs cwords.json = whisper words on cut.wav)
  python3 -I p2_edit.py render f0 f1 out.mp4
  python3 -I p2_edit.py sfx sfx.wav
Text layout uses 1080x1920 design units, scaled by K.
"""
import json, os, subprocess, sys, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, K, FPS = 1080, 1920, 2, 60
OW, OH = W * K, H * K
FONT_MI, FONT_M, FONT_S = "Montserrat-Italic-VariableFont_wght.ttf", "Montserrat-VariableFont_wght.ttf", "BiroScriptUSPlus-Regular.ttf"
def hx(s): return np.float32([int(s[i:i + 2], 16) for i in (1, 3, 5)]) / 255
INK, COCOA, ROSE, MIST, BLUSH = hx("#2F0600"), hx("#825D4D"), hx("#C5A29C"), hx("#C5B0AD"), hx("#F7DFDD")
WHITE = np.float32([1, 1, 1])

# ---------------------------------------------------------------- the cut (word index ranges on the source transcript)
SECTIONS = [
    ("hook", [(2, 36)]),
    ("work", [(52, 60), (61, 66), (73, 87), (99, 112)]),
    ("two",  [(113, 132), (139, 150), (164, 178)]),
    ("trips", [(258, 272), (273, 293), (303, 309)]),
    ("humor", [(310, 347)]),
    ("self", [(376, 388), (400, 411)]),
    ("why",  [(445, 477), (478, 492)]),
    ("final", [(505, 509), (511, 516), (520, 536), (537, 550)]),
]
TAIL_END = 297.0          # keep the phone falling after the last word
FREEZE = 2.6              # seconds of frozen, blurred last frame for the final card
TH, MINSIL, KEEPSIL = -52, 0.32, 0.14

def build_intervals():
    Wd = json.load(open("words.json")); db = np.load("db.npy")
    def quiet_before(t):
        for k in range(int(t * 100), max(0, int((t - 0.3) * 100)), -1):
            if db[k] < -50: return k / 100
        return t
    def quiet_after(t):
        for k in range(int(t * 100), int((t + 0.35) * 100)):
            if k < len(db) and db[k] < -50: return k / 100
        return t
    raw = []          # (a, b, section)
    for name, rngs in SECTIONS:
        for a, b in rngs:
            s = quiet_before(Wd[a][0] - 0.06); e = quiet_after(Wd[b][1] + 0.06)
            if name == "final" and b == 550: e = TAIL_END
            raw.append((s, e, name))
    iv = []
    for x, y, name in raw:      # compress internal silences
        fr = db[int(x * 100):int(y * 100)] < TH; cur = x; i = 0; n = len(fr)
        while i < n:
            if fr[i]:
                j = i
                while j < n and fr[j]: j += 1
                if (j - i) / 100 >= MINSIL and i > 0 and j < n:
                    iv.append((cur, x + i / 100 + KEEPSIL, name)); cur = x + j / 100 - KEEPSIL
                i = j
            else: i += 1
        iv.append((cur, y, name))
    iv = [(round(a * FPS) / FPS, round(b * FPS) / FPS, n) for a, b, n in iv if b - a > 0.1]
    return iv

def do_cut():
    iv = build_intervals(); json.dump(iv, open("cuts.json", "w"))
    print(len(iv), "pieces, total", round(sum(b - a for a, b, _ in iv), 2))
    SR = 48000
    src = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "in/src.mov", "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                                       capture_output=True).stdout, np.int16).astype(np.float32) / 32768
    out = []; F = int(0.006 * SR)
    for a, b, _ in iv:
        n = int(round((b - a) * FPS)) * SR // FPS
        s = src[int(a * SR):int(a * SR) + n].copy(); s = np.pad(s, (0, n - len(s)))
        s[:F] *= np.linspace(0, 1, F); s[-F:] *= np.linspace(1, 0, F); out.append(s)
    o = np.concatenate(out + [np.zeros(int(FREEZE * SR))])
    w = wave.open("cut.wav", "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((o * 32767).astype(np.int16).tobytes()); w.close()
    os.makedirs("seg", exist_ok=True); lst = []
    for i, (a, b, _) in enumerate(iv):
        n = int(round((b - a) * FPS)); outp = f"seg/s{i:03d}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.4f}", "-i", "in/src.mov", "-an", "-vf", f"fps={FPS}", "-frames:v", str(n),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "13", "-pix_fmt", "yuv420p", outp], check=True)
        got = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", outp],
                                 capture_output=True, text=True).stdout.split()[0].strip(","))
        if got != n: print("frame mismatch", i, n, got)
        lst.append(f"file 's{i:03d}.mp4'")
    open("seg/list.txt", "w").write("\n".join(lst))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "seg/list.txt", "-c", "copy", "cut60.mp4"], check=True)

if __name__ == "__main__" and sys.argv[1] == "cut":
    do_cut(); sys.exit()

# ---------------------------------------------------------------- timeline on the cut
CUTS = json.load(open("cuts.json"))
NF = [int(round((b - a) * FPS)) for a, b, _ in CUTS]
STARTS = np.cumsum([0] + NF) / FPS
SPEECH_END = STARTS[-1]
DUR = SPEECH_END + FREEZE
NFR = sum(NF) + int(round(FREEZE * FPS))
SEC_START = {}
for i, (_, _, n) in enumerate(CUTS): SEC_START.setdefault(n, STARTS[i])
def src2cut(t):
    """map a source time to the cut timeline (None if cut away)"""
    for i, (a, b, _) in enumerate(CUTS):
        if a - 0.05 <= t <= b + 0.05: return STARTS[i] + min(max(t - a, 0), b - a)
    return None
SRCW = json.load(open("words.json"))
def ws(i): return src2cut(SRCW[i][0])          # cut-time of source word i
def we(i): return src2cut(SRCW[i][1])

# ---------------------------------------------------------------- drawing
def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def ease_io(p): p = min(max(p, 0), 1); return p * p * (3 - 2 * p)
def back(p, c=1.6): p = min(max(p, 0), 1); return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2
def P(v): return int(round(v * K))

def blend(dst, img, x, y, alpha=1.0):
    """premultiplied float RGBA img over uint8 dst (patch-only float math)"""
    if alpha <= 0.003: return
    h, w = img.shape[:2]; x, y = P(x), P(y)
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0: return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1].astype(np.float32) * (1 / 255)
    d = d * (1 - s[..., 3:4] * alpha) + s[..., :3] * alpha
    dst[y0:y1, x0:x1] = np.clip(d * 255 + 0.5, 0, 255).astype(np.uint8)

_fc = {}
def font(path, size, wght=None):
    k = (path, size, wght)
    if k not in _fc:
        f = ImageFont.truetype(path, size)
        if wght: f.set_variation_by_axes([wght])
        _fc[k] = f
    return _fc[k]

def raster(txt, f, col, halo=None, halo_r=6, halo_a=0.85, dil=0):
    pad = 16 * K; asc, desc = f.getmetrics(); w = int(f.getlength(txt)) + 2 * pad + 20 * K
    im = Image.new("L", (w, asc + desc + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad, pad + asc), txt, font=f, fill=255, anchor="ls")
    a = np.asarray(im).astype(np.float32) / 255
    if dil: a = cv2.dilate(a, np.ones((dil, dil), np.uint8))
    out = np.zeros(a.shape + (4,), np.float32); out[..., :3] = col * a[..., None]; out[..., 3] = a
    sh = None
    if halo is not None:
        s_ = np.minimum(1, cv2.GaussianBlur(cv2.dilate(a, np.ones((3 * K, 3 * K), np.uint8)), (0, 0), halo_r * K) * 1.6) * halo_a
        sh = np.zeros_like(out); sh[..., :3] = halo * s_[..., None]; sh[..., 3] = s_
    return out, sh, (pad + asc) / K, pad / K, f.getlength(txt) / K

# subtitles: dark ink on a soft light halo (light coat / sky background)
S = 44
SUBSTY = {
    "t": dict(f=lambda: font(FONT_MI, S * K, 560), col=INK, halo=WHITE),
    "b": dict(f=lambda: font(FONT_M, int(S * 1.3) * K, 800), col=INK, halo=None),     # sits on a blush selection box
    "s": dict(f=lambda: font(FONT_S, int(S * 2.0) * K), col=COCOA, halo=WHITE),
}
class SWord:
    def __init__(self, txt, st):
        c = SUBSTY[st]; self.st = st
        self.img, self.sh, self.base, self.pad, self.adv = raster(txt, c["f"](), c["col"], c["halo"], dil=(2 if st == "s" else 0))
        self.space = SUBSTY["t"]["f"]().getlength(" ") / K
        self.box = None
        if st == "b":
            bw, bh = int((self.adv + 24) * K), int((self.base - self.pad + 18) * K)
            bx = np.zeros((bh, bw, 4), np.float32); bx[..., :3] = BLUSH * 0.92; bx[..., 3] = 0.92
            self.box = bx
class SubGroup:
    def __init__(self, lines, t_in, t_out):
        self.t_in, self.t_out, self.words = t_in, t_out, []
        for ln in lines:
            ws_ = [SWord(t, st) for t, st in ln["words"]]
            total = sum(w.adv for w in ws_) + sum(w.space * (1.3 if w.st == "b" else 1) for w in ws_[:-1])
            al = ln["align"]; x = ln["x"] - (total / 2 if al == "c" else total if al == "r" else 0)
            for w, t0 in zip(ws_, ln["t"]):
                w.x, w.y, w.t0, w.bx = x - w.pad, ln["y"] - w.base, t0, x
                x += w.adv + w.space * (1.3 if w.st == "b" else 1); self.words.append(w)
                w.line_y = ln["y"]
    def active(self, tt): return self.t_in - 0.01 <= tt <= self.t_out + 0.3
    def draw(self, dst, tt):
        out = ease((tt - self.t_out) / 0.25) if tt > self.t_out else 0
        for w in self.words:
            age = tt - w.t0
            if age < 0: continue
            al = min(1, age / 0.15) * (1 - out); img, sh = w.img, w.sh; dx = dy = -14 * out * 0
            dy = -14 * out; dx = 0
            if w.st == "s":
                p = ease_io(age / 0.45)
                if p < 1:
                    ww = img.shape[1]; ramp = np.clip((np.arange(ww) - p * ww * 1.15) / (-0.15 * ww), 0, 1)[None, :, None].astype(np.float32)
                    img = img * ramp; sh = sh * ramp
                al = 1 - out
            elif w.st == "b":
                p = ease(age / 0.25)
                bw = max(2, int(w.box.shape[1] * p))
                blend(dst, w.box[:, :bw], w.bx - 12, w.line_y - (w.base - w.pad) - 6 + dy, al)
                if p >= 0.999:   # selection handles
                    hb = np.zeros((int(70 * K), int(18 * K), 4), np.float32)
                    cv2.line(hb, (P(9), P(14)), (P(9), P(70)), tuple(float(c) for c in COCOA) + (1.0,), 3 * K, cv2.LINE_AA)
                    cv2.circle(hb, (P(9), P(9)), P(8), tuple(float(c) for c in COCOA) + (1.0,), -1, cv2.LINE_AA)
                    hb[..., :3] *= 1  # already premultiplied (alpha 1 where drawn)
                    blend(dst, hb, w.bx - 12 - 9, w.line_y - (w.base - w.pad) - 6 - 14 + dy, al)
                    blend(dst, hb[::-1], w.bx - 12 + w.box.shape[1] / K - 9, w.line_y - (w.base - w.pad) - 6 + dy, al)
            else:
                e = ease(age / 0.28); dx = 16 * (1 - e); k = int(18 * K * (1 - e)) | 1
                if k > 2: img = cv2.blur(img, (k, 1)); sh = cv2.blur(sh, (k, 1))
            if sh is not None: blend(dst, sh, w.x + dx, w.y + dy, al)
            blend(dst, img, w.x + dx, w.y + dy, al)

# ---------------------------------------------------------------- subtitle chunks
FIX = {310: "А это", 313: "Издалека", 314: None, 315: "чувствует", 316: "шизофрению.", 482: "наряжаться", 266: "долбаные"}
toks = []
for name, rngs in SECTIONS:
    for a, b in rngs:
        for i in range(a, b + 1):
            w = FIX.get(i, SRCW[i][2].strip()) if i in FIX else SRCW[i][2].strip()
            if w is None or w in ("—", "–", "-"): continue
            t0, t1 = ws(i), we(i)
            if t0 is None: continue
            if w.startswith("-") and toks: toks[-1][2] += w; toks[-1][1] = t1; continue
            toks.append([t0, t1, w])
chunks, cur = [], []
for t in toks:
    if cur and (len(" ".join(x[2] for x in cur + [t])) > 22 or t[0] - cur[-1][1] > 0.6): chunks.append(cur); cur = []
    cur.append(t)
    if t[2][-1] in ",.?!»" or len(cur) >= 3: chunks.append(cur); cur = []
if cur: chunks.append(cur)
merged = []
for ch in chunks:
    if merged and len(ch) == 1 and len(ch[0][2]) <= 6 and merged[-1][-1][2][-1] not in ",.?!»" \
            and len(" ".join(x[2] for x in merged[-1] + ch)) <= 24 and ch[0][0] - merged[-1][-1][1] < 0.5:
        merged[-1] = merged[-1] + ch
    else: merged.append(ch)
chunks = merged
ACC_B = {"бесит", "уволиться", "уволилась", "квест", "триггерит", "ад", "шизофрению", "больную", "помыться", "событие", "свихнулась",
         "успех", "телефон", "сложно", "стабильность", "мамы", "поликлинику", "ума"}
ACC_S = {"дочь", "материнство", "работать", "привычную", "прежняя", "съемки", "сон", "еду", "муж", "стрелки", "прическу", "пижаме",
         "косметики", "крем", "бабушки"}
def clean(w): return w.lower().strip(",.?!«»")
SPOTS = [(110, 1270, "l"), (970, 1320, "r"), (540, 1380, "c")]
subs = []
for ci, ch in enumerate(chunks):
    x, y, al = SPOTS[ci % 3]
    words = []
    for s, e, w in ch:
        c = clean(w); st = "b" if c in ACC_B else "s" if c in ACC_S else "t"
        words.append((w.strip("«»").rstrip(",."), st, s))
    txt = " ".join(w for w, _, _ in words); lines = [words]
    if len(txt) > 15 and len(words) > 1:
        k = (len(words) + 1) // 2; lines = [words[:k], words[k:]]
    L = []
    for li, wl in enumerate(lines):
        dx = (60 if al == "l" else -60 if al == "r" else 30) * li
        L.append(dict(words=[(w, st) for w, st, _ in wl], x=x + dx, y=y + li * int(S * 1.4), align=al, t=[t0 - 0.04 for _, _, t0 in wl]))
    nxt = chunks[ci + 1][0][0] - 0.08 if ci + 1 < len(chunks) else SPEECH_END
    subs.append(SubGroup(L, ch[0][0] - 0.06, min(nxt, ch[-1][1] + 0.7)))

# ---------------------------------------------------------------- text inserts: blush cards in the sky zone
class Card:
    """lines: [(text, kind, size)], kind: T heavy caps ink / n medium ink / s script cocoa / e emphasis cocoa heavy"""
    def __init__(self, lines, t_in, t_out, top=230, line_t=None, chip=None, deco=None):
        self.t_in, self.t_out, self.top = t_in, t_out, top
        rows = []
        for txt, kind, size in lines:
            f = {"T": lambda s: font(FONT_M, s * K, 850), "n": lambda s: font(FONT_M, s * K, 600),
                 "e": lambda s: font(FONT_M, s * K, 850), "s": lambda s: font(FONT_S, int(s * 1.9) * K)}[kind](size)
            col = COCOA if kind in ("s", "e") else INK
            img, _, base, pad, adv = raster(txt, f, col, dil=(2 if kind == "s" else 0))
            rows.append((img, base, pad, adv, size * (1.25 if kind != "s" else 1.35)))
        padx, pady = 50, 34
        cw = max(r[3] for r in rows) + 2 * padx; chh = sum(r[4] for r in rows) + 2 * pady + 6
        self.w, self.h = cw, chh; self.x = (W - cw) / 2
        im = Image.new("RGBA", (P(cw), P(chh)), (0, 0, 0, 0))
        ImageDraw.Draw(im).rounded_rectangle((0, 0, P(cw) - 1, P(chh) - 1), radius=P(34), fill=(247, 223, 221, 238))
        bg = np.asarray(im).astype(np.float32) / 255; bg[..., :3] *= bg[..., 3:4]; self.bg = bg
        sm = cv2.GaussianBlur(np.pad(bg[..., 3], P(40)), (0, 0), 16 * K) * 0.28
        self.sh = np.zeros(sm.shape + (4,), np.float32); self.sh[..., 3] = sm
        self.rows = []; y = pady
        for k, (img, base, pad, adv, lh) in enumerate(rows):
            y += lh * 0.86
            t0 = (line_t[k] if line_t else t_in + 0.2 + 0.35 * k)
            self.rows.append((img, (cw - adv) / 2 - pad, y - base, t0)); y += lh * 0.14
        self.chip = None
        if chip:
            f = font(FONT_M, 30 * K, 800); tw = f.getlength(chip) / K; ph = 64; pw = tw + 70
            ci = Image.new("RGBA", (P(pw), P(ph)), (0, 0, 0, 0)); d = ImageDraw.Draw(ci)
            d.rounded_rectangle((0, 0, P(pw) - 1, P(ph) - 1), radius=P(ph / 2), fill=(47, 6, 0, 245))
            d.text((P(pw / 2), P(ph / 2)), chip, font=f, fill=(247, 223, 221, 255), anchor="mm")
            c = np.asarray(ci).astype(np.float32) / 255; c[..., :3] *= c[..., 3:4]; self.chip = (c, pw, ph)
        self.deco = deco
    def active(self, tt): return self.t_in <= tt <= self.t_out + 0.35
    def draw(self, dst, tt):
        age = tt - self.t_in; out = ease((tt - self.t_out) / 0.3) if tt > self.t_out else 0
        sc = 0.88 + 0.12 * back(age / 0.45); al = min(1, age / 0.15) * (1 - out); dy = 40 * (1 - ease(age / 0.4)) - 30 * out
        cv = self.bg.copy()
        for img, x, y, t0 in self.rows:
            a = tt - t0
            if a < 0: continue
            e = ease(a / 0.3); im = img if e > 0.99 else cv2.blur(img, (int(24 * K * (1 - e)) | 1, 1))
            h_, w_ = im.shape[:2]; X, Y = P(x + 18 * (1 - e)), P(y)
            x0, y0 = max(X, 0), max(Y, 0); x1, y1 = min(X + w_, cv.shape[1]), min(Y + h_, cv.shape[0])
            if x1 > x0 and y1 > y0:
                s_ = im[y0 - Y:y1 - Y, x0 - X:x1 - X] * min(1, a / 0.12); cv[y0:y1, x0:x1] = cv[y0:y1, x0:x1] * (1 - s_[..., 3:4]) + s_
        if abs(sc - 1) > 0.003: cv = cv2.resize(cv, (max(2, int(cv.shape[1] * sc)), max(2, int(cv.shape[0] * sc))))
        ox = self.x + (self.w - cv.shape[1] / K) / 2; oy = self.top + (self.h - cv.shape[0] / K) / 2 + dy
        blend(dst, self.sh, ox - 40, oy - 40 + 12, al)
        blend(dst, cv, ox, oy, al)
        if self.chip:
            c, pw, ph = self.chip; e = back((tt - self.t_in - 0.15) / 0.4)
            if tt > self.t_in + 0.15:
                cc = cv2.resize(c, (max(2, int(c.shape[1] * e)), max(2, int(c.shape[0] * e))))
                blend(dst, cc, ox + 30, oy - ph * 0.55 + (ph - cc.shape[0] / K) / 2, al)

def sec_end(name):
    idx = [i for i, c in enumerate(CUTS) if c[2] == name]; return STARTS[idx[-1] + 1]
CARDS = [
    Card([("Я НЕ ЛЮБЛЮ", "T", 62), ("МАТЕРИНСТВО", "T", 74), ("что конкретно меня бесит?", "s", 50)],
         0.05, sec_end("hook") - 0.2, top=210, line_t=[0.15, 0.55, 1.3], chip="ЧАСТЬ 2"),
    Card([("Каждый день хочу уволиться.", "n", 44), ("Каждый день передумываю.", "e", 44)],
         ws(99) - 0.1, sec_end("work") - 0.15, line_t=[ws(99), ws(106)]),
    Card([("Закончила одну смену.", "n", 46), ("Поехала на вторую.", "e", 50)],
         ws(164) - 0.1, sec_end("two") - 0.15, line_t=[ws(164), ws(171)]),
    Card([("Выйти с ребёнком из дома", "n", 44), ("= спецоперация", "e", 56)],
         ws(273) - 0.1, sec_end("trips") - 0.15, line_t=[ws(273), ws(276)]),
    Card([("Раньше: стрелки, укладка, макияж.", "n", 40), ("Сейчас: успеть помыться.", "e", 46)],
         ws(376) - 0.1, sec_end("self") - 0.15, line_t=[ws(376), ws(400)]),
    Card([("Иногда просто хочется снова", "n", 42), ("почувствовать себя собой.", "s", 44)],
         ws(478) - 0.1, sec_end("why") - 0.15, line_t=[ws(478), ws(484)]),
]
FINAL = Card([("Материнство, 2:0.", "T", 66), ("Телефон тоже не выдержал.", "s", 48)],
             SPEECH_END - 0.25, DUR + 1, top=760, line_t=[SPEECH_END - 0.05, SPEECH_END + 0.6])
CARDS.append(FINAL)

# ---------------------------------------------------------------- frame composition
SEC_CHANGES = [STARTS[i] for i in range(1, len(CUTS)) if CUTS[i][2] != CUTS[i - 1][2]]
def zoom_at(tt):
    i = min(int(np.searchsorted(STARTS, tt, side="right") - 1), len(CUTS) - 1); d = tt - STARTS[i]
    base = 1.0 if i % 2 == 0 else 1.06
    z = base * (1 + 0.02 * d / max(0.5, STARTS[i + 1] - STARTS[i]))
    if i > 0 and d < 0.2: z *= 1 + 0.035 * (1 - ease(d / 0.2))
    return z

def render_frame(src, tt, freeze_src=None):
    if tt >= SPEECH_END:   # frozen, blurred last frame under the final card
        p = ease((tt - SPEECH_END) / 0.5)
        q = cv2.resize(freeze_src, (OW // 8, OH // 8), interpolation=cv2.INTER_AREA)
        q = cv2.GaussianBlur(q, (0, 0), 1 + 5 * p); fr = cv2.resize(q, (OW, OH))
        fr = (fr.astype(np.float32) * (1 - 0.25 * p) + BLUSH * 255 * 0.25 * p).astype(np.uint8)
    else:
        z = zoom_at(tt); fy = H * 0.42
        M = np.float32([[z, 0, (1 - z) * OW / 2], [0, z, (1 - z) * fy * K]])
        fr = cv2.warpAffine(src, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        for sc in SEC_CHANGES:
            d = tt - sc
            if 0 <= d < 0.22:
                a = 0.3 * (1 - d / 0.22); fr = cv2.addWeighted(fr, 1 - a, np.full_like(fr, 255), a, 0)
    for c in CARDS:
        if c.active(tt): c.draw(fr, tt)
    for g in subs:
        if g.active(tt): g.draw(fr, tt)
    return fr

def decoder(f0):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{f0 / FPS:.4f}", "-i", "cut60.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE, bufsize=OW * OH * 3)
_last = None
def last_frame():
    global _last
    if _last is None:
        n = sum(NF) - 1
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{(n - 2) / FPS:.4f}", "-i", "cut60.mp4", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        _last = np.frombuffer(p[:OW * OH * 3], np.uint8).reshape(OH, OW, 3)
    return _last

def sfx(path):
    SR = 48000; rng = np.random.default_rng(7); n = int((DUR + 0.2) * SR); trk = np.zeros(n)
    def onepole(x, cut):
        y = np.empty_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * cut / SR)
        for i in range(len(x)): s += a * (x[i] - s); y[i] = s
        return y
    def norm(y): return y / (np.abs(y).max() + 1e-9)
    def put(sig, t, g):
        i = int(max(0, t) * SR); seg = trk[i:i + len(sig)]; seg += sig[:len(seg)] * g
    def pop():
        m = int(0.09 * SR); t = np.arange(m) / SR; f = 300 + 700 * np.exp(-t * 55)
        return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 40))
    def whoosh(d=0.4):
        m = int(d * SR); t = np.linspace(0, 1, m); x = rng.standard_normal(m)
        return norm((onepole(x, 3500) - onepole(x, 500)) * np.sin(np.pi * t) ** 2)
    def scribble(d=0.45):
        m = int(d * SR); t = np.arange(m) / SR
        x = onepole(rng.standard_normal(m), 3500) - onepole(rng.standard_normal(m), 600)
        return norm(x * (0.55 + 0.45 * np.sin(2 * np.pi * 11 * t) ** 2) * np.sin(np.pi * t / d) ** 0.6)
    def ding(f0=1318.5):
        m = int(1.4 * SR); t = np.arange(m) / SR
        return norm(sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * k) for f, a, k in [(f0, 1, 3), (f0 * 1.5, .5, 4), (f0 * 2, .3, 6)]) * np.minimum(1, t / 0.004))
    def boom():
        m = int(1.0 * SR); t = np.arange(m) / SR; f = 42 + 70 * np.exp(-t * 18)
        return norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.5) * np.minimum(1, t / 0.004))
    POP, SCR, WH = pop(), scribble(), whoosh()
    put(boom(), 0.12, 0.3)
    for c in CARDS:
        put(WH, c.t_in - 0.15, 0.14); put(POP, c.t_in + 0.1, 0.2)
        for img, x, y, t0 in c.rows: put(SCR if img.shape[0] > 0 else POP, t0, 0.06)
    for sc in SEC_CHANGES: put(WH, sc - 0.2, 0.12)
    for g in subs:
        for w in g.words:
            if w.st == "b": put(POP, w.t0, 0.1)
    put(ding(), SPEECH_END + 0.1, 0.12)
    w = wave.open(path, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(trk, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sfx": sfx(sys.argv[2]); sys.exit()
    if mode == "info":
        print("pieces", len(CUTS), "speech", round(SPEECH_END, 2), "dur", round(DUR, 2), "frames", NFR)
        for n in dict.fromkeys(c[2] for c in CUTS): print(n, round(SEC_START[n], 2), round(sec_end(n), 2))
        sys.exit()
    if mode == "test":
        for t in [float(x) for x in sys.argv[2].split(",")]:
            fo = int(round(t * FPS)); src = None
            if t < SPEECH_END:
                dec = decoder(fo); src = np.frombuffer(dec.stdout.read(OW * OH * 3), np.uint8).reshape(OH, OW, 3); dec.kill()
            o = render_frame(src, t, last_frame() if t >= SPEECH_END else None)
            cv2.imwrite(f"test_{t:.2f}.png", cv2.cvtColor(cv2.resize(o, (540, 960), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        sys.exit()
    f0, f1, outp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    nsrc = sum(NF)
    dec = decoder(f0) if f0 < nsrc else None
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-g", "120", "-pix_fmt", "yuv420p",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", outp], stdin=subprocess.PIPE)
    src = None
    for fo in range(f0, f1):
        tt = fo / FPS
        if fo < nsrc:
            buf = dec.stdout.read(OW * OH * 3)
            if len(buf) == OW * OH * 3: src = np.frombuffer(buf, np.uint8).reshape(OH, OW, 3)
            enc.stdin.write(render_frame(src, tt).tobytes())
        else:
            enc.stdin.write(render_frame(None, tt, last_frame()).tobytes())
    if dec: dec.kill()
    enc.stdin.close(); enc.wait(); print("done", outp)
