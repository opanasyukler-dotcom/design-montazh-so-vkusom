"""«Я не люблю материнство. Часть 2» — 1-minute cut (no «крем» ending), built straight from the 4K 60 fps source.
Retention tools: jump-cut reframing (zoom levels change on every cut, slow push/pull inside), smooth reveal zooms on new
topics, B-roll from the extra clips on matching words, big key phrases that focus-pull in BEHIND the speaker
(person matte from MediaPipe selfie segmentation), phrase subtitles in Montserrat.
Work dir (p2): in/src.mov, words.json (whisper words on the source), db.npy (10 ms loudness), add/*.MOV, fonts.
Runs under a Python with mediapipe (mpenv):
  python short.py cut                -> short_cuts.json, short_raw.wav (voice cut, untreated)
  python short.py test 3.2,10        -> stest_*.png
  python short.py render f0 f1 out.mp4
  python short.py sfx sfx.wav
"""
import json, os, subprocess, sys, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, K, FPS = 1080, 1920, 2, 60
OW, OH = W * K, H * K
FONT_M = "Montserrat-VariableFont_wght.ttf"
MODELS = os.environ.get("MODELS", "../models")
def hx(s): return np.float32([int(s[i:i + 2], 16) for i in (1, 3, 5)]) / 255
INK, COCOA, ROSE, BLUSH, WHITE = hx("#2F0600"), hx("#825D4D"), hx("#C5A29C"), hx("#F7DFDD"), np.float32([1, 1, 1])

# ---------------------------------------------------------------- the cut: source word index ranges, ~60 s
SECTIONS = [
    ("hook",  [(2, 21)]),                         # «И это вторая часть … я люблю свою дочь»
    ("work",  [(61, 66), (99, 112)]),             # сто раз хотела уволиться / каждый день борюсь, пару раз почти уволилась
    ("two",   [(164, 178)]),                      # после работы еду на работу мамы
    ("trips", [(258, 272), (303, 309)]),          # долбаные поездки … поликлинику / личный ад
    ("self",  [(376, 388), (400, 411)]),          # раньше стрелки … / сейчас дай бог помыться
    ("why",   [(459, 477), (478, 492)]),          # в пижаме с грязной головой / наряжаться, чтобы не свихнуться
]
TH, MINSIL, KEEPSIL = -52, 0.28, 0.12
TAIL = 0.5                                        # hold after the last word, then fade

def build_intervals():
    Wd = json.load(open("words.json")); db = np.load("db.npy")
    def quiet_before(t):
        for k in range(int(t * 100), max(0, int((t - 0.3) * 100)), -1):
            if db[k] < -50: return k / 100
        return t
    def quiet_after(t):
        for k in range(int(t * 100), int((t + 0.3) * 100)):
            if k < len(db) and db[k] < -50: return k / 100
        return t
    iv = []
    for name, rngs in SECTIONS:
        for a, b in rngs:
            x = quiet_before(Wd[a][0] - 0.05); y = quiet_after(Wd[b][1] + 0.05)
            fr = db[int(x * 100):int(y * 100)] < TH; cur = x; i = 0; n = len(fr)
            while i < n:                                                   # compress internal silences
                if fr[i]:
                    j = i
                    while j < n and fr[j]: j += 1
                    if (j - i) / 100 >= MINSIL and i > 0 and j < n:
                        iv.append((cur, x + i / 100 + KEEPSIL, name)); cur = x + j / 100 - KEEPSIL
                    i = j
                else: i += 1
            iv.append((cur, y, name))
    return [(round(a * FPS) / FPS, round(b * FPS) / FPS, n) for a, b, n in iv if b - a > 0.1]

def do_cut():
    iv = build_intervals(); json.dump(iv, open("short_cuts.json", "w"))
    print(len(iv), "pieces, total", round(sum(b - a for a, b, _ in iv), 2))
    SR = 48000
    src = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "in/src.mov", "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                                       capture_output=True).stdout, np.int16).astype(np.float32) / 32768
    out = []; F = int(0.006 * SR)
    for a, b, _ in iv:
        n = int(round((b - a) * FPS)) * SR // FPS
        s = src[int(a * SR):int(a * SR) + n].copy(); s = np.pad(s, (0, n - len(s)))
        s[:F] *= np.linspace(0, 1, F); s[-F:] *= np.linspace(1, 0, F); out.append(s)
    o = np.concatenate(out + [np.zeros(int(TAIL * SR))])
    w = wave.open("short_raw.wav", "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((o * 32767).astype(np.int16).tobytes()); w.close()

if __name__ == "__main__" and sys.argv[1] == "cut":
    do_cut(); sys.exit()

# ---------------------------------------------------------------- timeline
CUTS = json.load(open("short_cuts.json"))
NF = [int(round((b - a) * FPS)) for a, b, _ in CUTS]
STARTS = np.cumsum([0] + NF) / FPS
FSTART = np.cumsum([0] + NF)
DUR = STARTS[-1] + TAIL
NFR = int(FSTART[-1] + round(TAIL * FPS))
SRCW = json.load(open("words.json"))
def src2cut(t, near=False):
    for i, (a, b, _) in enumerate(CUTS):
        if a - 0.05 <= t <= b + 0.05: return STARTS[i] + min(max(t - a, 0), b - a)
    if not near: return None
    i = min(range(len(CUTS)), key=lambda i: min(abs(t - CUTS[i][0]), abs(t - CUTS[i][1])))
    a, b, _ = CUTS[i]; return STARTS[i] + min(max(t - a, 0), b - a)
def ws(i): return src2cut(SRCW[i][0])
def we(i): return src2cut(SRCW[i][1])
def piece_at(fo):
    i = int(np.searchsorted(FSTART, fo, side="right") - 1); return min(i, len(CUTS) - 1)

# subtitle words (cut time), spelled as the speaker meant them
FIX = {"«Я": "«Я", "долбанные": "долбаные", "выряжаться": "наряжаться", "дается.": "даётся.", "все": "всё", "еще,": "ещё,", "ее": "её",
       "ребенком.": "ребёнком.", "прическу.": "причёску."}
IDX_FIX = {310: "А это", 311: "мой", 312: "муж.", 313: "Издалека", 314: "чувствует", 315: "", 316: "шизофрению."}
WORDS = []          # (t0, t1, text, src_idx)
for name, rngs in SECTIONS:
    for a, b in rngs:
        for i in range(a, b + 1):
            t0, t1 = ws(i), we(i)
            if t0 is None: continue
            if t1 is None: t1 = t0 + 0.3
            w = IDX_FIX.get(i, FIX.get(SRCW[i][2].strip(), SRCW[i][2].strip()))
            if w.startswith("-") and WORDS: WORDS[-1] = (WORDS[-1][0], t1, WORDS[-1][2] + w, WORDS[-1][3]); continue
            if not w: WORDS[-1] = (WORDS[-1][0], t1, WORDS[-1][2], WORDS[-1][3]); continue
            WORDS.append((t0, max(t1, t0 + 0.08), w.replace("«", "").replace("»", ""), i))

# ---------------------------------------------------------------- edit plan
# B-roll on matching words: (first src word, last src word, clip, clip start s, zoom from→to)
BROLL = [
    (17, 21, "IMG_6740", 2.6, (1.00, 1.08)),     # «что я люблю свою дочь» — selfie with her
    (106, 112, "IMG_6257", 8.0, (1.06, 1.00)),   # «и пару раз я даже почти уволилась» — feeding her in the car
    (265, 269, "IMG_7312", 1.0, (1.00, 1.10)),   # «долбаные поездки куда-либо» — pushing the stroller
    (270, 272, "IMG_6257", 21.0, (1.10, 1.02)),  # «элементарно в поликлинику» — the car again
    (475, 477, "IMG_6739", 3.2, (1.00, 1.08)),   # «гулять с ребёнком» — stroller in the park
    (481, 487, "IMG_6383", 4.0, (1.04, 1.00)),   # «наряжаться даже на улицу во двор» — dressed up with the stroller
]
# big phrases behind the speaker: (first src word, last src word, lines, centre y in design px)
BIG = [
    (8, 11, ["Я НЕ ЛЮБЛЮ", "МАТЕРИНСТВО"], 560),
    (62, 64, ["СТО РАЗ"], 520),
    (99, 100, ["КАЖДЫЙ", "ДЕНЬ"], 540),
    (176, 178, ["НА РАБОТУ", "МАМЫ"], 540),
    (307, 309, ["МОЙ ЛИЧНЫЙ", "АД"], 540),
    (381, 382, ["СТРЕЛКИ"], 520),
    (401, 404, ["ДАЙ БОГ", "ПОМЫТЬСЯ"], 540),
    (470, 471, ["В ПИЖАМЕ"], 520),
    (489, 492, ["НЕ", "СВИХНУЛАСЬ"], 540),
]
def wsn(i): return src2cut(SRCW[i][0], True)
def wen(i): return src2cut(SRCW[i][1], True)
BIG_T = [(wsn(a) - 0.12, max(wen(b) + 0.5, wsn(a) + 1.5), lines, y) for a, b, lines, y in BIG]
BR_T = [(wsn(a) - 0.04, wen(b) + 0.06, clip, cs, zz) for a, b, clip, cs, zz in BROLL]
SEC_START = {}
for i, (_, _, n) in enumerate(CUTS): SEC_START.setdefault(n, STARTS[i])

# ---------------------------------------------------------------- helpers
def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def ease_io(p): p = min(max(p, 0), 1); return p * p * (3 - 2 * p)
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
def font(size, wght):
    k = (size, wght)
    if k not in _fc:
        f = ImageFont.truetype(FONT_M, size); f.set_variation_by_axes([wght]); _fc[k] = f
    return _fc[k]

# ---------------------------------------------------------------- colour + light skin retouch
def _curve(x, lift, gain, contrast):
    x = x / 255.0; s = x + contrast * (x - 0.5) * (1 - np.abs(2 * x - 1))
    return np.clip((lift + (gain - lift) * np.clip(s, 0, 1)) * 255 + 0.5, 0, 255).astype(np.uint8)
_x = np.arange(256, dtype=np.float32)
LUT = np.stack([_curve(_x, 0.004, 1.000, 0.08), _curve(_x, 0.004, 0.995, 0.07), _curve(_x, 0.007, 0.985, 0.06)], -1).reshape(256, 1, 3)
def grade(fr): return cv2.LUT(fr, LUT)

DET = cv2.FaceDetectorYN_create(os.path.join(MODELS, "yunet.onnx"), "", (OW // 8, OH // 8), 0.6)
def face_box(fr):
    sm = cv2.cvtColor(cv2.resize(fr, (OW // 8, OH // 8), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    _, f = DET.detect(sm)
    if f is None or not len(f): return None
    return max(f[:, :4], key=lambda r: r[2] * r[3]) * 8
class Retouch:
    def __init__(self): self.box = None; self.n = 0
    def __call__(self, fr):
        if self.n % 6 == 0:
            b = face_box(fr)
            if b is not None:
                x, y, w, h = b; nb = np.float32([x - 0.25 * w, y - 0.3 * h, w * 1.5, h * 1.7])
                self.box = nb if self.box is None else self.box * 0.6 + nb * 0.4
        self.n += 1
        if self.box is None: return fr
        x, y, w, h = self.box.astype(int)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(OW, x + w), min(OH, y + h)
        if x1 - x0 < 64 or y1 - y0 < 64: return fr
        crop = fr[y0:y1, x0:x1]
        q = cv2.resize(crop, ((x1 - x0) // 4, (y1 - y0) // 4), interpolation=cv2.INTER_AREA)
        sm = cv2.bilateralFilter(q, 9, 28, 6)
        skin = cv2.inRange(cv2.cvtColor(q, cv2.COLOR_RGB2YCrCb), (40, 135, 85), (250, 175, 135)).astype(np.float32) / 255
        ex, ey = (x1 - x0) // 4, (y1 - y0) // 4
        fx = np.minimum(1, np.minimum(np.arange(ex), np.arange(ex)[::-1]) / (ex * 0.12 + 1))
        fy = np.minimum(1, np.minimum(np.arange(ey), np.arange(ey)[::-1]) / (ey * 0.12 + 1))
        m = cv2.GaussianBlur(skin, (0, 0), 3) * fy[:, None] * fx[None, :] * 0.30
        smooth = cv2.resize(sm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        m = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)[..., None]
        c = crop.astype(np.float32); low = cv2.GaussianBlur(c, (0, 0), 2.0)
        fr = fr.copy(); fr[y0:y1, x0:x1] = np.clip(c + (smooth - low) * m + 0.5, 0, 255).astype(np.uint8)
        return fr
RET = Retouch()

# ---------------------------------------------------------------- camera: reframing per jump cut
FACES = {}
def piece_face(i):
    """face centre (source px) for cut piece i, from its middle frame"""
    if i not in FACES:
        a, b, _ = CUTS[i]
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{(a + b) / 2:.3f}", "-i", "in/src.mov", "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(OH, OW, 3)
        bx = face_box(fr)
        FACES[i] = (OW / 2, OH * 0.33, OW * 0.22, OH * 0.16) if bx is None else (bx[0] + bx[2] / 2, bx[1] + bx[3] / 2, bx[2], bx[3])
    return FACES[i]
ZPAT = [1.00, 1.22, 1.08, 1.32, 1.00, 1.16, 1.28, 1.06]
def camera(fo):
    """zoom and face-anchored centre for output frame fo; every cut piece gets its own framing, with a slow push/pull,
    and the first piece of a section opens with a smooth 0.5 s pull-back reveal"""
    i = piece_at(fo); n = NF[i]; k = fo - FSTART[i]; p = k / max(1, n - 1)
    z0 = ZPAT[i % len(ZPAT)]; drift = 0.05 if i % 2 == 0 else -0.04
    z = z0 * (1 + drift * ease_io(p)) if drift > 0 else z0 * (1 - drift) * (1 + drift * ease_io(p))
    first = i == 0 or CUTS[i - 1][2] != CUTS[i][2]
    if first and i > 0: z *= 1 + 0.18 * (1 - ease(k / (0.5 * FPS)))
    z = max(1.0, z)
    fx, fy = piece_face(i)[:2]
    return z, fx, fy
def crop_origin(z, fx, fy):
    if z <= 1.001: return 0.0, 0.0
    cw, ch = OW / z, OH / z
    return np.clip(fx - cw / 2, 0, OW - cw), np.clip(fy - ch * 0.36, 0, OH - ch)     # face at ~1/3 height
def reframe(fr, z, fx, fy):
    if z <= 1.001: return fr
    x0, y0 = crop_origin(z, fx, fy)
    M = np.float32([[z, 0, -x0 * z], [0, z, -y0 * z]])
    return cv2.warpAffine(fr, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ---------------------------------------------------------------- B-roll
class BRoll:
    def __init__(self): self.key = None; self.dec = None; self.last = None
    def frame(self, j, k):
        t0, t1, clip, cs, (za, zb) = BR_T[j]
        if self.key != j:
            if self.dec: self.dec.kill()
            self.dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{cs:.3f}", "-i", f"add/{clip}.MOV", "-an", "-vf",
                                         f"scale={OW}:{OH}:force_original_aspect_ratio=increase:flags=lanczos,crop={OW}:{OH},fps={FPS}",
                                         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=OW * OH * 3)
            self.key = j; self.n = 0; self.last = None
        while self.n <= k:
            buf = self.dec.stdout.read(OW * OH * 3)
            if len(buf) == OW * OH * 3: self.last = np.frombuffer(buf, np.uint8).reshape(OH, OW, 3)
            self.n += 1
        p = k / max(1, (t1 - t0) * FPS); z = za + (zb - za) * ease_io(p)
        return reframe(self.last, z, OW / 2, OH * 0.45)
BR = BRoll()
def broll_at(tt):
    for j, (t0, t1, *_) in enumerate(BR_T):
        if t0 <= tt < t1: return j, int(round((tt - t0) * FPS))
    return None

# ---------------------------------------------------------------- person matte (MediaPipe selfie multiclass)
_SEG = None
def person_mask(fr):
    global _SEG
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    if _SEG is None:
        _SEG = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "mp", "selfie_multiclass.tflite")), output_confidence_masks=True))
    sm = np.ascontiguousarray(cv2.resize(fr, (540, 960), interpolation=cv2.INTER_AREA))
    r = _SEG.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=sm))
    m = 1 - r.confidence_masks[0].numpy_view()
    m = np.clip((m - 0.35) / 0.4, 0, 1).astype(np.float32)
    m = cv2.GaussianBlur(cv2.resize(m, (OW // 2, OH // 2), interpolation=cv2.INTER_LINEAR), (0, 0), 1.5)
    return cv2.resize(m, (OW, OH), interpolation=cv2.INTER_LINEAR)

# ---------------------------------------------------------------- big phrases behind the speaker
_big = {}
def big_layer(lines, y):
    """premultiplied RGBA of the stacked phrase, full frame width, plus its top y (output px)"""
    k = (tuple(lines), y)
    if k not in _big:
        size = 230
        f = font(int(size * K), 800)
        while max(f.getlength(l) for l in lines) > (W - 70) * K: size -= 6; f = font(int(size * K), 800)
        lh = int(size * 0.98 * K); hh = lh * len(lines) + 120 * K
        im = Image.new("L", (OW, hh), 0); d = ImageDraw.Draw(im)
        for n, l in enumerate(lines): d.text((OW // 2, 60 * K + lh * n + lh // 2), l, font=f, fill=255, anchor="mm")
        a = np.asarray(im).astype(np.float32) / 255
        rgba = np.zeros((hh, OW, 4), np.float32); rgba[..., :3] = BLUSH * a[..., None]; rgba[..., 3] = a
        glow = np.minimum(1, cv2.GaussianBlur(a, (0, 0), 18 * K) * 1.2) * 0.30          # soft dark halo for legibility
        _big[k] = (rgba, glow, 60 * K + lh * len(lines))
    return _big[k]
_anchor = {}
def big_top(t0, hb):
    """top of the phrase block (output px): its last line tucks behind the top of the head, so the words stay readable
    and the head overlaps only their lower part"""
    if t0 not in _anchor:
        fo = int(round(max(0, t0 + 0.2) * FPS)); i = piece_at(fo); z, fx, fy = camera(fo)
        cx, cy, bw, bh = piece_face(i); x0, y0 = crop_origin(z, fx, fy)
        head_top = ((cy - bh / 2) - y0) * z - 0.55 * bh * z
        _anchor[t0] = int(np.clip(head_top + 0.40 * bh * z - hb, 110 * K, OH * 0.45))
    return _anchor[t0]
def draw_big(fr, tt, mask_fn):
    for t0, t1, lines, y in BIG_T:
        if not (t0 <= tt < t1 + 0.3): continue
        rgba, glow, hb = big_layer(lines, y); top = big_top(t0, hb)
        pin = ease((tt - t0) / 0.42); pout = ease_io((tt - t1) / 0.3) if tt > t1 else 0
        blur = 26 * K * (1 - pin) + 18 * K * pout; al = pin * (1 - pout)
        sc = 1.10 - 0.10 * pin + 0.04 * (tt - t0) / max(0.5, t1 - t0)                  # settles, then keeps drifting closer (depth)
        if al < 0.01: continue
        hh = rgba.shape[0]
        lay = cv2.resize(rgba, (int(OW * sc), int(hh * sc)), interpolation=cv2.INTER_LINEAR)
        gl = cv2.resize(glow, (int(OW * sc), int(hh * sc)), interpolation=cv2.INTER_LINEAR)
        if blur > 1:
            lay = cv2.GaussianBlur(lay, (0, 0), blur / 2); gl = cv2.GaussianBlur(gl, (0, 0), blur / 2)
        x0 = (OW - lay.shape[1]) // 2; y0 = top - (lay.shape[0] - hh) // 2
        ys, ye = max(0, y0), min(OH, y0 + lay.shape[0]); xs, xe = max(0, x0), min(OW, x0 + lay.shape[1])
        L = lay[ys - y0:ye - y0, xs - x0:xe - x0]; G = gl[ys - y0:ye - y0, xs - x0:xe - x0]
        m = mask_fn()[ys:ye, xs:xe, None]
        d = fr[ys:ye, xs:xe].astype(np.float32) / 255
        bg = d * (1 - G[..., None] * al)                                                 # halo
        bg = bg * (1 - L[..., 3:4] * al) + L[..., :3] * al                               # phrase
        fr[ys:ye, xs:xe] = np.clip((d * m + bg * (1 - m)) * 255 + 0.5, 0, 255).astype(np.uint8)   # speaker stays in front

# ---------------------------------------------------------------- phrase subtitles (Montserrat)
def chunks():
    hidden = set()
    for a, b, *_ in BIG: hidden.update(range(a, b + 1))
    out, cur = [], []
    for w in WORDS:
        if w[3] in hidden:
            if cur: out.append(cur); cur = []
            continue
        cur.append(w)
        txt = " ".join(x[2] for x in cur)
        if len(cur) >= 3 or len(txt) >= 16 or w[2][-1] in ".,?!:—": out.append(cur); cur = []
    if cur: out.append(cur)
    res = []
    for n, c in enumerate(out):
        t0 = c[0][0] - 0.04; nxt = out[n + 1][0][0] - 0.02 if n + 1 < len(out) else c[-1][1] + 0.6
        res.append((t0, min(nxt, c[-1][1] + 0.45), " ".join(x[2] for x in c)))
    return res
SUBS = chunks()
_sub = {}
def sub_img(txt):
    if txt not in _sub:
        f = font(int(66 * K), 600); pad = 30 * K
        w = int(f.getlength(txt)) + 2 * pad; asc, desc = f.getmetrics(); h = asc + desc + 2 * pad
        im = Image.new("L", (w, h), 0); ImageDraw.Draw(im).text((pad, pad + asc), txt, font=f, fill=255, anchor="ls")
        a = np.asarray(im).astype(np.float32) / 255
        rgba = np.zeros((h, w, 4), np.float32); rgba[..., :3] = a[..., None]; rgba[..., 3] = a
        sh = np.zeros_like(rgba); sh[..., 3] = np.minimum(1, cv2.GaussianBlur(a, (0, 0), 7 * K) * 1.5) * 0.6
        _sub[txt] = (rgba, sh, (pad + asc) / K)
    return _sub[txt]
SUB_Y = 1360            # baseline in design px — clear of the Reels caption/buttons zone
def draw_subs(fr, tt):
    for t0, t1, txt in SUBS:
        if t0 <= tt < t1:
            rgba, sh, base = sub_img(txt); age = tt - t0
            al = min(1, age / 0.10); dy = 10 * (1 - ease(age / 0.18))
            x = W / 2 - rgba.shape[1] / K / 2; y = SUB_Y - base + dy
            blend(fr, sh, x, y + 4, al); blend(fr, rgba, x, y, al)

# ---------------------------------------------------------------- frame
def render_frame(src, fo):
    tt = fo / FPS
    br = broll_at(tt)
    if br is not None:
        fr = grade(BR.frame(*br)); talk = False
    else:
        fr = grade(RET(src)); z, fx, fy = camera(min(fo, FSTART[-1] - 1)); fr = reframe(fr, z, fx, fy); talk = True
    if talk:
        cache = {}
        def mfn():
            if "m" not in cache: cache["m"] = person_mask(fr)
            return cache["m"]
        draw_big(fr, tt, mfn)
    # soft flash-cut into / out of B-roll keeps the jump readable
    for t0, t1, *_ in BR_T:
        for edge in (t0, t1):
            d = tt - edge
            if 0 <= d < 0.08: fr = cv2.addWeighted(fr, 1 - 0.18 * (1 - d / 0.08), np.full_like(fr, 255), 0.18 * (1 - d / 0.08), 0)
    draw_subs(fr, tt)
    if tt > DUR - 0.45: fr = (fr.astype(np.float32) * max(0, (DUR - tt) / 0.45)).astype(np.uint8)
    return fr

def source_frames(f0, f1):
    """decode straight from the original (no intermediate): each cut piece at 60 fps, then the frozen tail"""
    last = None
    for i, (a, b, _) in enumerate(CUTS):
        p0, p1 = FSTART[i], FSTART[i + 1]
        if p1 <= f0 or p0 >= f1: continue
        skip, take = max(0, f0 - p0), min(p1, f1) - max(p0, f0)
        need = skip + take
        dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.4f}", "-i", "in/src.mov", "-an", "-vf", f"fps={FPS}", "-frames:v", str(NF[i]),
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=OW * OH * 3)
        for k in range(need):
            buf = dec.stdout.read(OW * OH * 3)
            if len(buf) == OW * OH * 3: last = np.frombuffer(buf, np.uint8).reshape(OH, OW, 3)
            if k >= skip: yield last
        dec.kill(); dec.wait()
    for fo in range(max(f0, FSTART[-1]), f1): yield last
# ---------------------------------------------------------------- sfx: soft swish on each big phrase, light whoosh into B-roll
def sfx(path):
    SR = 48000; n = int((DUR + 0.5) * SR); out = np.zeros(n); rng = np.random.default_rng(4)
    def onepole(x, cut):
        y = np.empty_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * cut / SR)
        for i in range(len(x)): s += a * (x[i] - s); y[i] = s
        return y
    def swish(d, lo, hi):
        m = int(d * SR); x = rng.standard_normal(m); y = onepole(x, hi) - onepole(x, lo); t = np.linspace(0, 1, m)
        y *= np.sin(np.pi * t ** 0.7) ** 2; return y / (np.abs(y).max() + 1e-9)
    def put(sig, t, g):
        i = int(max(0, t) * SR); seg = out[i:i + len(sig)]; seg += sig[:len(seg)] * g
    for t0, *_ in BIG_T: put(swish(0.45, 300, 3500), t0 - 0.05, 0.10)
    for t0, *_ in BR_T: put(swish(0.30, 800, 6000), t0 - 0.15, 0.07)
    w = wave.open(path, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "info":
        print("DUR", round(DUR, 2), "frames", NFR, "pieces", len(CUTS))
        for t0, t1, txt in SUBS[:12]: print(round(t0, 2), round(t1, 2), txt)
        for b in BIG_T: print("BIG", [round(b[0], 2), round(b[1], 2)], b[2])
        for b in BR_T: print("BR", [round(b[0], 2), round(b[1], 2)], b[2])
        sys.exit()
    if mode == "sfx": sfx(sys.argv[2]); sys.exit()
    if mode == "test":
        for t in [float(x) for x in sys.argv[2].split(",")]:
            fo = int(round(t * FPS)); RET.n = 0; RET.box = None
            src = next(source_frames(fo, fo + 1))
            o = render_frame(src, fo)
            cv2.imwrite(f"stest_{t:05.2f}.png", cv2.cvtColor(cv2.resize(o, (540, 960), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        sys.exit()
    f0, f1, outp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-g", "120", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "5.2",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", outp], stdin=subprocess.PIPE)
    fo = f0
    for src in source_frames(f0, f1):
        enc.stdin.write(render_frame(src, fo).tobytes()); fo += 1
    enc.stdin.close(); enc.wait(); print("done", outp, fo - f0)
