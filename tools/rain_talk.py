"""«Ливень / прямое включение» — selfie talk under an umbrella (4K 60 fps vertical, iPhone).
Clean cut (no «э-э», false start, stretched «и-и»/«в-в», filler «вот», pauses), the rain shot from the source as a
cut-away on «ливень», software stabilisation + reframing zooms (new framing on every jump cut, slow push-ins,
a closer push on the strong lines), blemish removal + light skin smoothing, denoise + gentle sharpen + grade,
Montserrat phrase subtitles with brand-colour key words. Straight from the source, x264 crf 12, 2160x3840 60 fps.
Work dir: in/src.mov, words.json (whisper, verbatim prompt), db.npy (10 ms loudness of a16.wav), fonts, models/yunet.onnx.
  python3 -I rain_talk.py cut | motion | info | test 3,10 | render f0 f1 out.mp4
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
DEC_VF = "hqdn3d=1.2:1.0:3.0:2.5,unsharp=5:5:0.45:5:5:0.0"        # light denoise of the low-light noise, gentle luma sharpen

# ---------------------------------------------------------------- cut: kept word index ranges
KEEP = [(0, 2), (5, 6), (10, 49), (51, 60), (62, 80), (82, 111), (112, 113)]
STRETCHED = {92: 0.22, 98: 0.26, 76: 0.30}        # «в-в», «и-и», «но-о»: keep only the last part of the drawn-out word
TH, MINSIL, KEEPSIL = -38, 0.20, 0.08
END_HOLD = 0.6
SUBFIX = {17: "ливень,", 22: "ребёнком.", 99: "единственный", 101: "воздуха,", 92: "В", 110: "придёт", 51: "Единственное", 60: "улица.",
          62: "Просто", 69: "площадке,", 75: "ливень,", 80: "остановит.", 82: "Поэтому", 88: "качестве,", 91: "есть.", 97: "ума.", 98: "И",
          74: "идёт", 58: "выдохнуть —"}
# cut-away: the rain shot from the pause in the source, over «когда у нас на улице ливень»
CUTAWAY = (12, 17, 6.25)                           # first word, last word, source start of the shot
HL = {"подлости.", "ливень,", "ручным", "капризной,", "противной.", "лежать", "выдохнуть", "одна", "остановит.",
      "ума.", "глоток", "воздуха,", "прогулка."}

def intervals():
    Wd = json.load(open("words.json")); db = np.load("db.npy")
    iv = []
    for a, b in KEEP:
        x = Wd[a][0] - 0.04
        if a in STRETCHED: x = Wd[a][1] - STRETCHED[a]
        y = Wd[b][1] + 0.06
        # split out stretched words inside the range
        marks = [(x, None)]
        for i in range(a + 1, b + 1):
            if i in STRETCHED:
                marks.append((Wd[i - 1][1] + 0.03, Wd[i][1] - STRETCHED[i]))
        segs = []; cur = x
        for s, e in marks[1:]:
            segs.append((cur, s)); cur = e
        segs.append((cur, y))
        for x0, y0 in segs:
            fr = db[int(x0 * 100):int(y0 * 100)] < TH; c = x0; i = 0; n = len(fr)
            while i < n:
                if fr[i]:
                    j = i
                    while j < n and fr[j]: j += 1
                    if (j - i) / 100 >= MINSIL and i > 0 and j < n:
                        iv.append((c, x0 + i / 100 + KEEPSIL)); c = x0 + j / 100 - KEEPSIL
                    i = j
                else: i += 1
            iv.append((c, y0))
    iv[-1] = (iv[-1][0], iv[-1][1] + END_HOLD)
    return [(max(0.0, round(a * FPS) / FPS), round(b * FPS) / FPS) for a, b in iv if b - a > 0.08]

def do_cut():
    iv = intervals(); json.dump(iv, open("cuts.json", "w"))
    print(len(iv), "pieces, total", round(sum(b - a for a, b in iv), 2))
    SR = 48000
    src = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "in/src.mov", "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                                       capture_output=True).stdout, np.int16).astype(np.float32) / 32768
    out = []; F = int(0.008 * SR)
    for a, b in iv:
        n = int(round((b - a) * FPS)) * SR // FPS
        s = src[int(a * SR):int(a * SR) + n].copy(); s = np.pad(s, (0, n - len(s)))
        s[:F] *= np.linspace(0, 1, F); s[-F:] *= np.linspace(1, 0, F); out.append(s)
    o = np.concatenate(out)
    w = wave.open("cut.wav", "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((o * 32767).astype(np.int16).tobytes()); w.close()

if __name__ == "__main__" and sys.argv[1] == "cut":
    do_cut(); sys.exit()

CUTS = json.load(open("cuts.json"))
NF = [int(round((b - a) * FPS)) for a, b in CUTS]
FSTART = np.cumsum([0] + NF); STARTS = FSTART / FPS
NFR = int(FSTART[-1]); DUR = NFR / FPS
SRCW = json.load(open("words.json"))
def src2cut(t):
    best = None
    for i, (a, b) in enumerate(CUTS):
        if a - 0.03 <= t <= b + 0.03: return STARTS[i] + min(max(t - a, 0), b - a)
        d = min(abs(t - a), abs(t - b))
        if best is None or d < best[0]: best = (d, STARTS[i] + min(max(t - a, 0), b - a))
    return best[1]
def piece_at(fo): return min(int(np.searchsorted(FSTART, fo, side="right") - 1), len(CUTS) - 1)
# framing groups: a new framing only on a real jump (gap > 0.45 s in the source); tiny silence trims keep the framing
GROUP = [0]
for i in range(1, len(CUTS)): GROUP.append(GROUP[-1] + (1 if CUTS[i][0] - CUTS[i - 1][1] > 0.45 else 0))
G_F0 = {g: FSTART[GROUP.index(g)] for g in set(GROUP)}
G_F1 = {g: FSTART[len(GROUP) - GROUP[::-1].index(g)] for g in set(GROUP)}

# ---------------------------------------------------------------- stabilisation: per-piece camera path, smoothed
def do_motion():
    """global motion between consecutive frames (LK on 270x480 grey), per piece; saved as cumulative (x, y, angle)"""
    paths = []
    for a, b in CUTS:
        n = int(round((b - a) * FPS))
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a:.4f}", "-i", "in/src.mov", "-an", "-vf", f"fps={FPS},scale=270:480",
                              "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(-1, 480, 270)
        acc = [(0.0, 0.0, 0.0)]; x = y = r = 0.0
        for k in range(1, len(fr)):
            p0 = cv2.goodFeaturesToTrack(fr[k - 1], 200, 0.01, 8)
            dx = dy = da = 0.0
            if p0 is not None and len(p0) > 10:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(fr[k - 1], fr[k], p0, None)
                ok = st.ravel() == 1
                if ok.sum() > 10:
                    M, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok])
                    if M is not None: dx, dy, da = M[0, 2], M[1, 2], np.arctan2(M[1, 0], M[0, 0])
            x += dx; y += dy; r += da; acc.append((x, y, r))
        while len(acc) < n: acc.append(acc[-1])
        paths.append(np.float32(acc[:n]) * np.float32([OW / 270, OH / 480, 1]))
    np.save("motion.npy", np.array(paths, dtype=object), allow_pickle=True)
    print("motion ok", len(paths))
if __name__ == "__main__" and sys.argv[1] == "motion":
    do_motion(); sys.exit()

PATHS = list(np.load("motion.npy", allow_pickle=True))
def smooth_path(p, sig=22):
    if len(p) < 3: return p
    k = int(sig * 3); pad = np.concatenate([p[k:0:-1] if len(p) > k else p[::-1], p, p[-2:-k - 2:-1] if len(p) > k else p[::-1]])
    g = np.exp(-0.5 * (np.arange(-k, k + 1) / sig) ** 2); g /= g.sum()
    out = np.stack([np.convolve(pad[:, c], g, "same") for c in range(3)], 1)
    off = len(p[k:0:-1]) if len(p) > k else len(p)
    return out[off:off + len(p)]
CORR = [smooth_path(p) - p for p in PATHS]          # how far to move each frame back onto the smooth path

# ---------------------------------------------------------------- faces (source coords), cached per piece
DET = cv2.FaceDetectorYN_create(os.path.join(MODELS, "yunet.onnx"), "", (OW // 8, OH // 8), 0.6)
def detect(fr):
    sm = cv2.cvtColor(cv2.resize(fr, (OW // 8, OH // 8), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    _, f = DET.detect(sm)
    if f is None or not len(f): return None
    return max(f, key=lambda r: r[2] * r[3])[:14] * 8          # box + 5 landmarks
_pf = {}
def piece_face(i):
    if i not in _pf:
        a, b = CUTS[i]
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{(a + b) / 2:.3f}", "-i", "in/src.mov", "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        f = detect(np.frombuffer(raw, np.uint8).reshape(OH, OW, 3))
        _pf[i] = (OW / 2, OH * 0.35, OW * 0.4) if f is None else (f[0] + f[2] / 2, f[1] + f[3] / 2, f[2])
    return _pf[i]

# ---------------------------------------------------------------- camera: stabilised, reframed per jump cut
ZPAT = [1.10, 1.24, 1.12, 1.30, 1.10, 1.20, 1.34, 1.14]
PUSH = {6: 1.18, 97: 1.18, 113: 1.22, 80: 1.12, 49: 1.10}       # closer push-in that peaks on these words
PUSH_T = {i: src2cut(SRCW[i][1]) for i in PUSH}
def camera(fo):
    i = piece_at(fo); k = fo - FSTART[i]; g = GROUP[i]; p = (fo - G_F0[g]) / max(1, G_F1[g] - G_F0[g] - 1)
    z = ZPAT[g % len(ZPAT)] * (1 + (0.04 * p if g % 2 == 0 else 0.035 * (1 - p)))
    tt = fo / FPS
    for w_, amt in PUSH.items():                                  # smooth push that builds into the word, eases back after
        d = tt - PUSH_T[w_]
        if -1.6 < d < 1.0: z *= 1 + (amt - 1) * (ease_io((d + 1.6) / 1.4) if d < -0.2 else 1 - ease_io((d + 0.2) / 1.2))
    cx, cy, fw = piece_face(GROUP.index(g) + (G_F1[g] > G_F0[g] and 0))
    c = CORR[i][min(k, len(CORR[i]) - 1)]
    return z, cx, cy, c
def warp(fr, z, cx, cy, c):
    """stabilise (shift/rotate onto the smoothed path) then crop-zoom with the face at ~1/3 height"""
    dx, dy, da = c
    R = cv2.getRotationMatrix2D((OW / 2, OH / 2), np.degrees(da), 1.0); R[0, 2] += dx; R[1, 2] += dy
    cw, ch = OW / z, OH / z
    x0 = np.clip(cx + dx - cw / 2, 0, OW - cw); y0 = np.clip(cy + dy - ch * 0.38, 0, OH - ch)
    Z = np.float32([[z, 0, -x0 * z], [0, z, -y0 * z], [0, 0, 1]])
    M = (Z @ np.vstack([R, [0, 0, 1]]))[:2]
    return cv2.warpAffine(fr, M, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

# ---------------------------------------------------------------- skin: blemish removal + light smoothing
class Skin:
    def __init__(self): self.f = None; self.n = 0
    def __call__(self, fr):
        if self.n % 3 == 0:
            f = detect(fr)
            if f is not None: self.f = f if self.f is None else self.f * 0.5 + f * 0.5
        self.n += 1
        if self.f is None: return fr
        x, y, w, h = self.f[:4]
        x0, y0 = int(max(0, x - 0.15 * w)), int(max(0, y - 0.35 * h)); x1, y1 = int(min(OW, x + 1.15 * w)), int(min(OH, y + 1.1 * h))
        if x1 - x0 < 100 or y1 - y0 < 100: return fr
        crop = fr[y0:y1, x0:x1]; s = 2                           # work at half resolution inside the face box
        q = cv2.resize(crop, ((x1 - x0) // s, (y1 - y0) // s), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(q, cv2.COLOR_RGB2LAB).astype(np.float32)
        L, A = lab[..., 0], lab[..., 1]
        kk = max(5, int(w / s / 14)) | 1
        Lm = cv2.medianBlur(lab[..., 0].astype(np.uint8), kk).astype(np.float32)
        Am = cv2.medianBlur(lab[..., 1].astype(np.uint8), kk).astype(np.float32)
        # face oval only (umbrella and hair have skin-like colours), minus eyes, brows, nostrils and lips
        lm = (self.f[4:14].reshape(5, 2) - [x0, y0]) / s
        fw, fh = w / s, h / s; ox, oy = (x - x0) / s, (y - y0) / s
        skin = np.zeros(q.shape[:2], np.float32)
        cv2.ellipse(skin, (int(ox + fw / 2), int(oy + fh * 0.52)), (int(fw * 0.46), int(fh * 0.56)), 0, 0, 360, 1.0, -1)
        for (px, py) in lm[:2]:
            cv2.ellipse(skin, (int(px), int(py - 0.01 * fh)), (int(0.15 * fw), int(0.085 * fh)), 0, 0, 360, 0.0, -1)      # eye + lashes
            cv2.ellipse(skin, (int(px), int(py - 0.13 * fh)), (int(0.19 * fw), int(0.065 * fh)), 0, 0, 360, 0.0, -1)      # brow
        nx, ny = lm[2]; cv2.ellipse(skin, (int(nx), int(ny + 0.05 * fh)), (int(0.15 * fw), int(0.07 * fh)), 0, 0, 360, 0.0, -1)
        ml = lm[3:5].mean(0); mw = np.abs(lm[4, 0] - lm[3, 0])
        cv2.ellipse(skin, (int(ml[0]), int(ml[1])), (int(0.62 * mw + 0.04 * fw), int(0.085 * fh)), 0, 0, 360, 0.0, -1)
        skin = cv2.GaussianBlur(skin, (0, 0), 4)
        dL = cv2.GaussianBlur(Lm - L, (0, 0), 1.0); dA = cv2.GaussianBlur(A - Am, (0, 0), 1.0)
        spot = np.maximum(np.clip((dL - 1.8) / 3.0, 0, 1), np.clip((dA - 1.5) / 3, 0, 1))
        big = cv2.morphologyEx(spot, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        spot = np.clip(spot - big, 0, 1) * skin                    # only small spots on facial skin
        keep = skin
        spot = cv2.GaussianBlur(cv2.dilate(spot, np.ones((5, 5), np.uint8)), (0, 0), 1.5)
        spot = np.clip(spot * 2.4, 0, 1) * skin; self.dbg = spot
        lab2 = lab.copy(); lab2[..., 0] = Lm; lab2[..., 1] = Am; lab2[..., 2] = cv2.medianBlur(lab[..., 2].astype(np.uint8), kk)
        fixed = cv2.cvtColor(np.clip(lab * (1 - spot[..., None]) + lab2 * spot[..., None], 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        # light smoothing of the low frequencies on skin (texture kept)
        sm = cv2.bilateralFilter(fixed, 9, 22, 5)
        ex, ey = q.shape[1], q.shape[0]
        fx = np.minimum(1, np.minimum(np.arange(ex), np.arange(ex)[::-1]) / (ex * 0.12 + 1))
        fy = np.minimum(1, np.minimum(np.arange(ey), np.arange(ey)[::-1]) / (ey * 0.12 + 1))
        m = cv2.GaussianBlur(skin, (0, 0), 3) * fy[:, None] * fx[None, :] * 0.30 * keep
        delta = (fixed.astype(np.float32) - q.astype(np.float32)) + (sm.astype(np.float32) - cv2.GaussianBlur(fixed, (0, 0), 1.5).astype(np.float32)) * m[..., None]
        edge = (fy[:, None] * fx[None, :])[..., None]
        delta = cv2.resize(delta * edge, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        out = fr.copy(); out[y0:y1, x0:x1] = np.clip(crop.astype(np.float32) + delta + 0.5, 0, 255).astype(np.uint8)
        return out
SKIN = Skin()

# ---------------------------------------------------------------- grade
def _curve(x, lift, gain, contrast, gamma=1.0):
    x = (x / 255.0) ** gamma; s = x + contrast * (x - 0.5) * (1 - np.abs(2 * x - 1))
    return np.clip((lift + (gain - lift) * np.clip(s, 0, 1)) * 255 + 0.5, 0, 255).astype(np.uint8)
_x = np.arange(256, dtype=np.float32)
LUT = np.stack([_curve(_x, 0.010, 1.000, 0.14, 0.93), _curve(_x, 0.008, 0.990, 0.13, 0.94), _curve(_x, 0.012, 0.975, 0.12, 0.95)], -1).reshape(256, 1, 3)
def grade(fr):
    fr = cv2.LUT(fr, LUT)
    g = cv2.cvtColor(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(fr, 1.08, g, -0.08, 0)

# ---------------------------------------------------------------- subtitles
def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def ease_io(p): p = min(max(p, 0), 1); return p * p * (3 - 2 * p)
_fc = {}
def font(size, wght):
    k = (size, wght)
    if k not in _fc:
        f = ImageFont.truetype(FONT_M, size); f.set_variation_by_axes([wght]); _fc[k] = f
    return _fc[k]
WORDS = []
for a, b in KEEP:
    for i in range(a, b + 1):
        w = SUBFIX.get(i, SRCW[i][2].strip())
        if w.startswith("-") and WORDS: WORDS[-1] = (WORDS[-1][0], src2cut(SRCW[i][1]), WORDS[-1][2] + w); continue
        t0 = src2cut(SRCW[i][1] - STRETCHED[i]) if i in STRETCHED else src2cut(SRCW[i][0])
        WORDS.append((t0, max(src2cut(SRCW[i][1]), t0 + 0.1), w))
WORDS[0] = (WORDS[0][0], WORDS[0][1], WORDS[0][2].capitalize())
def chunks():
    out, cur = [], []
    for w in WORDS:
        cur.append(w); txt = " ".join(x[2] for x in cur)
        if len(cur) >= 3 or len(txt) >= 15 or w[2][-1] in ".,?!": out.append(cur); cur = []
    if cur: out.append(cur)
    res = []
    for n, c in enumerate(out):
        nxt = out[n + 1][0][0] - 0.02 if n + 1 < len(out) else DUR
        res.append((c[0][0] - 0.05, min(nxt, c[-1][1] + 0.5), [x[2] for x in c]))
    return res
SUBS = chunks()
_sub = {}
def sub_img(words):
    k = tuple(words)
    if k not in _sub:
        f = font(int(68 * K), 600); fh = font(int(76 * K), 800); sp = f.getlength(" ")
        parts = [(w, fh if w in HL else f, w in HL) for w in words]
        tw = sum(ff.getlength(w) for w, ff, _ in parts) + sp * (len(parts) - 1); pad = 30 * K
        asc = max(ff.getmetrics()[0] for _, ff, _ in parts); desc = max(ff.getmetrics()[1] for _, ff, _ in parts)
        hgt = asc + desc + 2 * pad; wid = int(tw) + 2 * pad
        a_w = Image.new("L", (wid, hgt), 0); a_h = Image.new("L", (wid, hgt), 0); x = pad
        for w, ff, hl in parts:
            ImageDraw.Draw(a_h if hl else a_w).text((x, pad + asc), w, font=ff, fill=255, anchor="ls"); x += ff.getlength(w) + sp
        aw = np.asarray(a_w).astype(np.float32) / 255; ah = np.asarray(a_h).astype(np.float32) / 255
        a = np.maximum(aw, ah)
        rgba = np.zeros((hgt, wid, 4), np.float32); rgba[..., :3] = WHITE * aw[..., None] + BLUSH * ah[..., None]; rgba[..., 3] = a
        sh = np.zeros_like(rgba); sh[..., 3] = np.minimum(1, cv2.GaussianBlur(a, (0, 0), 8 * K) * 1.6) * 0.65
        _sub[k] = (rgba, sh, (pad + asc) / K)
    return _sub[k]
def blend(dst, img, x, y, alpha=1.0):
    h, w = img.shape[:2]; x, y = int(round(x * K)), int(round(y * K))
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0 or alpha < 0.003: return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]; d = dst[y0:y1, x0:x1].astype(np.float32) / 255
    dst[y0:y1, x0:x1] = np.clip((d * (1 - s[..., 3:4] * alpha) + s[..., :3] * alpha) * 255 + 0.5, 0, 255).astype(np.uint8)
SUB_Y = 1400
def draw_subs(fr, tt):
    for t0, t1, words in SUBS:
        if t0 <= tt < t1:
            rgba, sh, base = sub_img(words); age = tt - t0
            al = min(1, age / 0.10); dy = 12 * (1 - ease(age / 0.2))
            x = W / 2 - rgba.shape[1] / K / 2; y = SUB_Y - base + dy
            blend(fr, sh, x, y + 5, al); blend(fr, rgba, x, y, al)

# ---------------------------------------------------------------- cut-away
CA_T0, CA_T1 = src2cut(SRCW[CUTAWAY[0]][0]) - 0.05, src2cut(SRCW[CUTAWAY[1]][1]) + 0.05
class Cutaway:
    def __init__(self): self.dec = None; self.n = 0; self.last = None
    def frame(self, k):
        if self.dec is None:
            self.dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{CUTAWAY[2]:.3f}", "-i", "in/src.mov", "-an", "-vf", f"fps={FPS},{DEC_VF}",
                                         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=OW * OH * 3)
        while self.n <= k:
            buf = self.dec.stdout.read(OW * OH * 3)
            if len(buf) == OW * OH * 3: self.last = np.frombuffer(buf, np.uint8).reshape(OH, OW, 3)
            self.n += 1
        z = 1.08 + 0.06 * ease_io(k / ((CA_T1 - CA_T0) * FPS))
        return warp(self.last, z, OW / 2, OH * 0.5, (0.0, 0.0, 0.0))
CA = Cutaway()

# ---------------------------------------------------------------- frame
def render_frame(src, fo):
    tt = fo / FPS
    if CA_T0 <= tt < CA_T1:
        fr = grade(CA.frame(int(round((tt - CA_T0) * FPS))))
    else:
        z, cx, cy, c = camera(fo); fr = grade(warp(SKIN(src), z, cx, cy, c))
    for edge in (CA_T0, CA_T1):                                         # soft flash into / out of the cut-away
        d = tt - edge
        if 0 <= d < 0.08: fr = cv2.addWeighted(fr, 1 - 0.15 * (1 - d / 0.08), np.full_like(fr, 255), 0.15 * (1 - d / 0.08), 0)
    draw_subs(fr, tt)
    if tt < 0.15: fr = (fr.astype(np.float32) * (tt / 0.15)).astype(np.uint8)
    if tt > DUR - 0.4: fr = (fr.astype(np.float32) * max(0, (DUR - tt) / 0.4)).astype(np.uint8)
    return fr

def source_frames(f0, f1):
    last = None
    for i, (a, b) in enumerate(CUTS):
        p0, p1 = FSTART[i], FSTART[i + 1]
        if p1 <= f0 or p0 >= f1: continue
        skip, take = max(0, f0 - p0), min(p1, f1) - max(p0, f0)
        dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.4f}", "-i", "in/src.mov", "-an", "-vf", f"fps={FPS},{DEC_VF}", "-frames:v", str(NF[i]),
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=OW * OH * 3)
        for k in range(skip + take):
            buf = dec.stdout.read(OW * OH * 3)
            if len(buf) == OW * OH * 3: last = np.frombuffer(buf, np.uint8).reshape(OH, OW, 3)
            if k >= skip: yield last
        dec.kill(); dec.wait()

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "info":
        print("DUR", round(DUR, 2), "frames", NFR, "pieces", len(CUTS), "cutaway", round(CA_T0, 2), round(CA_T1, 2))
        for t0, t1, w in SUBS: print(f"{t0:6.2f} {t1:6.2f}", " ".join(w))
        sys.exit()
    if mode == "test":
        for t in [float(x) for x in sys.argv[2].split(",")]:
            fo = int(round(t * FPS)); SKIN.n = 0; SKIN.f = None
            o = render_frame(next(source_frames(fo, fo + 1)), fo)
            cv2.imwrite(f"test_{t:05.2f}.png", cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
        sys.exit()
    f0, f1, outp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-g", "120", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "5.2",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", outp], stdin=subprocess.PIPE)
    fo = f0
    for src in source_frames(f0, f1):
        enc.stdin.write(render_frame(src, fo).tobytes()); fo += 1
    enc.stdin.close(); enc.wait(); print("done", outp, fo - f0)
