"""@mcg.va — Reels 1 «Я думала, самое страшное — это роды. Но нет» (intro to the maternity-ward story), 16:9 like the source.
Source C0973 (№5, 3:21), 3840x2160 @ 25.  Minimal look: brown-moody grade (luts/mcgva_brown_moody.cube),
punch-ins / slow push-ins and pull-outs on the face, thin Montserrat for labels and word-by-word subtitles, Gogol
handwriting drawn on by a pencil (with hand-drawn underlines and sparkles), labels typed on with key clicks, one
insert where the picture eases into a square card on a dark blurred backdrop, micro shakes on the heavy words, a thin
progress line, key-word subtitles that bounce in, hand-drawn animated doodles (?!, hospital, clapper that claps, dream
cloud, pin, ₽ coin, magic wand, flames, heart) drawn by the pencil on the free wall, real CC0 key clicks for typing, extra punch-ins inside long takes, a deeper «cinema» finish (S-curve, vignette, fine grain), a blurred end card.
Cuts sit in the pauses (energy-checked); each join gets a 25 ms raised-cosine fade so nothing clicks.
  python3 rody_reel.py SRC.mov WORK_DIR [--mic REC.m4a] [--retouch MP_MODELS_DIR]  → WORK_DIR/rody_4k.mp4
  (voice from the mic recording, sfx inside; --retouch: natural skin + soft makeup via face_retouch.py, mediapipe venv)
  python3 rody_reel.py SRC.mov WORK_DIR --cover    → WORK_DIR/cover_4k.jpg
"""
import difflib, json, os, re, subprocess, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import font
from posuda_compose import text_patch, blend, ease
import posuda_sfx as FX
import doodles

W, H, FPS, SR = 3840, 2160, 25, 48000
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LUT = os.path.join(ROOT, "luts", "mcgva_brown_moody.cube")                                     # make_brown_lut.py
SCRIPT_FONT = os.path.join(ROOT, "skills", "montage-svg-typography", "Шрифты", "Коллекция шрифтов", "Gogol-Regular.otf")
INK = (43, 31, 26)                                 # #2B1F1A from the brown palette
BLUSH = (220, 225, 231)                            # #DCE1E7 — the palette's pale cool grey, for highlighted words
CREAM = (232, 228, 222)                            # type on the dark square-mode backdrop
CINEMA_IN = "curves=all='0/0.025 0.25/0.2 0.5/0.48 0.75/0.79 1/0.965'"            # deeper S, lifted brown blacks
CINEMA_OUT = "vignette=angle=PI/5,noise=c0s=6:c0f=t+u"                            # soft vignette + fine moving grain
GRADE = True                                       # --no-grade: source colours untouched (no LUT, curves, vignette, grain)
YUV2RGB = "scale=in_color_matrix=bt709:in_range=tv,format=rgb24"                    # exact BT.709 both ways, no colour shift
RGB2YUV = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "iec61966-2-4", "-color_range", "tv"]  # as the camera
SQUARE = (1440, 0.31, 0.46)                        # square-mode card: side, centre x, centre y (fractions)
# extra punch-ins inside long takes: segment → [(source time, zoom)]; the zoom drifts +3 % inside each step
STEPS = {2: [(37.72, 1.00), (40.80, 1.28)], 4: [(51.06, 1.00), (53.82, 1.30)],
         7: [(64.50, 1.00), (67.95, 1.32)], 12: [(189.56, 1.22), (193.62, 1.40)]}
FACE = (0.485, 0.42)                               # zoom centre (fraction of the frame): her face sits in the upper third
END_CARD = 3.0

# (source in, source out, zoom start, zoom end, subtitle text as said, lightly corrected)
SEGS = [
    (29.62, 33.12, 1.00, 1.06, "я думала что самое сложное это действительно роды"),
    (33.98, 36.80, 1.25, 1.25, "но как бы не так"),
    (37.72, 45.92, 1.00, 1.05, "для меня оказалось самым сложным это то что происходит в послеродовом отделении"),
    (47.10, 50.45, 1.18, 1.24, "роды у меня прошли просто шикарно"),
    (51.06, 56.18, 1.00, 1.05, "я считаю и после таких родов я готова идти за вторым"),
    (58.66, 60.10, 1.30, 1.30, "у меня роды были как в кино"),
    (60.88, 63.28, 1.12, 1.00, "о таком я даже не могла и мечтать"),
    (64.50, 70.80, 1.00, 1.06, "но к сожалению у меня был ужасный опыт в послеродовом отделении"),
    (71.62, 74.50, 1.30, 1.45, "вот там я хапнула по полной"),
    (170.42, 171.60, 1.00, 1.00, "я не буду томить"),
    (172.26, 174.86, 1.20, 1.35, "я рожала в роддоме юдина"),
    (182.52, 184.20, 1.00, 1.04, "я рожала платно"),
    (189.56, 195.84, 1.25, 1.12, "нет я не требую какого-то волшебного отношения если я рожаю платно но точно не такого что было"),
]
KEY = ("сложн", "роды", "шикарно", "кино", "мечтать", "послеродов", "ужасный", "хапнула", "полной", "томить", "юдина",
       "платно", "волшебного", "вторым")
# inserts: (from segment, start offset, to segment, end offset, small typed label, handwriting, layout, doodle)
INSERTS = [
    (3, 0.35, 6, -0.2, "", "Роды мечты", "wall", "stars"),
    (7, 0.4, 8, -0.15, "ПОСЛЕРОДОВОЕ ОТДЕЛЕНИЕ —", "мой личный ад", "square", "underline"),
    (11, 0.1, 12, 2.6, "И ДА,", "я рожала платно", "wall", "double"),
]
TYPE_STEP = 0.05                                   # one typed letter of a label
KEYS_DIR = os.path.join(ROOT, "sfx", "keyboard")   # real key presses, CC0 (sfx/keyboard/LICENSE.txt)
# hand-drawn animated doodles on the free wall left of her: (segment, source from, source to, icon, colour key)
DOODLES = [(1, 34.05, 36.80, "qmark"), (2, 42.95, 45.92, "hospital"), (5, 58.70, 61.30, "clapper"), (6, 61.75, 63.28, "cloud"),
           (10, 172.95, 174.86, "pin"), (11, 182.70, 184.20, "ruble"), (12, 189.60, 190.60, "ruble"), (12, 191.00, 193.55, "wand")]
DOODLE_BOX = (0.085, 0.17, 0.25)                   # x, y (fractions of the frame), size (fraction of the height)
HITS_SRC = [(7, 68.00), (8, 73.12), (10, 173.72), (11, 183.50)]   # micro shake + low hit: «ужасный», «хапнула», «Юдина», «платно»

# multi-source projects (--project NAME → tools/NAME.py overrides the UPPER_CASE settings above)
SEG_SRC = None                                     # per segment: key into SOURCES (None → every segment from "main")
SOURCES = None                                     # key → dict(video, mic, sync=((t0, off0), (t1, off1)), words)
PUNCH_SEGS = {8}                                   # segments whose zoom snaps in over 0.3 s instead of drifting
SQUARE_ICON = "flames"                             # doodle drawn under the square-mode insert
CARD_SCRIPT, CARD_LABEL = "Продолжение", "В СЛЕДУЮЩЕЙ ЧАСТИ"
COVER_LINES, COVER_SCRIPT, COVER_AT = ("САМОЕ СТРАШНОЕ", "НАЧАЛОСЬ"), "после родов", ("main", 60.0)
OUT_NAME = "rody_4k.mp4"
COVER_SCRIPT_PX = 250

def seg_src(k): return SEG_SRC[k] if SEG_SRC else "main"

def timeline():
    t, out = 0.0, []
    for s in SEGS: out.append(t); t += s[1] - s[0]
    return out, t

def snap_words(words, a, b):
    """whisper words inside [a, b]; a start that follows a pause is moved to the real onset (energy), never late"""
    ws = [list(w) for w in words if a - 0.3 <= w[1] < b]
    return ws

def onset_fix(x, ws, a):
    hop = 480; prev = a
    for w in ws:
        if w[1] - prev > 0.12 or w is ws[0]:
            lo, hi = int(max(prev, a) * SR), int((w[1] + 0.1) * SR)
            if hi > lo + hop:
                e = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(lo, hi - hop, hop)])
                floor = np.percentile(e, 10); on = np.argmax(e > max(floor * 4, 10 ** (-40 / 20)))
                if e.max() > floor * 4: w[1] = min(w[1], (lo + on * hop) / SR)
        prev = w[2]
    return ws

def cues(srcd):
    """display words (lower case) on whisper timings via a diff alignment; one word per cue, 1–2 letter words ride along"""
    starts, _ = timeline(); out = []
    for k, (a, b, _, _, text) in enumerate(SEGS):
        sd = srcd[seg_src(k)]; ws = onset_fix(sd["x"], snap_words(sd["wl"], a, b), a)
        src = [re.sub(r"[^\w-]", "", w[0].lower()) for w in ws]; dst = text.split()
        sm = difflib.SequenceMatcher(a=src, b=dst, autojunk=False); tm = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal": tm += [(ws[i][1], ws[i][2]) for i in range(i1, i2)]
            elif j2 > j1:
                t0 = ws[i1][1] if i1 < len(ws) else (ws[-1][2] if ws else a)
                t1 = ws[i2 - 1][2] if i2 > i1 else t0 + 0.3 * (j2 - j1)
                for j in range(j2 - j1): tm.append((t0 + (t1 - t0) * j / (j2 - j1), t0 + (t1 - t0) * (j + 1) / (j2 - j1)))
        for wd, (s, e) in zip(dst, tm):
            s, e = max(s, a), min(max(e, s + 0.12), b)
            out.append([wd, starts[k] + s - a, starts[k] + e - a])
    merged, carry = [], None
    for w in out:
        if carry: w = [carry[0] + " " + w[0], carry[1], w[2]]; carry = None
        if len(w[0]) <= 2: carry = w; continue
        merged.append(w)
    if carry: merged.append(carry)
    for p, q in zip(merged, merged[1:]): q[1] = max(q[1], p[1] + 0.14)           # whisper sometimes gives zero-length words
    for p, q in zip(merged, merged[1:]): p[2] = q[1] if q[1] - p[2] < 0.5 else p[2] + 0.25
    return [(w, s - 0.05, e - 0.05) for w, s, e in merged]

MIC_SYNC = ((35.0, 0.01492), (192.0, 0.01340))      # mic time = camera time + offset (cross-correlated, ~1 ms drift)

def mic_off(t, sync=None):
    (t0, o0), (t1, o1) = sync or MIC_SYNC; return o0 + (o1 - o0) * (t - t0) / (t1 - t0)

def voice(srcd, work):
    """the separately recorded mic voice (camera sound dropped), cut on the camera timeline"""
    xs = {k: FX.read(d["mic"] or d["video"], af="highpass=f=80,afftdn=nf=-45") for k, d in srcd.items()}
    parts = []
    for k, (a, b, *_) in enumerate(SEGS):
        n = int(round((b - a) * SR)); d = srcd[seg_src(k)]; x = xs[seg_src(k)]
        if d["mic"]: a = a + mic_off(a, d.get("sync"))
        y = x[int(round(a * SR)):int(round(a * SR)) + n].copy(); f = int(0.025 * SR); r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))
        y[:f] *= r[:, None]; y[-f:] *= r[::-1, None]; parts.append(y)
    v = np.concatenate(parts + [np.zeros((int(END_CARD * SR), 2), np.float32)])
    raw = f"{work}/voice_raw.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", raw], input=v.tobytes(), check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af", "loudnorm=I=-15:TP=-1.5:LRA=9", "-ar", str(SR), f"{work}/voice.wav"], check=True)
    return FX.read(f"{work}/voice.wav")

def pencil_noise(dur, seed, gain):
    """graphite on paper: band-passed grain in short strokes"""
    n = int(dur * SR); rng = np.random.default_rng(seed); z = rng.standard_normal(n).astype(np.float32)
    F = np.fft.rfft(z); f = np.fft.rfftfreq(n, 1 / SR); F *= np.exp(-0.5 * ((np.log(f + 1) - np.log(3800)) / 0.45) ** 2)
    z = np.fft.irfft(F, n).astype(np.float32); env_ = np.zeros(n, np.float32); i = 0
    while i < n:                                                          # strokes 60–170 ms, tiny lifts between
        L = int(rng.uniform(0.06, 0.17) * SR); e = np.sin(np.linspace(0, np.pi, min(L, n - i))) ** 0.6 * rng.uniform(0.6, 1.0)
        env_[i:i + len(e)] = e; i += L + int(rng.uniform(0.015, 0.05) * SR)
    grain = np.abs(np.convolve(rng.standard_normal(n), np.ones(120) / 120, "same")).astype(np.float32)
    y = z * env_ * (0.6 + grain * 2) * FX.env(n, 300, 1200)
    return np.stack([y, y * 0.92], 1) * (gain / (np.abs(y).max() + 1e-9))

_KEYS = []
def key_sample(rng, gain):
    """a recorded key press (onset-aligned, peak-normalised), random pick, slight level/pan variation"""
    if not _KEYS:
        for f in sorted(os.listdir(KEYS_DIR)):
            if not f.endswith(".wav"): continue
            x = FX.read(os.path.join(KEYS_DIR, f))[:, 0]; e = np.abs(x); on = max(int(np.argmax(e > e.max() * 0.2)) - int(0.004 * SR), 0)
            y = x[on:on + int(0.16 * SR)]; y = y * FX.env(len(y), 0, int(0.03 * SR)); _KEYS.append(y / (np.abs(y).max() + 1e-9))
    y = _KEYS[rng.integers(len(_KEYS))] * gain * rng.uniform(0.7, 1.0); pan = rng.uniform(0.4, 0.6)
    return np.stack([y * (1 - pan) * 2, y * pan * 2], 1).astype(np.float32)

def clap_snd(gain):
    """the clapper shutting: a dry wooden clack"""
    n = int(0.12 * SR); t = np.arange(n) / SR; rng = np.random.default_rng(2)
    z = np.diff(rng.standard_normal(n + 1)).astype(np.float32) * np.exp(-t / 0.006)
    y = z * 0.7 + np.sin(2 * np.pi * 1150 * t) * np.exp(-t * 70) * 0.5 + np.sin(2 * np.pi * 420 * t) * np.exp(-t * 45) * 0.4
    y *= gain / np.abs(y).max(); return np.stack([y, y], 1).astype(np.float32)

def key_click(rng, gain, space=False):
    """one key of a soft keyboard: click + small body resonance + thock, and the key coming back up"""
    n = int(0.09 * SR); t = np.arange(n) / SR; y = np.zeros(n, np.float32)
    for t0, g in ((0.0, 1.0), (rng.uniform(0.03, 0.045), 0.35)):
        i = int(t0 * SR); tt = t[: n - i]
        c = np.diff(rng.standard_normal(len(tt) + 1)).astype(np.float32) * np.exp(-tt / 0.0012)
        body = np.sin(2 * np.pi * rng.uniform(1700, 2300) * tt) * np.exp(-tt * 110) * 0.35
        thock = np.sin(2 * np.pi * (140 if space else rng.uniform(190, 240)) * tt) * np.exp(-tt * 55) * 0.6
        y[i:] += (c * 0.5 + body + thock) * g
    y *= gain * rng.uniform(0.75, 1.0) / (np.abs(y).max() + 1e-9); pan = rng.uniform(0.4, 0.6)
    return np.stack([y * (1 - pan) * 2, y * pan * 2], 1)

def sfx(v, keys, pencils, punch_times, hits, ticks, card_t, claps=(), twinkles=()):
    N = len(v); bus = np.zeros((N, 2), np.float32); rng = np.random.default_rng(9)
    def place(x, t):
        i = int(t * SR); n = min(len(x), N - i)
        if n > 0 and i >= 0: bus[i:i + n] += x[:n]
    for t in punch_times: place(FX.noise_sweep(0.55, 500, 4500, FX.db(-27)), t - 0.32)        # soft whoosh into a punch-in
    for t, space in keys: place(key_sample(rng, FX.db(-17)), t)                                 # labels type on: real key clicks
    for t in claps: place(clap_snd(FX.db(-16)), t)
    for t in twinkles: place(FX.bells([2093, 2637, 3136], FX.db(-27), 0.06), t)
    for k, (t, dur) in enumerate(pencils): place(pencil_noise(dur, 40 + k, FX.db(-22)), t)    # pencil on the screen
    for t in hits:                                                                              # low soft hit on the heavy words
        n = int(0.9 * SR); tt = np.arange(n) / SR; y = np.sin(2 * np.pi * (52 + 30 * np.exp(-tt * 9)) * tt) * np.exp(-tt * 4.5)
        y = (y * FX.db(-18)).astype(np.float32); place(np.stack([y, y], 1), t)
    for t in ticks: place(FX.pop(FX.db(-30)), t)                                                # tiny tick on highlighted words
    place(FX.noise_sweep(0.9, 300, 3000, FX.db(-24)), card_t - 0.45)
    place(FX.bells([1046.5, 1318.5, 1568, 2093], FX.db(-24), 0.09), card_t + 2.2)
    lv = np.convolve(np.abs(v).mean(1), np.ones(960) / 960, "same"); g = np.where(lv > FX.db(-40), FX.db(-4), 1.0)
    return v + bus * g[:, None].astype(np.float32)

def zoom_frame(fr, z, cx=FACE[0], cy=FACE[1], shake=(0.0, 0.0)):
    if z <= 1.001 and shake == (0.0, 0.0): return fr
    z = max(z, 1.0 + 0.012 * (shake != (0.0, 0.0)))                     # a hair of zoom so a shake never shows an edge
    w, h = W / z, H / z; x0 = np.clip(cx * W - w / 2, 0, W - w); y0 = np.clip(cy * H - h / 2, 0, H - h)
    M = np.float32([[z, 0, -x0 * z + shake[0]], [0, z, -y0 * z + shake[1]]])
    return cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR)

def script_patch(text, size, col=INK):
    f = ImageFont.truetype(SCRIPT_FONT, size); return text_patch(text, f, col, sh=0.0)

def square_frame(fr, k):
    """k 0..1: the full frame eases into a square card on the left over its own blurred, darkened copy"""
    if k <= 0.001: return fr
    side, fx, fy = SQUARE; s = side; cxs, cys = fx * W, fy * H
    # display rect: full frame → square; content rect: full frame → a square of the frame around her face
    dw, dh = W + (s - W) * k, H + (s - H) * k; dx, dy = (W / 2) + (cxs - W / 2) * k - dw / 2, (H / 2) + (cys - H / 2) * k - dh / 2
    cw = W + (H - W) * k; ccx = W / 2 + (FACE[0] * W - W / 2) * k; cx0 = float(np.clip(ccx - cw / 2, 0, W - cw))
    content = fr[:, int(cx0):int(cx0 + cw)]
    card = cv2.resize(content, (max(2, int(dw)), max(2, int(dh))), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(cv2.resize(fr, (384, 216), interpolation=cv2.INTER_AREA), (0, 0), 9)
    bg = cv2.resize(sm, (W, H)).astype(np.float32) * (1 - 0.55 * k) + np.array([43, 31, 26], np.float32) * 0.25 * k
    bg = bg.clip(0, 255).astype(np.uint8)
    sh = np.zeros((H // 8, W // 8), np.uint8); cv2.rectangle(sh, (int(dx / 8), int((dy + 40) / 8)), (int((dx + dw) / 8), int((dy + dh + 40) / 8)), int(170 * k), -1)
    sh = cv2.resize(cv2.GaussianBlur(sh, (0, 0), 6), (W, H)).astype(np.uint16)
    bg = (bg.astype(np.uint16) * (255 - sh[..., None]) >> 8).astype(np.uint8)
    x0, y0 = int(round(dx)), int(round(dy)); h, w = card.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    bg[ya:yb, xa:xb] = card[ya - y0:yb - y0, xa - x0:xb - x0]
    return bg

def tracked(text, f, track, col=INK):
    """letter-spaced line as an RGBA patch (tracking in px between glyphs)"""
    adv = [f.getlength(c) + track for c in text]; w = int(sum(adv) - track) + 80; h = int(f.size * 1.6) + 80
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 40
    for c, a in zip(text, adv): d.text((x, h / 2), c, font=f, fill=col + (255,), anchor="lm"); x += a
    return np.asarray(im, np.float32)

def label_patch(text, size=62, col=INK):
    return tracked(text, font("second", size, 300), size * 0.28, col)

def pencil_sprite(scale=1.0):
    """a small pencil, tip at the returned (tx, ty) inside the RGBA patch, body up and to the right"""
    S = int(420 * scale); im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    a = np.deg2rad(52); u = np.array([np.cos(a), -np.sin(a)]); nrm = np.array([np.sin(a), np.cos(a)])
    tip = np.array([50 * scale, S - 50 * scale]); L, w = 330 * scale, 20 * scale
    P = lambda s_, o: tuple(tip + u * s_ * L + nrm * o * w)
    sh = Image.new("RGBA", (S, S), (0, 0, 0, 0)); ds = ImageDraw.Draw(sh)
    ds.polygon([tuple(np.array(P(0.0, 0)) + (14, 14)), tuple(np.array(P(0.2, 1)) + (14, 14)), tuple(np.array(P(1, 1)) + (14, 14)),
                tuple(np.array(P(1, -1)) + (14, 14)), tuple(np.array(P(0.2, -1)) + (14, 14))], fill=(20, 12, 8, 90))
    im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(10 * scale))); d = ImageDraw.Draw(im)
    d.polygon([P(0.0, 0), P(0.2, 1), P(0.2, -1)], fill=(226, 200, 166, 255))                     # sharpened wood
    d.polygon([P(0.0, 0), P(0.06, 0.3), P(0.06, -0.3)], fill=(52, 42, 38, 255))                 # graphite
    d.polygon([P(0.2, 1), P(0.86, 1), P(0.86, -1), P(0.2, -1)], fill=(139, 107, 92, 255))       # #8B6B5C body
    d.polygon([P(0.2, 0.15), P(0.86, 0.15), P(0.86, -0.25), P(0.2, -0.25)], fill=(160, 128, 112, 255))
    d.polygon([P(0.86, 1), P(0.92, 1), P(0.92, -1), P(0.86, -1)], fill=(196, 192, 186, 255))    # ferrule
    d.polygon([P(0.92, 1), P(1.0, 0.9), P(1.0, -0.9), P(0.92, -1)], fill=(197, 162, 156, 255))  # eraser
    return np.asarray(im, np.float32), (int(tip[0]), int(tip[1]))

def ink_track(p):
    """per column of a text patch: the vertical centre of the ink (where the pencil tip should be)"""
    a = p[..., 3]; s_ = a.sum(0); ys = (a * np.arange(a.shape[0])[:, None]).sum(0) / np.maximum(s_, 1e-6)
    idx = np.where(s_ > 1)[0]
    if len(idx) == 0: return np.full(a.shape[1], a.shape[0] / 2), 0, a.shape[1]
    ys = np.interp(np.arange(a.shape[1]), idx, ys[idx]); ys = np.convolve(ys, np.ones(31) / 31, "same")
    return ys, idx[0], idx[-1]

def draw_on(fr, p, track, prog, x, y, alpha=1.0):
    """reveal a handwriting patch left→right up to prog (0..1 of the inked width); returns the pencil tip position"""
    ys, i0, i1 = track; cut = i0 + (i1 - i0) * min(max(prog, 0), 1)
    q = p.copy(); q[..., 3:4] *= np.clip((cut - np.arange(p.shape[1])) / 18, 0, 1)[None, :, None]
    blend(fr, q, x, y, alpha); ci = int(min(cut, p.shape[1] - 1))
    return x + cut, y + ys[ci]

def stroke(fr, pts, prog, col, th=7):
    """a hand-drawn polyline drawn up to prog; returns the pencil tip position"""
    pts = np.asarray(pts, np.float32); seg = np.hypot(*np.diff(pts, axis=0).T); L = seg.sum(); cum = np.r_[0, np.cumsum(seg)]
    d = L * min(max(prog, 0), 1); k = int(np.searchsorted(cum, d)); k = min(max(k, 1), len(pts) - 1)
    f = (d - cum[k - 1]) / max(seg[k - 1], 1e-6); end = pts[k - 1] + (pts[k] - pts[k - 1]) * f
    poly = np.vstack([pts[:k], end]).astype(np.int32)
    if len(poly) > 1: cv2.polylines(fr, [poly], False, col, th, cv2.LINE_AA)
    return float(end[0]), float(end[1])

def wobble_line(x0, x1, y0, y1, seed, amp=9, n=40):
    rng = np.random.default_rng(seed); xs = np.linspace(x0, x1, n); ph = rng.uniform(0, 6)
    ys = np.linspace(y0, y1, n) + amp * np.sin(np.linspace(0, 3.2, n) + ph) + rng.normal(0, 1.2, n)
    return np.stack([xs, ys], 1)

def sparkle(cx, cy, r):
    return [[(cx - r, cy), (cx + r, cy)], [(cx, cy - r), (cx, cy + r)], [(cx - r * .6, cy - r * .6), (cx + r * .6, cy + r * .6)]]

def render(srcd, work, out, retouch=None):
    for d in srcd.values(): d["wl"] = json.load(open(d["words"])); d["x"] = FX.read(d["video"])[:, 0].copy()
    v = voice(srcd, work)
    starts, total = timeline(); full = total + END_CARD; n = round(full * FPS)
    cs = cues(srcd)
    fs = font("second", 124, 300)
    subs = []
    for w, a, b in cs:                                                    # key words bounce in on a pill, the rest pop
        key = any(k in w for k in KEY)
        p = text_patch(w, fs, INK if key else (255, 255, 255), pill=BLUSH if key else None)
        sc = (0.6, 0.92, 1.12, 1.04, 1.0) if key else (0.84, 0.93, 1.0)
        subs.append((a, b, [cv2.resize(p, None, fx=q, fy=q, interpolation=cv2.INTER_AREA) for q in sc]))
    pen, pen_tip = pencil_sprite()
    keys, pencils, ins = [], [], []
    for s0, o0, s1, o1, lab, hand, mode, doodle, *sizes in INSERTS:                # optional (label px, handwriting px)
        t0 = starts[s0] + o0; t1 = (starts[s1] + (SEGS[s1][1] - SEGS[s1][0]) + o1) if o1 <= 0 else starts[s1] + o1
        col = CREAM if mode == "square" else INK
        lsz, hsz = sizes if sizes else (60, (165 if mode == "square" else 150) if len(hand) > 12 else (230 if mode == "square" else 180))
        labs = [label_patch(lab[:i], lsz, col) for i in range(1, len(lab) + 1)] if lab else []
        hp = script_patch(hand, hsz, col)
        tw = t0 + (0.35 if mode == "square" else 0.0) + (len(lab) * TYPE_STEP + 0.15 if lab else 0.1); wd = 0.07 * len(hand) + 0.35
        keys += [(t0 + (0.35 if mode == "square" else 0.0) + i * TYPE_STEP, c == " ") for i, c in enumerate(lab)]
        pencils += [(tw, wd), (tw + wd + 0.05, 0.55)]
        ins.append(dict(t0=t0, t1=t1, mode=mode, doodle=doodle, col=col, labs=labs, hp=hp, track=ink_track(hp), tw=tw, wd=wd,
                        tl=t0 + (0.35 if mode == "square" else 0.0)))
    card_h = script_patch(CARD_SCRIPT, 260); card_tr = ink_track(card_h); card_lab = CARD_LABEL
    card_ls = [label_patch(card_lab[:i], 58) for i in range(1, len(card_lab) + 1)]
    keys += [(total + 1.55 + i * TYPE_STEP, c == " ") for i, c in enumerate(card_lab)]; pencils += [(total + 0.35, 1.0)]
    dds = []
    for k, a_, b_, name in DOODLES:
        t0 = starts[k] + a_ - SEGS[k][0]; t1 = starts[k] + b_ - SEGS[k][0]; dd = min(0.9, (t1 - t0) * 0.45)
        dds.append(dict(t0=t0, t1=t1, dd=dd, name=name, seed=len(dds) + 1)); pencils.append((t0, dd))
    claps = [d_["t0"] + d_["dd"] + 0.3 for d_ in dds if d_["name"] == "clapper"]      # the bar shuts 0.3 s after drawing
    twinkles = [d_["t0"] + d_["dd"] + 0.1 for d_ in dds if d_["name"] == "wand"]
    # the square insert gets little flames under «мой личный ад»
    for d_ in ins:
        if d_["mode"] == "square":
            tf = d_["tw"] + d_["wd"] + 0.65; dds.append(dict(t0=tf, t1=d_["t1"], dd=0.6, name=SQUARE_ICON, seed=99, sq=True)); pencils.append((tf, 0.6))
    hx = (W - card_h.shape[1]) // 2 + card_h.shape[1] + 30
    dds.append(dict(t0=total + 1.4, t1=total + END_CARD + 1, dd=0.5, name="heart", seed=7, card=True)); pencils.append((total + 1.4, 0.5))
    # audio: voice + effects
    punch = [starts[k] for k, s_ in enumerate(SEGS) if s_[2] >= 1.18 and k > 0]
    punch += [starts[k] + st[0] - SEGS[k][0] for k, sts in STEPS.items() for st in sts[1:]]
    punch += [t for d_ in ins if d_["mode"] == "square" for t in (d_["t0"] + 0.3, d_["t1"] + 0.3)]
    hits = [starts[k] + ts - SEGS[k][0] for k, ts in HITS_SRC]
    ticks = [a_ for w, a_, b_ in cs if any(k in w for k in KEY)]
    mix = sfx(v, keys, pencils, punch, hits, ticks, total, claps, twinkles)
    mix *= min(1, FX.db(-1) / (np.abs(mix).max() + 1e-9))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", f"{work}/mix.wav"],
                   input=mix.astype(np.float32).tobytes(), check=True)
    wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-i", f"{work}/mix.wav", "-map", "0:v", "-map", "1:a", "-t", f"{full:.3f}",
                           "-vf", f"{RGB2YUV}," + (f"{CINEMA_OUT}," if GRADE else "") + f"fade=t=in:d=0.3,fade=t=out:st={full - 0.6:.3f}:d=0.6",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "12" if not GRADE else "17", "-profile:v", "high",
                           "-pix_fmt", "yuv420p"] + TAGS + ["-c:a", "aac", "-b:a", "320k",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    sz = W * H * 3; fi = 0; last = None
    yy = np.linspace(0, 1, H, dtype=np.float32); r0 = int(0.62 * H)
    spray = (256 * (1 - np.clip((yy[r0:] - 0.62) / 0.38, 0, 1) ** 1.6 * 0.45)).astype(np.uint16)
    def square_k(t):
        return max([ease((t - d_["t0"]) / 0.45) * (1 - ease((t - d_["t1"]) / 0.45)) for d_ in ins
                    if d_["mode"] == "square" and d_["t0"] <= t < d_["t1"] + 0.45] or [0.0])
    def shake(t):
        for h_ in hits:
            dt = t - h_
            if 0 <= dt < 0.35:
                a_ = 16 * (1 - dt / 0.35); return (a_ * np.sin(dt * 95), a_ * 0.7 * np.cos(dt * 120))
        return (0.0, 0.0)
    def put_pen(fr, x, y, bob=0.0):
        blend(fr, pen, int(x - pen_tip[0]), int(y - pen_tip[1] + bob))
    def doodle(fr, d_, t):
        if not (d_["t0"] <= t < d_["t1"] + 0.3): return
        prog = (t - d_["t0"]) / d_["dd"]
        if t > d_["t1"]: prog = 1 - (t - d_["t1"]) / 0.3                    # undraw quickly at the end
        if d_.get("sq"):
            left = int(SQUARE[1] * W + SQUARE[0] / 2 + 90); size = 300; x = (left + W - 110) // 2 - size // 2; y = int(0.46 * H) + 170; col, th = CREAM, 7
        elif d_.get("card"):
            size = 190; x = hx; y = int(H * 0.44) - size // 2 - 20; col, th = INK, 7
        else:
            size = int(DOODLE_BOX[2] * H); x = int(DOODLE_BOX[0] * W); y = int(DOODLE_BOX[1] * H); col, th = INK, 9
        tip = doodles.draw_icon(fr, d_["name"], x, y, size, prog, t, col, th, motion=max(0.0, t - d_["t0"] - d_["dd"]), seed=d_["seed"])
        if tip is not None and 0 < prog < 1 and t <= d_["t1"]: put_pen(fr, *tip, bob=4 * np.sin(t * 40))
    def insert(fr, d_, t):
        fade = 1 - ease((t - d_["t1"]) / 0.3)
        if d_["mode"] == "square":
            left = int(SQUARE[1] * W + SQUARE[0] / 2 + 90); cx = (left + W - 110) // 2; yl = int(0.46 * H) - 150; yh = int(0.46 * H) + 30
        else:
            left = 0; cx = int(0.81 * W); yl = int(0.30 * H); yh = int(0.30 * H) + 150
        pen_at = None
        if d_["labs"]:                                                             # typed label
            nch = int((t - d_["tl"]) / TYPE_STEP) + 1
            if nch > 0:
                lp = d_["labs"][min(nch, len(d_["labs"])) - 1]; full_w = d_["labs"][-1].shape[1]
                lx = max(left - 40, min(cx - full_w // 2, W - 150 - full_w)); blend(fr, lp, lx, int(yl - lp.shape[0] / 2), fade)
        hp = d_["hp"]; w = hp.shape[1]; hx = max(left - 40, min(cx - w // 2, W - 110 - w)); hy = int(yh - hp.shape[0] / 2)
        k = (t - d_["tw"]) / d_["wd"]
        if k > 0:
            tip = draw_on(fr, hp, d_["track"], k, hx, hy, fade)
            if k < 1: pen_at = tip
        ys_, i0, i1 = d_["track"]; x0, x1 = hx + i0, hx + i1
        a_ = hp[..., 3]; rows = np.where(a_.sum(1) > 1)[0]; yt, yb = hy + rows[0], hy + rows[-1]
        kd = (t - d_["tw"] - d_["wd"] - 0.05) / 0.55; th = 6 if d_["mode"] == "wall" else 5
        if kd > 0 and fade > 0.05:
            col = d_["col"]
            if d_["doodle"] in ("underline", "double"):
                L1 = wobble_line(x0 + 10, x1 - 10, yb + 34, yb + 24, int(d_["t0"] * 7))
                tip = stroke(fr, L1, kd * (1.4 if d_["doodle"] == "double" else 1.0), col, th)
                if d_["doodle"] == "double" and kd > 0.7:
                    tip = stroke(fr, wobble_line(x0 + 0.25 * (x1 - x0), x1 - 30, yb + 66, yb + 58, int(d_["t0"] * 11), 6), (kd - 0.7) / 0.5, col, th)
                if kd < 1.2: pen_at = tip
            elif d_["doodle"] == "stars":
                spots = [(x1 + 70, yt + 10, 34), (x0 - 70, yt + 60, 24), (x1 + 20, yb + 70, 22)]
                lines = [l for c_ in spots for l in sparkle(*c_)]; per = 1.0 / len(lines)
                for j, l in enumerate(lines):
                    pj = (kd - j * per) / per
                    if pj > 0: tip = stroke(fr, l, pj, col, th); pen_at = tip if pj < 1 else pen_at
        if pen_at is not None and fade > 0.5: put_pen(fr, *pen_at, bob=4 * np.sin(t * 40))
    def overlays(fr, t):
        fr[r0:] = (fr[r0:].astype(np.uint16) * spray[:, None, None] >> 8).astype(np.uint8)
        sq = square_k(t)
        for d_ in ins:
            if d_["t0"] <= t < d_["t1"] + 0.45: insert(fr, d_, t)
        for d_ in dds:
            if not d_.get("card"): doodle(fr, d_, t)
        for a_, b_, ps in subs:
            if a_ <= t < b_:
                f = int((t - a_) * FPS); p = ps[min(f, len(ps) - 1)]; sx = W / 2 + (SQUARE[1] * W - W / 2) * sq
                blend(fr, p, int(sx - p.shape[1] / 2), H - 300 - p.shape[0] // 2, min(1, (f + 1) / 2))
        pw = int(W * min(t / full, 1)); fr[0:10, :pw] = (fr[0:10, :pw] * 0.3 + np.array(CREAM) * 0.7).astype(np.uint8)   # progress line
    for k, (a, b, z0, z1, _) in enumerate(SEGS):
        m = round((starts[k] + b - a) * FPS) - round(starts[k] * FPS)
        pre = round(max(0.0, -a) * FPS)                                     # voice starts before the clip: hold frame 0
        rd = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{max(a, 0.0):.3f}", "-i", srcd[seg_src(k)]["video"], "-frames:v", str(m),
                               "-vf", YUV2RGB + (f",lut3d=file={LUT}:interp=tetrahedral,{CINEMA_IN}" if GRADE else ""), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                              stdout=subprocess.PIPE, bufsize=sz * 2)
        for j in range(m):
            buf = rd.stdout.read(sz) if (j == 0 or j >= pre) else b""          # b"" → repeat the last frame
            fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy() if len(buf) == sz else last.copy()
            if retouch is not None and len(buf) == sz: fr = retouch.apply(fr)             # natural skin + soft makeup
            last = fr; u = j / max(m - 1, 1)
            ts = a + j / FPS
            if k in STEPS:                                                                   # punch-in steps + slow drift
                st = [p for p in STEPS[k] if p[0] <= ts + 1e-6][-1]; nx = [p[0] for p in STEPS[k] if p[0] > ts] + [b]
                z = st[1] * (1 + 0.03 * ease((ts - st[0]) / max(nx[0] - st[0], 0.1)))
            elif z0 != z1 and k in PUNCH_SEGS: z = z0 + (z1 - z0) * ease(j / (0.3 * FPS))  # a quick punch («хапнула»)
            else: z = z0 + (z1 - z0) * ease(u)
            t = fi / FPS; fr = square_frame(zoom_frame(fr, z, shake=shake(t)), square_k(t)); overlays(fr, t)
            wr.stdin.write(fr.tobytes()); fi += 1
        rd.kill()
    # end card: the last frame freezes, blurs and veils; the pencil writes «Продолжение», the label types on
    still = zoom_frame(last, SEGS[-1][3])
    for j in range(n - fi):
        t = fi / FPS; k = ease(j / (0.5 * FPS))
        sm = cv2.GaussianBlur(cv2.resize(still, (480, 270), interpolation=cv2.INTER_AREA), (0, 0), 6)
        bl = cv2.resize(sm, (W, H)); fr = cv2.addWeighted(still, 1 - k, bl, k, 0)
        fr = cv2.addWeighted(fr, 1 - 0.45 * k, np.full_like(fr, 252), 0.45 * k, 0)
        kt = (t - total - 0.35) / 1.0
        if kt > 0:
            tip = draw_on(fr, card_h, card_tr, kt, (W - card_h.shape[1]) // 2, int(H * 0.44 - card_h.shape[0] / 2))
            if kt < 1: put_pen(fr, *tip, bob=4 * np.sin(t * 40))
        nch = int((t - total - 1.55) / TYPE_STEP) + 1
        if nch > 0:
            lp = card_ls[min(nch, len(card_ls)) - 1]; blend(fr, lp, (W - card_ls[-1].shape[1]) // 2, int(H * 0.60 - lp.shape[0] / 2))
        for d_ in dds:
            if d_.get("card"): doodle(fr, d_, t)
        pw = int(W * min(t / full, 1)); fr[0:10, :pw] = (fr[0:10, :pw] * 0.3 + np.array(INK) * 0.7).astype(np.uint8)
        wr.stdin.write(fr.tobytes()); fi += 1
    wr.stdin.close(); wr.wait()

def cover(srcd, work, retouch=None):
    key, at = COVER_AT
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", srcd[key]["video"], "-frames:v", "1",
                          "-vf", YUV2RGB + (f",lut3d=file={LUT},{CINEMA_IN},vignette=angle=PI/5" if GRADE else ""),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(H, W, 3).copy()
    if retouch is not None: fr = retouch.apply(fr)
    fr = zoom_frame(fr, 1.08, 0.47, 0.45)
    cx = int(0.80 * W); y = int(0.30 * H)
    for line in COVER_LINES:
        p = label_patch(line, 92); blend(fr, p, cx - p.shape[1] // 2, y - p.shape[0] // 2); y += 150
    p = script_patch(COVER_SCRIPT, COVER_SCRIPT_PX); blend(fr, p, min(cx - p.shape[1] // 2, W - 110 - p.shape[1]), y + 90 - p.shape[0] // 2)
    Image.fromarray(fr).save(f"{work}/cover_4k.jpg", quality=95, subsampling=0)

if __name__ == "__main__":
    src, work = sys.argv[1:3]
    mic = sys.argv[sys.argv.index("--mic") + 1] if "--mic" in sys.argv else None
    if "--project" in sys.argv:                                            # src is then the folder with the project's files
        import importlib; cfg = importlib.import_module(sys.argv[sys.argv.index("--project") + 1])
        globals().update({k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
        srcd = {k: dict(d, video=os.path.join(src, d["video"]), mic=os.path.join(src, d["mic"]) if d.get("mic") else None,
                        words=os.path.join(src, d["words"])) for k, d in SOURCES.items()}
    else:
        srcd = {"main": dict(video=src, mic=mic, sync=MIC_SYNC, words=f"{work}/words.json")}
    rt = None
    if "--no-grade" in sys.argv: GRADE = False
    if "--retouch" in sys.argv:                                            # needs the mediapipe venv
        import face_retouch; rt = face_retouch.Retoucher(sys.argv[sys.argv.index("--retouch") + 1])
    if "--cover" in sys.argv: cover(srcd, work, retouch=rt)
    else: render(srcd, work, f"{work}/{OUT_NAME}", rt)
