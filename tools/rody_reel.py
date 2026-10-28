"""@mcg.va — Reels 1 «Я думала, самое страшное — это роды. Но нет» (intro to the maternity-ward story), 16:9 like the source.
Source C0973 (№5, 3:21), 3840x2160 @ 25.  Minimal look: brown-moody grade (luts/mcgva_brown_moody.cube),
punch-ins / slow push-ins and pull-outs on the face, thin Montserrat for labels and word-by-word subtitles, Gogol
handwriting for the inserts (written on with a wipe) while the picture eases into a square card on a dark blurred
backdrop, extra punch-ins inside long takes, a deeper «cinema» finish (S-curve, vignette, fine grain), a blurred end card.
Cuts sit in the pauses (energy-checked); each join gets a 25 ms raised-cosine fade so nothing clicks.
  python3 rody_reel.py SRC.mov WORK_DIR [--mic REC.m4a]  → WORK_DIR/rody_4k.mp4 (voice from the mic recording, sfx inside)
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

W, H, FPS, SR = 3840, 2160, 25, 48000
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LUT = os.path.join(ROOT, "luts", "mcgva_brown_moody.cube")                                     # make_brown_lut.py
SCRIPT_FONT = os.path.join(ROOT, "skills", "montage-svg-typography", "Шрифты", "Коллекция шрифтов", "Gogol-Regular.otf")
INK = (43, 31, 26)                                 # #2B1F1A from the brown palette
BLUSH = (220, 225, 231)                            # #DCE1E7 — the palette's pale cool grey, for highlighted words
CREAM = (232, 228, 222)                            # type on the dark square-mode backdrop
CINEMA_IN = "curves=all='0/0.025 0.25/0.2 0.5/0.48 0.75/0.79 1/0.965'"            # deeper S, lifted brown blacks
CINEMA_OUT = "vignette=angle=PI/5,noise=c0s=6:c0f=t+u"                            # soft vignette + fine moving grain
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
# inserts: (from segment, start offset, to segment, end offset, small label, handwriting)
INSERTS = [
    (3, 0.35, 6, -0.2, "", "Роды мечты"),
    (7, 0.4, 8, -0.15, "ПОСЛЕРОДОВОЕ ОТДЕЛЕНИЕ —", "мой личный ад"),
    (11, 0.1, 12, 2.6, "И ДА,", "я рожала платно"),
]

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

def cues(words, x):
    """display words (lower case) on whisper timings via a diff alignment; one word per cue, 1–2 letter words ride along"""
    starts, _ = timeline(); out = []
    for k, (a, b, _, _, text) in enumerate(SEGS):
        ws = onset_fix(x, snap_words(words, a, b), a)
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

def mic_off(t):
    (t0, o0), (t1, o1) = MIC_SYNC; return o0 + (o1 - o0) * (t - t0) / (t1 - t0)

def voice(src, work, mic=None):
    """the separately recorded mic voice (camera sound dropped), cut on the camera timeline"""
    x = FX.read(mic or src, af="highpass=f=80,afftdn=nf=-45")
    parts = []
    for a, b, *_ in SEGS:
        n = int(round((b - a) * SR))
        if mic: a = a + mic_off(a)
        y = x[int(round(a * SR)):int(round(a * SR)) + n].copy(); f = int(0.025 * SR); r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))
        y[:f] *= r[:, None]; y[-f:] *= r[::-1, None]; parts.append(y)
    v = np.concatenate(parts + [np.zeros((int(END_CARD * SR), 2), np.float32)])
    raw = f"{work}/voice_raw.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", raw], input=v.tobytes(), check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af", "loudnorm=I=-15:TP=-1.5:LRA=9", "-ar", str(SR), f"{work}/voice.wav"], check=True)
    return FX.read(f"{work}/voice.wav")

def sfx(v, total, ins_times, punch_times, hits, card_t):
    N = len(v); bus = np.zeros((N, 2), np.float32)
    def place(x, t):
        i = int(t * SR); n = min(len(x), N - i)
        if n > 0 and i >= 0: bus[i:i + n] += x[:n]
    for t in punch_times: place(FX.noise_sweep(0.55, 500, 4500, FX.db(-27)), t - 0.32)        # soft whoosh into a punch-in
    for t, dur in ins_times:                                                                    # pen on paper while writing
        n = int(dur * SR); rng = np.random.default_rng(int(t * 10)); z = rng.standard_normal(n).astype(np.float32)
        z = np.diff(np.r_[0, z]); m = 0.5 + 0.5 * np.sin(2 * np.pi * 7 * np.arange(n) / SR + rng.uniform(0, 6)) ** 2
        z = z * m * FX.env(n, 800, 3000) * FX.db(-36); place(np.stack([z, z * 0.9], 1), t)
        place(FX.pop(FX.db(-22)), t - 0.05)
    for t in hits:                                                                              # low soft hit on the heavy words
        n = int(0.9 * SR); tt = np.arange(n) / SR; y = np.sin(2 * np.pi * (52 + 30 * np.exp(-tt * 9)) * tt) * np.exp(-tt * 4.5)
        y = (y * FX.db(-18)).astype(np.float32); place(np.stack([y, y], 1), t)
    place(FX.noise_sweep(0.9, 300, 3000, FX.db(-24)), card_t - 0.45)
    place(FX.bells([1046.5, 1318.5, 1568, 2093], FX.db(-22), 0.09), card_t + 0.35)
    # duck effects a touch under the voice so they never mask a word
    lv = np.convolve(np.abs(v).mean(1), np.ones(960) / 960, "same"); g = np.where(lv > FX.db(-40), FX.db(-5), 1.0)
    return v + bus * g[:, None].astype(np.float32)

def zoom_frame(fr, z, cx=FACE[0], cy=FACE[1]):
    if z <= 1.001: return fr
    w, h = W / z, H / z; x0 = np.clip(cx * W - w / 2, 0, W - w); y0 = np.clip(cy * H - h / 2, 0, H - h)
    M = np.float32([[z, 0, -x0 * z], [0, z, -y0 * z]])
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

def render(src, work, out, mic=None):
    words = json.load(open(f"{work}/words.json"))
    v = voice(src, work, mic); x = FX.read(src)[:, 0].copy()
    starts, total = timeline(); full = total + END_CARD; n = round(full * FPS)
    cs = cues(words, x)
    fs = font("second", 124, 300)
    subs = [(a, b, [cv2.resize(p, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for s in (0.84, 0.93, 1.0)])
            for a, b, p in [(a, b, text_patch(w, fs, INK if any(k in w for k in KEY) else (255, 255, 255),
                                              pill=BLUSH if any(k in w for k in KEY) else None)) for w, a, b in cs]]
    ins = []
    for s0, o0, s1, o1, lab, hand in INSERTS:
        t0 = starts[s0] + o0; t1 = (starts[s1] + (SEGS[s1][1] - SEGS[s1][0]) + o1) if o1 <= 0 else starts[s1] + o1
        ins.append((t0, t1, label_patch(lab, 60, CREAM) if lab else None, script_patch(hand, 165 if len(hand) > 12 else 230, CREAM)))
    card_l, card_h = label_patch("В СЛЕДУЮЩЕЙ ЧАСТИ", 58), script_patch("Продолжение", 260)
    # audio: voice + effects
    punch = [starts[k] for k, s in enumerate(SEGS) if s[2] >= 1.18 and k > 0]
    punch += [starts[k] + st[0] - SEGS[k][0] for k, sts in STEPS.items() for st in sts[1:]]
    punch += [t for t0, t1, _, _ in ins for t in (t0 + 0.3, t1 + 0.3)]                       # into / out of the square
    hits = [starts[7] + (68.00 - SEGS[7][0]), starts[10] + (173.72 - SEGS[10][0])]
    mix = sfx(v, full, [(t0 + (0.45 if l is not None else 0.0), 0.8) for t0, _, l, _ in ins], punch, hits, total)
    mix *= min(1, FX.db(-1) / (np.abs(mix).max() + 1e-9))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", f"{work}/mix.wav"],
                   input=mix.astype(np.float32).tobytes(), check=True)
    wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-i", f"{work}/mix.wav", "-map", "0:v", "-map", "1:a", "-t", f"{full:.3f}",
                           "-vf", f"format=yuv420p,{CINEMA_OUT},fade=t=in:d=0.3,fade=t=out:st={full - 0.6:.3f}:d=0.6", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    sz = W * H * 3; fi = 0; last = None
    y = np.linspace(0, 1, H, dtype=np.float32); r0 = int(0.62 * H)
    spray = (256 * (1 - np.clip((y[r0:] - 0.62) / 0.38, 0, 1) ** 1.6 * 0.45)).astype(np.uint16)
    def square_k(t):
        return max([ease((t - t0) / 0.45) * (1 - ease((t - t1) / 0.45)) for t0, t1, _, _ in ins if t0 <= t < t1 + 0.45] or [0.0])
    def overlays(fr, t):
        fr[r0:] = (fr[r0:].astype(np.uint16) * spray[:, None, None] >> 8).astype(np.uint8)
        sq = 0.0
        for t0, t1, lp, hp in ins:
            if not (t0 <= t < t1 + 0.45): continue
            sq = ease((t - t0) / 0.45) * (1 - ease((t - t1) / 0.45)); yc = int(0.46 * H)
            left = int(SQUARE[1] * W + SQUARE[0] / 2 + 90); cx = (left + W - 110) // 2     # centred in the space right of the card
            if lp is not None:
                k = ease((t - t0 - 0.3) / 0.35) * (1 - ease((t - t1) / 0.3))
                blend(fr, lp, cx - lp.shape[1] // 2, int(yc - 150 - lp.shape[0] / 2 + 30 * (1 - k)), k)
            tw = t0 + (0.6 if lp is not None else 0.35); k = (t - tw) / 0.8                  # handwriting: wipe on
            if k > 0:
                w = hp.shape[1]; cut = int(w * min(1, ease(k)) * 1.1); p = hp.copy()
                if cut < w:
                    ramp = np.clip((cut - np.arange(w)) / 60, 0, 1)[None, :, None]; p[..., 3:4] *= ramp
                blend(fr, p, max(left - 40, min(cx - w // 2, W - 110 - w)), int(yc + 30 - hp.shape[0] / 2), 1 - ease((t - t1) / 0.3))
        for a, b, ps in subs:
            if a <= t < b:
                f = int((t - a) * FPS); p = ps[min(f, 2)]; sx = W / 2 + (SQUARE[1] * W - W / 2) * sq   # under the square while it is on
                blend(fr, p, int(sx - p.shape[1] / 2), H - 300 - p.shape[0] // 2, min(1, (f + 1) / 2))
    for k, (a, b, z0, z1, _) in enumerate(SEGS):
        m = round((starts[k] + b - a) * FPS) - round(starts[k] * FPS)
        rd = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-i", src, "-frames:v", str(m),
                               "-vf", f"lut3d=file={LUT}:interp=tetrahedral,{CINEMA_IN}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                              stdout=subprocess.PIPE, bufsize=sz * 2)
        for j in range(m):
            buf = rd.stdout.read(sz)
            fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy() if len(buf) == sz else last.copy()
            last = fr; u = j / max(m - 1, 1)
            ts = a + j / FPS
            if k in STEPS:                                                                   # punch-in steps + slow drift
                st = [p for p in STEPS[k] if p[0] <= ts + 1e-6][-1]; nx = [p[0] for p in STEPS[k] if p[0] > ts] + [b]
                z = st[1] * (1 + 0.03 * ease((ts - st[0]) / max(nx[0] - st[0], 0.1)))
            elif z0 != z1 and k == 8: z = z0 + (z1 - z0) * ease(j / (0.3 * FPS))           # «хапнула»: a quick punch
            else: z = z0 + (z1 - z0) * ease(u)
            t = fi / FPS; fr = square_frame(zoom_frame(fr, z), square_k(t)); overlays(fr, t); wr.stdin.write(fr.tobytes()); fi += 1
        rd.kill()
    # end card: the last frame freezes, blurs and veils, then «Продолжение / в следующей части»
    still = zoom_frame(last, SEGS[-1][3])
    for j in range(n - fi):
        t = fi / FPS; k = ease(j / (0.5 * FPS))
        sm = cv2.GaussianBlur(cv2.resize(still, (480, 270), interpolation=cv2.INTER_AREA), (0, 0), 6)
        bl = cv2.resize(sm, (W, H)); fr = cv2.addWeighted(still, 1 - k, bl, k, 0)
        fr = cv2.addWeighted(fr, 1 - 0.45 * k, np.full_like(fr, 252), 0.45 * k, 0)
        kt = ease((j - 0.35 * FPS) / (0.9 * FPS))
        if kt > 0:
            w = card_h.shape[1]; p = card_h.copy(); cut = int(w * kt * 1.1)
            if cut < w: p[..., 3:4] *= np.clip((cut - np.arange(w)) / 60, 0, 1)[None, :, None]
            blend(fr, p, (W - w) // 2, int(H * 0.44 - card_h.shape[0] / 2))
        kl = ease((j - 1.0 * FPS) / (0.5 * FPS))
        if kl > 0: blend(fr, card_l, (W - card_l.shape[1]) // 2, int(H * 0.60 - card_l.shape[0] / 2 + 30 * (1 - kl)), kl)
        wr.stdin.write(fr.tobytes()); fi += 1
    wr.stdin.close(); wr.wait()

def cover(src, work, at=60.0):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", src, "-frames:v", "1", "-vf", f"lut3d=file={LUT},{CINEMA_IN},vignette=angle=PI/5",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(H, W, 3).copy(); fr = zoom_frame(fr, 1.08, 0.47, 0.45)
    cx = int(0.80 * W); y = int(0.30 * H)
    for line in ("САМОЕ СТРАШНОЕ", "НАЧАЛОСЬ"):
        p = label_patch(line, 92); blend(fr, p, cx - p.shape[1] // 2, y - p.shape[0] // 2); y += 150
    p = script_patch("после родов", 250); blend(fr, p, min(cx - p.shape[1] // 2, W - 110 - p.shape[1]), y + 90 - p.shape[0] // 2)
    Image.fromarray(fr).save(f"{work}/cover_4k.jpg", quality=95, subsampling=0)

if __name__ == "__main__":
    src, work = sys.argv[1:3]
    mic = sys.argv[sys.argv.index("--mic") + 1] if "--mic" in sys.argv else None
    if "--cover" in sys.argv: cover(src, work)
    else: render(src, work, f"{work}/rody_4k.mp4", mic)
