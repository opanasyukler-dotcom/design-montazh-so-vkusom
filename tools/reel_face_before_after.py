import subprocess, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from faceret import retouch2

W, H, FPS, SR = 1080, 1920, 30, 48000
SP = "/tmp/claude-0/-home-user-design-montazh-so-vkusom/b26b20a4-6ac5-50f8-8404-dcbad88c88d3/scratchpad/"
BLUE = (16, 46, 70); POWDER = (228, 210, 210); WHITE = (255, 255, 255)
def osw(s): return ImageFont.truetype(SP + "v4/fonts/Oswald-600.ttf", s)
def mont(w, s): return ImageFont.truetype(SP + f"fonts2/M{w}.ttf", s)
SUBS = []; DUR = 0
exec(open("helpers_src.py").read())

# ---------------- timeline ----------------
BL = [("hook", 2.2), ("before", 6.5), ("after", 6.5), ("compare", 3.2), ("final", 4.0)]
T0 = {}; t = 0
for n, d in BL: T0[n] = (t, t + d); t += d
DUR = t

# ---------------- sources ----------------
def read_all(src):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, 1280, 720, 3)
VB, VA = read_all("before.mov"), read_all("after.mov")
LB, LA = np.load("lm_before.npy"), np.load("lm_after.npy")
def gs(a, s):
    r = int(3 * s); k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * s * s)); k /= k.sum()
    pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)])
    return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, pad)
LB = gs(LB.reshape(len(LB), -1), 1.5).reshape(LB.shape); LA = gs(LA.reshape(len(LA), -1), 1.5).reshape(LA.shape)
RET = {}
def after_frame(i):
    i = min(i, len(VA) - 1)
    if i not in RET: RET[i] = retouch2(VA[i], LA[i], k_col=0.4, yshift=2)
    return RET[i]

# camera: face-centred crop, face width -> FW px of output
FW = 660
def cam(Lm):
    c = Lm[[1, 168, 152]].mean(0); fw = np.linalg.norm(Lm[454] - Lm[234])
    s = FW / fw                          # output px per source px
    return c, s
CAMB = gs(np.array([np.r_[cam(LB[i])[0], cam(LB[i])[1]] for i in range(len(LB))]), 6)
CAMA = gs(np.array([np.r_[cam(LA[i])[0], cam(LA[i])[1]] for i in range(len(LA))]), 6)
def place(img, c, s, zoom=1.0, cy_out=860):
    s = max(s * zoom, 1.52); c = np.array(c, float)
    c[0] = np.clip(c[0], 540 / s, 720 - 540 / s); c[1] = np.clip(c[1], cy_out / s, 1280 - (H - cy_out) / s)
    M = np.float32([[s, 0, 540 - s * c[0]], [0, s, cy_out - s * c[1]]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), M
def mp(M, p): return np.array([M[0, 0] * p[0] + M[0, 2], M[1, 1] * p[1] + M[1, 2]])

# ---------------- feature geometry ----------------
UP_R = [33, 246, 161, 160, 159, 158, 157, 173, 133]; UP_L = [263, 466, 388, 387, 386, 385, 384, 398, 362]
LO_R = [33, 7, 163, 144, 145, 153, 154, 155, 133]; LO_L = [263, 249, 390, 373, 374, 380, 381, 382, 362]
BROW_R = [70, 63, 105, 66, 107]; BROW_L = [300, 293, 334, 296, 336]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
def smooth_curve(pts, n=50):
    pts = np.asarray(pts, float); tt = np.linspace(0, 1, len(pts)); u = np.linspace(0, 1, n); k = min(3, len(pts) - 1)
    return np.stack([np.polyval(np.polyfit(tt, pts[:, i], k), u) for i in range(2)], 1)
def feats(Lm, M):
    P = lambda k: mp(M, Lm[k])
    f = {}
    for side, up, lo, br in (("r", UP_R, LO_R, BROW_R), ("l", UP_L, LO_L, BROW_L)):
        lid = np.array([P(k) for k in up]); b = np.mean([P(k) for k in br], 0); off = (b[1] - lid[:, 1].mean()) * 0.4
        c = lid.copy(); c[:, 1] += off; f["lid" + side] = smooth_curve(c[1:-1])
        low = np.array([P(k) for k in lo]); h = low[:, 1].mean() - lid[:, 1].mean()
        c = low.copy(); c[:, 1] += h * 0.9; f["und" + side] = smooth_curve(c[1:-1])
    f["cheekr"], f["cheekl"] = P(50), P(280)
    for side, ala, cor in (("r", 129, 61), ("l", 358, 291)):
        a, e = P(ala), P(cor); a = a + (e - a) * 0.05; e = e + (e - a) * 0.08
        f["nl" + side] = bez(a, e, 0.12 if side == "r" else -0.12, 30)
        f["cor" + side] = P(cor)
    f["lips"] = np.array([P(k) for k in LIPS]); f["fw"] = np.linalg.norm(P(454) - P(234))
    return f
def tag2(lines, size=30, bg=BLUE, fg=WHITE):
    f = osw(size); asc, desc = f.getmetrics(); lh = asc + desc
    w = int(max(f.getlength(l) for l in lines)) + 28; h = lh * len(lines) + 12
    im = Image.new("RGBA", (w, h), bg + (255,)); d = ImageDraw.Draw(im)
    for i, l in enumerate(lines): d.text((14, 6 + i * lh + asc), l, font=f, fill=fg + (255,), anchor="ls")
    return to_np(im)

# label specs: key, lines(before), lines(after), side, anchor feature
SPEC = [("lid", ["НАВИСШЕЕ", "ВЕКО"], ["ВЕРХНЯЯ", "БЛЕФАРОПЛАСТИКА"], "L"),
        ("und", ["ПУСТОТА", "ПОД ГЛАЗАМИ"], ["ЛИПОФИЛИНГ", "НИЖНИХ ВЕК"], "R"),
        ("cheek", ["ПЛОСКИЕ", "СКУЛЫ"], ["ЛИПОФИЛИНГ", "СКУЛ"], "L"),
        ("nl", ["ГЛУБОКИЕ", "НОСОГУБКИ"], ["НОСОГУБНЫЕ", "СКЛАДКИ"], "R"),
        ("cor", ["ОПУЩЕННЫЕ", "УГОЛКИ ГУБ"], ["ЛИПОФИЛИНГ", "УГОЛКОВ ГУБ"], "L"),
        ("lips", ["ТОНКИЕ", "ГУБЫ"], ["ГУБЫ:", "ВЕРХ И НИЗ"], "R")]
TAGB = [tag2(s[1], 28, (40, 40, 40)) for s in SPEC]; TAGA = [tag2(s[2], 28) for s in SPEC]
def anchor(key, f, side):
    s = "r" if side == "L" else "l"     # image-left label -> patient's right side
    if key in ("lid", "und", "nl"): c = f[key + s]; return c[len(c) // 2]
    if key == "cheek": return f["cheek" + s]
    if key == "cor": return f["cor" + s]
    if key == "lips": lp = f["lips"]; return lp[lp[:, 0].argmax()] + [6, 0]
def draw_feature(fr, key, f, q, al, dash):
    if key in ("lid", "und", "nl"):
        for s in "rl": stroke(fr, partial(f[key + s], q), al, 6, dash=dash or key == "und")
    elif key == "cheek":
        r = f["fw"] * 0.075
        for s in "rl":
            c = f["cheek" + s]; th = np.linspace(0, 2 * np.pi, 60)
            stroke(fr, partial(np.stack([c[0] + r * 1.4 * np.cos(th), c[1] + r * np.sin(th)], 1), q), al, 5, dash=True)
    elif key == "cor":
        for s in "rl":
            c = f["cor" + s]; d = 1 if dash else -1
            stroke(fr, partial(np.array([c + [0, -30 * d], c + [0, 22 * d]]), q), al, 5, head=q > 0.9)
    elif key == "lips":
        lp = f["lips"]; c = lp.mean(0); stroke(fr, partial(c + (lp - c) * 1.12, q), al, 5, dash=dash)
def slots(f, tags):
    ys = {}
    for side in "LR":
        items = [(i, anchor(k, f, sd)[1] + (-40 if k == "lid" else 0)) for i, (k, _, _, sd) in enumerate(SPEC) if sd == side]
        items.sort(key=lambda x: x[1]); last = -1e9
        for i, y in items:
            hh = tags[i].shape[0]; y = max(y, last + hh + 14); ys[i] = y; last = y
    return ys
def annotate(fr, f, lt, t0s, after):
    tags = TAGA if after else TAGB
    YS = slots(f, tags)
    for i, (key, _, _, side) in enumerate(SPEC):
        st = t0s + i * 0.85
        if lt < st: continue
        q, al = env(lt, st, 99, 0.35)
        draw_feature(fr, key, f, q, al, not after)
        im = tags[i]; a = anchor(key, f, side)
        x = 20 + im.shape[1] / 2 if side == "L" else W - 20 - im.shape[1] / 2
        y = np.clip(YS[i], 340, 1500)
        sx = x + (im.shape[1] / 2 if side == "L" else -im.shape[1] / 2)
        q2, al2 = env(lt, st + 0.15, 99, 0.25)
        stroke(fr, partial(bez((sx, y), a, 0.15 if side == "L" else -0.15), q2), al2, 4, head=q2 > 0.95)
        pop_img(fr, im, x, y, lt, st)

T_HOOK = Txt([("ОМОЛОЖЕНИЕ БЕЗ ПОДТЯЖКИ", "HB"), ("5 зон за одну операцию", "S")], size=58)
TAG_B = tag("ДО", 40, (40, 40, 40)); TAG_A = tag("ПОСЛЕ", 40)
T_FIN = Txt([("ЧТО СДЕЛАЛИ:", "HB"), ("— верхняя блефаропластика", "S"), ("— липофилинг нижних век и скул", "S"),
             ("— липофилинг уголков губ", "S"), ("— губы: верх и низ", "S"), ("— носогубные складки", "S")], size=54)

def panel(fr, y, h, lt, t0):
    a = min(1, max(0, (lt - t0) / 0.25)) * 0.82
    if a <= 0: return
    fr[y:y + h, 50:W - 50] = fr[y:y + h, 50:W - 50] * (1 - a) + np.float32(BLUE) / 255 * a
def render(n, lt):
    if n == "hook":
        k = int(lt / 0.22) % 2
        if k == 0: img, Lm, cs = after_frame(int(1.0 * FPS)), LA[30], CAMA[30]
        else: img, Lm, cs = VB[int(3.5 * FPS)], LB[105], CAMB[105]
        fr, M = place(img, cs[:2], cs[2], 1.0 + 0.08 * lt / 2.2)
        fr = grade(fr.astype(np.float32)[..., ::-1] / 255)
        blend(fr, TAG_A if k == 0 else TAG_B, 70, 300, 1)
        panel(fr, 1240, 175, lt, 0.45)
        T_HOOK.draw(fr, 1260, lt, 0.5, cps=34)
        return fr
    if n == "before":
        i = min(int((1.6 + min(lt, 4.45)) * FPS), len(VB) - 1)
        cs = CAMB[i]; z = 1.0 + 0.06 * lt / 6.5
        fr, M = place(VB[i], cs[:2], cs[2], z)
        f = feats(LB[i], M); fr = grade(fr.astype(np.float32)[..., ::-1] / 255)
        annotate(fr, f, lt, 0.4, False); blend(fr, TAG_B, 70, 300, 1)
        return fr
    if n == "after":
        i = min(int(min(lt, 4.9) * FPS), len(VA) - 1)
        cs = CAMA[i]; z = 1.0 + 0.06 * lt / 6.5
        fr, M = place(after_frame(i), cs[:2], cs[2], z)
        f = feats(LA[i], M); fr = grade(fr.astype(np.float32)[..., ::-1] / 255)
        annotate(fr, f, lt, 0.4, True); blend(fr, TAG_A, 70, 300, 1)
        return fr
    if n == "compare":
        ib, ia = len(VB) - 1, int(2.0 * FPS)
        fb, _ = place(VB[ib], CAMB[ib][:2], CAMB[ib][2], 1.12, 880); fa, _ = place(after_frame(ia), CAMA[ia][:2], CAMA[ia][2], 1.12, 880)
        fb = grade(fb.astype(np.float32)[..., ::-1] / 255); fa = grade(fa.astype(np.float32)[..., ::-1] / 255)
        xpos = int(W * (0.5 + 0.42 * np.sin(np.pi * 2 * min(lt / 2.6, 1) - np.pi / 2) * -1)) if lt < 2.6 else W // 2
        fr = fa.copy(); fr[:, :xpos] = fb[:, :xpos]
        cv2.line(fr, (xpos, 0), (xpos, H), (1, 1, 1), 5, cv2.LINE_AA)
        blend(fr, TAG_B, 70, 300, 1); blend(fr, TAG_A, W - 70 - TAG_A.shape[1], 300, 1)
        return fr
    if n == "final":
        i = min(int((5.0 + lt * 1.0) * FPS), len(VA) - 1)
        cs = CAMA[int(4.9 * FPS)]
        fr, M = place(after_frame(i), cs[:2], cs[2], 1.0 + 0.04 * lt / 4, 720)
        fr = grade(fr.astype(np.float32)[..., ::-1] / 255)
        panel(fr, 1220, 400, lt, 0.15)
        T_FIN.draw(fr, 1240, lt, 0.2, cps=40)
        return fr

enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "video_only.mp4"], stdin=subprocess.PIPE)
for fi in range(int(round(DUR * FPS))):
    tt = fi / FPS
    n = next(k for k, (a, b) in T0.items() if a <= tt < b) if tt < DUR else BL[-1][0]
    lt = tt - T0[n][0]
    fr = render(n, lt)
    if n != "hook" and lt < 0.25:
        e = ease(lt / 0.25); fr = kenburns(fr, 1 + 0.08 * (1 - e), 540, 900); fr = fr * (1 - 0.45 * (1 - e)) + 0.45 * (1 - e)
    blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
    enc.stdin.write((np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
enc.stdin.close(); enc.wait()

# ---------------- music + sfx ----------------
rng = np.random.default_rng(3); N = int(DUR * SR) + SR
def note(f): return 440 * 2 ** ((f - 69) / 12)
music = np.zeros(N); chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]; cl = 2.0; tt_ = np.arange(int(cl * SR)) / SR
envp = np.minimum(1, tt_ / 0.3) * np.minimum(1, (cl - tt_) / 0.4)
for i in range(int(DUR / cl) + 2):
    ch = chords[i % 4]; s = int(i * cl * SR)
    if s >= N: continue
    seg = sum(np.sin(2 * np.pi * note(m) * tt_) * 0.5 + np.sin(2 * np.pi * note(m) * 2 * tt_) * 0.08 for m in ch + [ch[0] - 12]) * envp
    e_ = min(N, s + len(seg)); music[s:e_] += seg[:e_ - s]
    for j in range(8):
        ps = s + int(j * cl / 8 * SR)
        if ps >= N: continue
        pl = np.arange(int(0.3 * SR)) / SR; pk = np.sin(2 * np.pi * note((ch + [ch[0] + 12])[j % 4] + 12) * pl) * np.exp(-pl * 10) * 0.4
        e_ = min(N, ps + len(pk)); music[ps:e_] += pk[:e_ - ps]
    for j in range(4):   # soft kick on beats for a dynamic feel
        ps = s + int(j * cl / 4 * SR)
        if ps >= N: continue
        kl = np.arange(int(0.18 * SR)) / SR; kk = np.sin(2 * np.pi * np.cumsum(50 + 70 * np.exp(-kl * 30)) / SR) * np.exp(-kl * 18) * 0.9
        e_ = min(N, ps + len(kk)); music[ps:e_] += kk[:e_ - ps]
music = music[:int(DUR * SR)]; music = music / np.abs(music).max() * 0.22
def lp(x, cut):
    y = np.zeros_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * cut / SR)
    for i in range(len(x)): s += a[i] * (x[i] - s); y[i] = s
    return y
def whoosh(d=0.5, lo=300, hi=5000, pk=0.6):
    n = int(d * SR); x_ = np.linspace(0, 1, n); e = np.where(x_ < pk, (x_ / pk) ** 2, ((1 - x_) / (1 - pk)) ** 1.5)
    cut = lo + (hi - lo) * np.sin(np.pi * np.clip(x_ / (pk * 2), 0, 1)) ** 2
    x = rng.standard_normal(n); y = (lp(x, cut) - lp(x, cut * 0.25)) * e; return y / np.abs(y).max()
def pop():
    n = int(0.09 * SR); x = np.arange(n) / SR; y = np.sin(2 * np.pi * np.cumsum(260 + 700 * np.exp(-x * 60)) / SR) * np.exp(-x * 45); return y / np.abs(y).max()
def ding():
    n = int(1.4 * SR); x = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * fq * x) * np.exp(-x * k) for fq, a, k in [(1318.5, 1, 3), (1975.5, .5, 4), (2637, .3, 6), (659.25, .4, 2.5)])
    return y * np.minimum(1, x / 0.004) / np.abs(y).max()
def tick():
    n = int(0.03 * SR); x = rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 200); return x / np.abs(x).max()
sfx = np.zeros(len(music) + SR)
def put(s, t_, g): i = int(max(0, t_) * SR); seg = sfx[i:i + len(s)]; seg += s[:len(seg)] * g
WB, PP, DG, TK = whoosh(0.55), pop(), ding(), tick()
for k in range(10): put(TK, k * 0.22, 0.12)
for n_ in ("before", "after", "compare", "final"): put(WB, T0[n_][0] - 0.3, 0.3)
for n_ in ("before", "after"):
    for i in range(6): put(PP, T0[n_][0] + 0.4 + i * 0.85, 0.25)
put(DG, T0["after"][0] + 0.05, 0.15); put(DG, T0["final"][0] + 0.3, 0.12)
mix = np.clip(music + sfx[:len(music)], -1, 1)
with wave.open("mix.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("dur", DUR)
