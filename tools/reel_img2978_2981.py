import subprocess, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 48000
SP = "/tmp/claude-0/-home-user-design-montazh-so-vkusom/b26b20a4-6ac5-50f8-8404-dcbad88c88d3/scratchpad/"
BLUE = (16, 46, 70); POWDER = (228, 210, 210); WHITE = (255, 255, 255)
def osw(s): return ImageFont.truetype(SP + "v4/fonts/Oswald-600.ttf", s)
def mont(w, s): return ImageFont.truetype(SP + f"fonts2/M{w}.ttf", s)

# ---------------- audio plan ----------------
def rd(p):
    w = wave.open(p); return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
AUD = {"81": rd("a81.wav"), "78": rd("a78.wav")}
# block: name, audio segments (src,a,b), hold, words [(src_time, word)]
BL = [
 dict(n="hook", aud=[("81", 10.20, 13.08)], hold=0.0,
      w=[(10.24, "и"), (10.42, "меня"), (10.56, "муж"), (10.70, "спрашивает:"), (11.10, "откуда"), (11.40, "у"), (11.48, "тебя"), (11.68, "такое?"),
         (12.12, "я"), (12.28, "говорю:"), (12.50, "здравствуйте!")]),
 dict(n="before", aud=[("81", 13.62, 15.80)], hold=1.0,
      w=[(13.70, "а"), (13.74, "ты"), (14.08, "как"), (14.36, "могла"), (14.72, "это"), (14.90, "носить?")]),
 dict(n="collage", aud=[("81", 15.94, 17.72), ("81", 18.04, 20.32)], hold=0.2,
      w=[(16.02, "это"), (16.26, "вообще"), (16.70, "совсем"), (17.08, "по-другому"), (18.08, "он"), (18.66, "смотрит,"), (19.26, "очень"), (19.52, "смешно"), (20.02, "было")]),
 dict(n="surgeon", aud=[("78", 5.05, 6.34), ("78", 7.05, 8.64)], hold=0.0,
      w=[(5.12, "да,"), (5.38, "у"), (5.42, "вас"), (5.62, "очень"), (5.86, "красиво"), (7.12, "да,"), (7.50, "спасибо"), (7.86, "вам"), (8.26, "большое")]),
 dict(n="tank", aud=[("81", 4.30, 8.40), ("81", 9.02, 10.04)], hold=0.0,
      w=[(4.34, "у"), (4.42, "меня"), (4.50, "даже"), (4.70, "муж…"), (5.10, "я"), (5.16, "мерила"), (5.54, "сегодня"), (5.98, "своё"), (6.56, "бельё,"),
         (6.94, "которое"), (7.12, "без"), (9.06, "пуш-апа")]),
 dict(n="final", aud=[], hold=3.4, w=[]),
]
voice = []; t = 0.0; SUBS = []
for b in BL:
    b["t0"] = t; b["map"] = []
    for (src, a, e) in b["aud"]:
        seg = AUD[src][int(a * SR):int(e * SR)].copy()
        sp = seg[np.abs(seg) > np.percentile(np.abs(seg), 70)]
        seg *= 0.12 / max(np.sqrt(np.mean(sp ** 2)), 1e-4)            # per-segment level match
        fi, fo = int(0.015 * SR), int(0.06 * SR)
        seg[:fi] *= np.linspace(0, 1, fi); seg[-fo:] *= np.linspace(1, 0, fo) ** 1.5
        voice.append(seg); b["map"].append((src, a, e, t))
        for (wt, wd) in b["w"]:
            if a - 0.05 <= wt < e: SUBS.append((t + wt - a, wd))
        t += e - a
    if b["hold"]: voice.append(np.zeros(int(b["hold"] * SR), np.float32)); t += b["hold"]
    b["t1"] = t
DUR = t
voice = np.concatenate(voice)
print("dur", round(DUR, 2), [(b["n"], round(b["t0"], 2), round(b["t1"], 2)) for b in BL])

# ---------------- drawing helpers ----------------
def to_np(im): a = np.asarray(im).astype(np.float32) / 255; a[..., :3] *= a[..., 3:4]; return a
def blend(dst, img, x, y, al):
    if al <= 0.003: return
    h, w = img.shape[:2]; x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]; d = dst[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4] * al; d += s[..., :3] * al
def shadow_of(img, s=10, a=0.5):
    sh = cv2.GaussianBlur(img[..., 3], (0, 0), s) * a; o = np.zeros(sh.shape + (4,), np.float32); o[..., 3] = sh; return o
def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def back(p): p = min(max(p, 0), 1); c = 1.9; return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2
def pop_img(dst, img, cx, cy, t, t0, t1=1e9):
    if t < t0 or t > t1 + 0.25: return
    s = max(back((t - t0) / 0.3), 0.02); al = min(1, (t - t0) / 0.08) * (1 - min(1, max(0, (t - t1) / 0.25)))
    im = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s)))) if abs(s - 1) > 0.01 else img
    blend(dst, shadow_of(im, 10, 0.45), cx - im.shape[1] / 2, cy - im.shape[0] / 2 + 6, al)
    blend(dst, im, cx - im.shape[1] / 2, cy - im.shape[0] / 2, al)
def tag(txt, size=40, bg=BLUE, fg=WHITE):
    f = osw(size); asc, desc = f.getmetrics(); w = int(f.getlength(txt))
    im = Image.new("RGBA", (w + 32, asc + desc + 12), bg + (255,)); ImageDraw.Draw(im).text((16, 6 + asc), txt, font=f, fill=fg + (255,), anchor="ls")
    return to_np(im)
class Txt:
    def __init__(s, lines, size=56, align="C", x=80, maxw=930):
        s.items = []; s.align = align; s.x = x
        for txt, kind in lines:
            if kind in ("H", "HB"):
                sz = size
                while osw(sz).getlength(txt) > maxw: sz -= 2
                f = osw(sz)
            else: f = mont(500 if kind == "SB" else 400, 38)
            asc, desc = f.getmetrics(); s.items.append(dict(txt=txt, kind=kind, f=f, asc=asc, desc=desc, w=f.getlength(txt)))
    def draw(s, frame, y, t, t0, t1=1e9, cps=30):
        if t < t0 or t > t1 + 0.25: return
        fade = 1 - min(1, max(0, (t - t1) / 0.25)); nchar = (t - t0) * cps
        im = Image.new("RGBA", (W, 600), (0, 0, 0, 0)); d = ImageDraw.Draw(im); yy = 10; used = 0
        for it in s.items:
            lh = (it["asc"] + it["desc"]) * (0.95 if it["kind"] in ("H", "HB") else 1.2)
            n = int(min(len(it["txt"]), max(0, nchar - used))); used += len(it["txt"])
            if n <= 0: break
            txt = it["txt"][:n]; x = s.x if s.align == "L" else (W - it["w"]) / 2
            if it["kind"] == "HB":
                wv = it["f"].getlength(txt)
                d.rectangle((x - 14, yy + it["asc"] * 0.1, x + wv + 14, yy + lh + 14), fill=BLUE + (255,))
                d.text((x, yy + 7 + it["asc"]), txt, font=it["f"], fill=WHITE + (255,), anchor="ls"); yy += lh + 18
            else:
                d.text((x, yy + it["asc"]), txt, font=it["f"], fill=WHITE + (255,), anchor="ls"); yy += lh
        a = to_np(im)
        blend(frame, shadow_of(a, 10, 0.6), 0, y + 4, fade); blend(frame, a, 0, y, fade)
def partial(pts, p):
    pts = np.asarray(pts, np.float64)
    if p >= 1: return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
    L_ = cum[-1] * max(p, 0); k = np.searchsorted(cum, L_)
    if k == 0: return pts[:1]
    q = pts[k - 1] + (pts[k] - pts[k - 1]) * ((L_ - cum[k - 1]) / max(seg[k - 1], 1e-6)); return np.vstack([pts[:k], q])
def bez(a, b, bend=0.2, n=40):
    a, b = np.asarray(a, float), np.asarray(b, float); m = (a + b) / 2; d = b - a; c = m + np.array([-d[1], d[0]]) * bend
    tt = np.linspace(0, 1, n)[:, None]; return (1 - tt) ** 2 * a + 2 * (1 - tt) * tt * c + tt ** 2 * b
def stroke(frame, pts, al, th=5, head=False, dash=False, col=WHITE):
    if al <= 0 or len(pts) < 2: return
    mk = np.zeros((H, W), np.uint8)
    if dash:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
        for d0 in np.arange(0, cum[-1], 26):
            cv2.circle(mk, (int(np.interp(d0, cum, pts[:, 0])), int(np.interp(d0, cum, pts[:, 1]))), th // 2 + 1, 255, -1, cv2.LINE_AA)
    else: cv2.polylines(mk, [np.round(pts).astype(np.int32)], False, 255, th, cv2.LINE_AA)
    if head and len(pts) > 3:
        e = pts[-1]; d = pts[-1] - pts[-4]; d /= max(np.linalg.norm(d), 1e-6); n_ = np.array([-d[1], d[0]]); hl = 28
        cv2.fillPoly(mk, [np.array([e + d * 6, e - d * hl + n_ * hl * 0.55, e - d * hl - n_ * hl * 0.55]).astype(np.int32)], 255, cv2.LINE_AA)
    sh = cv2.GaussianBlur(mk, (0, 0), 4).astype(np.float32)[..., None] / 255 * 0.5 * al
    frame *= 1 - sh; a = mk.astype(np.float32)[..., None] / 255 * al; frame *= 1 - a; frame += np.float32(col) / 255 * a
def glow(frame, c, ax, al):
    if al <= 0: return
    m = np.zeros((H, W), np.float32); cv2.ellipse(m, (int(c[0]), int(c[1])), (int(ax[0]), int(ax[1])), 0, 0, 360, 1, -1, cv2.LINE_AA)
    a = cv2.GaussianBlur(m, (0, 0), max(ax) / 4)[..., None] * al; frame *= 1 - a; frame += a * 0.97
def env(t, t0, t1, dur):
    if t < t0 or t > t1 + 0.3: return 0, 0
    return ease((t - t0) / dur), 1 - min(1, max(0, (t - t1) / 0.3))
def kenburns(img, z, fx=540, fy=960):
    M = np.float32([[z, 0, fx - z * fx], [0, z, fy - z * fy]]); return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
yy_, xx_ = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.38 * (((xx_ - 540) / 760) ** 2 + ((yy_ - 960) / 1250) ** 2))[..., None].clip(0.55, 1)
def grade(fr):
    g = fr @ np.float32([0.299, 0.587, 0.114]); fr = fr * 0.84 + g[..., None] * 0.16
    return np.clip((fr - 0.5) * 1.07 + 0.475, 0, 1) * VIG
LOGO = np.asarray(Image.open("logo_clean.png").resize((320, int(115 * 320 / 620)), Image.LANCZOS)).astype(np.float32) / 255
LOGO[..., :3] *= LOGO[..., 3:4]
SUBC = {}
def subtitles(frame, t):
    for i, (t0, w) in enumerate(SUBS):
        t1 = min(SUBS[i + 1][0] if i + 1 < len(SUBS) else DUR, t0 + 1.0)
        if t0 <= t < t1:
            if w not in SUBC:
                f = mont(500, 44); asc, desc = f.getmetrics(); tw = int(f.getlength(w))
                im = Image.new("RGBA", (tw + 20, asc + desc + 10), (0, 0, 0, 0)); ImageDraw.Draw(im).text((10, 5 + asc), w, font=f, fill=WHITE + (255,), anchor="ls")
                a = to_np(im); SUBC[w] = (a, shadow_of(a, 6, 0.75))
            im, sh = SUBC[w]; al = min(1, (t - t0) / 0.06)
            blend(frame, sh, 540 - im.shape[1] / 2, 1520 - im.shape[0] / 2 + 3, al); blend(frame, im, 540 - im.shape[1] / 2, 1520 - im.shape[0] / 2, al)

# ---------------- sources ----------------
def load_png(n): return cv2.cvtColor(cv2.imread(n), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
PH = {k: load_png(f"r_{k}.png") for k in ["434d627b", "4f55eeac", "e4a87217", "3681468f", "af0d1e43", "34b6868e"]}
POSE = {"79": np.load("pose79.npy"), "80": np.load("pose80.npy"), "80o": np.load("pose80.npy"), "78": np.load("pose78.npy")}
VSRC = {"80o": "dl/v80.mov", "79": "b79.mp4", "80": "b80.mp4", "78": "b78.mp4"}
def read_frames(src, a, e):
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(a), "-i", src, "-t", str(e - a + 0.1), "-vf", "fps=30,scale=1080:1920:flags=bicubic",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)
def blur_face(img, P):
    nose, m1, m2, e1, e2 = P[0], P[9], P[10], P[7], P[8]
    c = (nose + m1 + m2) / 3; r = max(np.linalg.norm(e1 - e2) * 0.75, np.linalg.norm(P[11] - P[12]) * 0.28)
    m = np.zeros((H, W), np.float32); cv2.ellipse(m, (int(c[0]), int(c[1] - r * 0.25)), (int(r), int(r * 1.35)), 0, 0, 360, 1, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), r / 6)[..., None]
    sm = cv2.resize(img, (W // 40, H // 40), interpolation=cv2.INTER_AREA)
    bl = cv2.GaussianBlur(cv2.resize(sm, (W, H)), (0, 0), 25)
    return img * (1 - m) + bl * m
CLIPS = {}
def clip(key, src, a, e):
    if key not in CLIPS: CLIPS[key] = (read_frames(VSRC[src], a, e), src, a)
    return CLIPS[key]
def vframe(key, src, a, e, lt):
    fr, s, a0 = clip(key, src, a, e); k = min(int(lt * FPS), len(fr) - 1)
    img = fr[k].astype(np.float32) / 255
    P = POSE[s][min(int(round((a0 + lt) * FPS)), len(POSE[s]) - 1)]
    return blur_face(img, P)

# ---------------- texts ----------------
T_HOOK = Txt([("«ОТКУДА У ТЕБЯ ТАКОЕ?»", "HB"), ("— спросил муж после операции", "S")], size=64)
T_BEFORE = Txt([("ДО ОПЕРАЦИИ", "H")], size=58)
L_POLE = tag("ПУСТОЙ ВЕРХНИЙ ПОЛЮС", 34); L_VOL = tag("МАЛО ОБЪЁМА", 34)
TAG_B = tag("ДО", 34, (40, 40, 40)); TAG_A = tag("ПОСЛЕ", 34)
L_FULL = tag("НАПОЛНЕН", 30); L_EMPTY = tag("ПУСТОЙ", 30, (40, 40, 40))
T_DIFF = Txt([("«ЭТО ВООБЩЕ СОВСЕМ", "H"), ("ПО-ДРУГОМУ»", "HB")], size=56)
T_SURG = Txt([("«У ВАС ОЧЕНЬ КРАСИВО»", "HB"), ("— хирург на осмотре", "S")], size=58)
T_TANK = Txt([("И ДАЖЕ В ОДЕЖДЕ", "H"), ("БЕЗ ПУШ-АПА", "HB")], size=60)
T_FIN = Txt([("РЕЗУЛЬТАТ, КОТОРЫЙ", "H"), ("ЗАМЕТИЛ ДАЖЕ МУЖ", "HB")], size=62)
L_SHAPE = tag("ОКРУГЛАЯ НАТУРАЛЬНАЯ ФОРМА", 32)

def render(b, lt):
    n = b["n"]; d = b["t1"] - b["t0"]; p = lt / d
    if n == "hook":
        fr = grade(kenburns(vframe("h", "79", 0.0, 3.2, lt), 1.0 + 0.06 * p, 540, 800))
        T_HOOK.draw(fr, 1180, lt, 0.1, cps=34)
        return fr
    if n == "before":
        k = "3681468f" if lt < 1.7 else "af0d1e43"
        fr = grade(kenburns(PH[k], 1.0 + 0.05 * p, 540, 760))
        blend(fr, TAG_B, 80, 300, 1)
        if k == "3681468f":
            z = 1.0 + 0.05 * p
            def M(x, y): return np.array([540 + (x - 540) * z, 760 + (y - 760) * z])
            q, al = env(lt, 0.2, 1.62, 0.4)
            for (sx, sy) in [(335, 700), (750, 686)]:
                arc = np.array([M(sx + 95 * np.cos(th), sy - 40 - 75 * np.sin(th)) for th in np.linspace(np.pi, 0, 40)])
                stroke(fr, partial(arc, q), al, 6, dash=True)
            q2, al2 = env(lt, 0.45, 1.62, 0.3)
            for (sx, sy) in [(335, 700), (750, 686)]:
                stroke(fr, partial(bez((540 + (-70 if sx < 540 else 70), 455), M(sx, sy - 125), 0.12 if sx < 540 else -0.12), q2), al2, 4, head=q2 > 0.95)
            pop_img(fr, L_POLE, 540, 430, lt, 0.2, 1.6)
            q, al = env(lt, 0.8, 1.62, 0.35)
            for (sx, sy) in [(335, 805), (750, 795)]:
                stroke(fr, partial(bez((540 + (-60 if sx < 540 else 60), 990), M(sx, sy), -0.12 if sx < 540 else 0.12), q), al, 4, head=q > 0.95)
            pop_img(fr, L_VOL, 540, 1015, lt, 0.75, 1.6)
        else:
            T_BEFORE.draw(fr, 1240, lt, 1.75, cps=30)
        return fr
    if n == "collage":
        seq = [("434d627b", 0, 1.9), ("4f55eeac", 1.9, 3.1), ("e4a87217", 3.1, 99)]
        for (k, a, e) in seq:
            if a <= lt < e:
                pp = (lt - a) / max(0.1, min(e, d) - a)
                fr = grade(kenburns(PH[k], 1.0 + 0.04 * pp, 540, 960))
                blend(fr, TAG_A, W - 80 - TAG_A.shape[1], 300, 1); blend(fr, TAG_B, W - 80 - TAG_B.shape[1], 1830 - TAG_B.shape[0], 1)
                if k == "434d627b":
                    z = 1.0 + 0.04 * pp
                    def M(x, y): return np.array([540 + (x - 540) * z, 960 + (y - 960) * z])
                    q, al = env(lt, 0.15, 1.8, 0.45)
                    for (sx, sy) in [(405, 265), (673, 258)]:
                        arc = np.array([M(sx + 85 * np.cos(th), sy - 30 - 70 * np.sin(th)) for th in np.linspace(np.pi, 0, 40)])
                        stroke(fr, partial(arc, q), al, 6)
                    for (sx, sy) in [(407, 1252), (671, 1243)]:
                        arc = np.array([M(sx + 85 * np.cos(th), sy - 30 - 55 * np.sin(th)) for th in np.linspace(np.pi, 0, 40)])
                        stroke(fr, partial(arc, q), al, 6, dash=True)
                    q2, al2 = env(lt, 0.45, 1.8, 0.3)
                    stroke(fr, partial(bez((175, 255), M(318, 232), 0.15), q2), al2, 4, head=q2 > 0.95)
                    stroke(fr, partial(bez((175, 1245), M(320, 1222), 0.15), q2), al2, 4, head=q2 > 0.95)
                    pop_img(fr, L_FULL, 150, 215, lt, 0.4, 1.8); pop_img(fr, L_EMPTY, 150, 1205, lt, 0.5, 1.8)
        if lt >= 1.95: T_DIFF.draw(fr, 870, lt, 1.95, cps=32)
        return fr
    if n == "surgeon":
        (s1, a1, e1, t1), (s2, a2, e2, t2) = b["map"]
        if lt < e1 - a1: fr = vframe("s1", "78", a1, e1, lt)
        else: fr = kenburns(vframe("s2", "78", a2, e2, lt - (e1 - a1)), 1.08, 540, 900)
        fr = grade(fr)
        T_SURG.draw(fr, 1230, lt, 0.5, cps=30)
        return fr
    if n == "tank":
        cln = vframe("t", "80", 1.4, 7.0, lt); org = vframe("to", "80o", 1.4, 7.0, lt)
        u8 = (org * 255).astype(np.uint8); hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
        shirt = ((hsv[..., 2] > 150) & (hsv[..., 1] < 45)).astype(np.uint8)
        shirt = cv2.morphologyEx(shirt, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
        shirt = cv2.morphologyEx(shirt, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        n_, lab, st, _ = cv2.connectedComponentsWithStats(shirt, 8)
        if n_ > 1: shirt = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        mk = cv2.GaussianBlur(cv2.dilate(shirt, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 3)[..., None]
        fr = grade(kenburns(cln * (1 - mk) + org * mk, 1.0 + 0.05 * p, 540, 900))
        T_TANK.draw(fr, 1210, lt, 3.9, cps=30)
        return fr
    if n == "final":
        fr = grade(kenburns(vframe("f", "79", 3.6, 7.2, lt), 1.0 + 0.06 * p, 540, 800))
        q, al = env(lt, 0.3, 99, 0.5)
        pop_img(fr, L_SHAPE, 540, 330, lt, 0.4)
        T_FIN.draw(fr, 1180, lt, 1.0, cps=30)
        return fr

import os
if not os.environ.get('SKIPV'):
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "video_only.mp4"], stdin=subprocess.PIPE)
    NF = int(round(DUR * FPS)); bi = 0
    for f in range(NF):
        t = f / FPS
        while bi + 1 < len(BL) and t >= BL[bi + 1]["t0"]: bi += 1
        b = BL[bi]; lt = t - b["t0"]
        fr = render(b, lt)
        if bi > 0 and lt < 0.25:   # punch-in + white flash transition
            e = ease(lt / 0.25); fr = kenburns(fr, 1 + 0.08 * (1 - e), 540, 960)
            fr = fr * (1 - 0.35 * (1 - e)) + 0.35 * (1 - e)
        la = 0.9
        if b["n"] == "collage": la = 0.9 * max(0, 1 - lt / 0.2) if lt < 1.9 else 0.9 * min(1, (lt - 1.9) / 0.3)
        blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, la)
        subtitles(fr, t)
        enc.stdin.write((np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()

# ---------------- music bed + SFX ----------------
rng = np.random.default_rng(11); N = int(DUR * SR) + SR
def note(f): return 440 * 2 ** ((f - 69) / 12)
music = np.zeros(N)
chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]   # Am F C G
cl = 2.4; tt = np.arange(int(cl * SR)) / SR
envp = np.minimum(1, tt / 0.5) * np.minimum(1, (cl - tt) / 0.6)
for i in range(int(DUR / cl) + 2):
    ch = chords[i % 4]; s = int(i * cl * SR); seg = np.zeros(len(tt))
    if s >= N: continue
    for m in ch + [ch[0] - 12]:
        fq = note(m); seg += np.sin(2 * np.pi * fq * tt) * 0.5 + np.sin(2 * np.pi * fq * 2 * tt) * 0.08
    seg *= envp; e_ = min(N, s + len(seg)); music[s:e_] += seg[:e_ - s]
    for j in range(8):   # soft pluck arpeggio
        m = (ch + [ch[0] + 12])[j % 4] + 12; ps = s + int(j * cl / 8 * SR)
        if ps >= N: continue
        pl = np.arange(int(0.35 * SR)) / SR
        pk = np.sin(2 * np.pi * note(m) * pl) * np.exp(-pl * 9) * 0.35; e_ = min(N, ps + len(pk)); music[ps:e_] += pk[:e_ - ps]
music = music[:int(DUR * SR)]; music /= np.abs(music).max()
vabs = np.abs(np.concatenate([voice, np.zeros(max(0, len(music) - len(voice)))]))[:len(music)]
duck = cv2.GaussianBlur((vabs > 0.02).astype(np.float32).reshape(1, -1), (0, 0), sigmaX=SR * 0.15).ravel()
music *= 0.13 * (1 - 0.6 * np.clip(duck * 3, 0, 1))
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
def thump():
    n = int(0.35 * SR); x = np.arange(n) / SR; y = np.sin(2 * np.pi * np.cumsum(55 + 90 * np.exp(-x * 25)) / SR) * np.exp(-x * 9); return y / np.abs(y).max()
def ding():
    n = int(1.4 * SR); x = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * fq * x) * np.exp(-x * k) for fq, a, k in [(1318.5, 1, 3), (1975.5, .5, 4), (2637, .3, 6), (659.25, .4, 2.5)])
    return y * np.minimum(1, x / 0.004) / np.abs(y).max()
CLK = (lambda n: (lambda x: x / np.abs(x).max())(np.diff(rng.standard_normal(n), prepend=0) * np.exp(-np.arange(n) / SR * 500)))(int(0.012 * SR))
sfx = np.zeros(len(music) + SR)
def put(s, t_, g): i = int(max(0, t_) * SR); seg = sfx[i:i + len(s)]; seg += s[:len(seg)] * g
WB, PP, TH, DG = whoosh(0.55), pop(), thump(), ding()
def at(nm, lt): return next(b for b in BL if b["n"] == nm)["t0"] + lt
for b in BL[1:]: put(WB, b["t0"] - 0.3, 0.22)
put(TH, 0.1, 0.4)
for nm, lt, nch, cps in [("hook", 0.1, 50, 34), ("before", 1.75, 11, 30), ("collage", 1.95, 28, 32), ("surgeon", 0.5, 38, 30), ("tank", 3.9, 26, 30), ("final", 1.0, 34, 30)]:
    for k in range(nch): put(CLK, at(nm, lt) + k / cps, 0.045 * rng.uniform(0.6, 1))
for nm, lt in [("before", 0.2), ("before", 0.75), ("collage", 0.1), ("final", 0.4)]: put(PP, at(nm, lt), 0.22)
put(DG, at("final", 1.0), 0.13)
sfx = sfx[:len(music)]
v = np.concatenate([voice, np.zeros(max(0, len(music) - len(voice)))])[:len(music)]
mix = np.clip(v * 1.0 + music + sfx * 0.9, -1, 1)
with wave.open("mix.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
