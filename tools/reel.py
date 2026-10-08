import subprocess, json, wave, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

W, H, FPS, SR = 1080, 1920, 30, 48000
FONTS = "../fonts2/M%d.ttf"
POWDER = (228, 210, 210); NAVY = (16, 46, 70); MAUVE = (103, 96, 104); WHITE = (255, 255, 255)

# ---------------- blocks: audio segments (src, a, b) / holds ----------------
B = [
 dict(name="after1", aud=[("surgeon", 15.35, 21.10)]),
 dict(name="before", aud=[("interview", 43.35, 45.65)], hold=1.3),
 dict(name="after_photo", aud=[("surgeon", 21.15, 22.45), ("surgeon", 24.05, 29.05)]),
 dict(name="edema", aud=[("surgeon", 50.20, 55.45)], hold=0.3),
 dict(name="months", aud=[("surgeon", 29.15, 35.95), ("surgeon", 42.75, 46.55)]),
 dict(name="feel", aud=[("interview", 0.0, 2.5), ("interview", 4.35, 7.85)]),
 dict(name="final", aud=[("interview", 45.70, 50.30)], hold=1.6),
]
AUD = {}
for n in ("surgeon", "interview"):
    with wave.open(f"a_{n}.wav") as w: AUD[n] = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
WJ = {n: json.load(open(f"{n}.json")) for n in ("surgeon", "interview")}
FIX = {"объемный": "объёмный", "нее": "неё", "отечная": "отёчная", "объем": "объём", "еще": "ещё", "все": "всё", "Ане": "Анне", "Романе": "Романовне", "Чем": "о чём"}
REPL_42 = ["мы", "нашей", "сегодняшней", "пациентке", "показали", "как", "раз", "ту,", "чтобы", "она", "не", "волновалась"]

voice = []; t = 0.0; SUBS = []
for b in B:
    b["t0"] = t
    for (src, a, e) in b["aud"]:
        seg = AUD[src][int(a * SR):int(e * SR)].copy(); fd = int(0.012 * SR)
        seg[:fd] *= np.linspace(0, 1, fd); seg[-fd:] *= np.linspace(1, 0, fd)
        voice.append(seg)
        ws = [w for w in WJ[src] if w[0] >= a - 0.05 and w[0] < e - 0.05]
        if src == "surgeon" and abs(a - 42.75) < 0.01:
            ws = [[w[0], w[1], REPL_42[i]] for i, w in enumerate(ws)]
        for w in ws:
            txt = w[2].strip(",.?!:")
            txt = FIX.get(txt, txt)
            SUBS.append((t + max(0, w[0] - a), txt))
        t += e - a
    if b.get("hold"):
        voice.append(np.zeros(int(b["hold"] * SR), np.float32)); t += b["hold"]
    b["t1"] = t
DUR = t
voice = np.concatenate(voice)
with wave.open("voice_cat.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(voice, -1, 1) * 32767).astype(np.int16).tobytes())
SUBS.sort()
def bt(name): return next(b for b in B if b["name"] == name)
print("dur", DUR, [(b["name"], round(b["t0"], 2), round(b["t1"], 2)) for b in B])

# ---------------- drawing helpers ----------------
def font(w, s): return ImageFont.truetype(FONTS % w, s)
def premul(a): a = a.copy(); a[..., :3] *= a[..., 3:4]; return a
def to_np(im): return premul(np.asarray(im).astype(np.float32) / 255)
def pill(txt, size=44, bg=NAVY, fg=POWDER, weight=600):
    f = font(weight, size); asc, desc = f.getmetrics(); tw = int(f.getlength(txt))
    ph, pw = asc + desc + 18, tw + 52
    im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, pw - 1, ph - 1), radius=ph // 2, fill=bg + (255,))
    d.text((pw // 2, ph // 2 + 2), txt, font=f, fill=fg + (255,), anchor="mm")
    return to_np(im)
def headline_lines(lines, size=86, accent=()):
    """list of per-line images: navy bar + white/powder Black text"""
    out = []
    while max(font(900, size).getlength(ln) for ln in lines) > 930: size -= 2
    for ln in lines:
        f = font(900, size); asc, desc = f.getmetrics()
        words = ln.split(" "); sp = f.getlength(" ")
        tw = int(sum(f.getlength(x) for x in words) + sp * (len(words) - 1))
        ph, pw = asc + desc + 14, tw + 48
        im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, pw - 1, ph - 1), radius=18, fill=NAVY + (255,))
        x = 24
        for wd in words:
            d.text((x, 7 + asc), wd, font=f, fill=(POWDER if wd.strip("?«»,") in accent else WHITE) + (255,), anchor="ls")
            x += f.getlength(wd) + sp
        out.append(to_np(im))
    return out
def hl_h(imgs, gap=10): return sum(i.shape[0] + gap for i in imgs)
def blend(dst, img, x, y, al):
    if al <= 0.003: return
    h, w = img.shape[:2]; x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]; d = dst[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4] * al; d += s[..., :3] * al
def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def back(p): p = min(max(p, 0), 1); c = 1.9; return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2
def shadow_of(img, s=10, a=0.45):
    sh = cv2.GaussianBlur(img[..., 3], (0, 0), s) * a; o = np.zeros(sh.shape + (4,), np.float32); o[..., 3] = sh; return o
def pop_img(dst, img, cx, cy, t, t0, t1=1e9, sh=True):
    if t < t0 or t > t1 + 0.25: return
    s = max(back((t - t0) / 0.3), 0.02); al = min(1, (t - t0) / 0.08) * (1 - min(1, max(0, (t - t1) / 0.25)))
    im = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s)))) if abs(s - 1) > 0.01 else img
    if sh: blend(dst, shadow_of(im), cx - im.shape[1] / 2, cy - im.shape[0] / 2 + 8, al)
    blend(dst, im, cx - im.shape[1] / 2, cy - im.shape[0] / 2, al)
def slide_lines(dst, imgs, cx, y, t, t0, t1=1e9, gap=10, stagger=0.12):
    for i, im in enumerate(imgs):
        a = t - t0 - i * stagger
        if a < 0 or t > t1 + 0.25: y += im.shape[0] + gap; continue
        e = ease(a / 0.3); al = min(1, a / 0.15) * (1 - min(1, max(0, (t - t1) / 0.25)))
        k = int(30 * (1 - e)) | 1
        im2 = cv2.blur(im, (1, k)) if k > 2 else im
        blend(dst, shadow_of(im2, 12, 0.5), cx - im.shape[1] / 2, y + 50 * (1 - e) + 8, al)
        blend(dst, im2, cx - im.shape[1] / 2, y + 50 * (1 - e), al)
        y += im.shape[0] + gap

# vector strokes (arrows / arcs / glows)
def partial(pts, p):
    pts = np.asarray(pts, np.float64)
    if p >= 1: return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
    L_ = cum[-1] * max(p, 0); k = np.searchsorted(cum, L_)
    if k == 0: return pts[:1]
    q = pts[k - 1] + (pts[k] - pts[k - 1]) * ((L_ - cum[k - 1]) / max(seg[k - 1], 1e-6))
    return np.vstack([pts[:k], q])
def bez(a, b, bend=0.25, n=40):
    a, b = np.asarray(a, float), np.asarray(b, float); m = (a + b) / 2; d = b - a
    c = m + np.array([-d[1], d[0]]) * bend; t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b
def stroke(frame, pts, al, col=WHITE, th=7, head=False, dash=False):
    if al <= 0 or len(pts) < 2: return
    mk = np.zeros((H, W), np.uint8)
    if dash:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
        for d0 in np.arange(0, cum[-1], 28):
            cv2.circle(mk, (int(np.interp(d0, cum, pts[:, 0])), int(np.interp(d0, cum, pts[:, 1]))), th // 2 + 1, 255, -1, cv2.LINE_AA)
    else:
        cv2.polylines(mk, [np.round(pts).astype(np.int32)], False, 255, th, cv2.LINE_AA)
    if head and len(pts) > 3:
        e = pts[-1]; d = pts[-1] - pts[-4]; d /= max(np.linalg.norm(d), 1e-6); n_ = np.array([-d[1], d[0]]); hl = 32
        cv2.fillPoly(mk, [np.array([e + d * 6, e - d * hl + n_ * hl * 0.55, e - d * hl - n_ * hl * 0.55]).astype(np.int32)], 255, cv2.LINE_AA)
    sm = cv2.resize(mk, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    sh = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 5), (W, H)).astype(np.float32)[..., None] / 255 * 0.5 * al
    frame *= 1 - sh
    a = mk.astype(np.float32)[..., None] / 255 * al
    frame *= 1 - a; frame += np.float32(col) / 255 * a
def glow(frame, c, ax, ang, al, col=WHITE):
    if al <= 0: return
    m = np.zeros((H // 2, W // 2), np.uint8)
    cv2.ellipse(m, (int(c[0] / 2), int(c[1] / 2)), (int(ax[0] / 2), int(ax[1] / 2)), ang, 0, 360, 255, -1, cv2.LINE_AA)
    a = cv2.resize(cv2.GaussianBlur(m, (0, 0), max(ax) / 5).astype(np.float32) / 255, (W, H))[..., None] * al
    frame *= 1 - a; frame += np.float32(col) / 255 * a
def arc_pts(c, rx, ry, a0, a1, n=50):
    t = np.radians(np.linspace(a0, a1, n)); return np.stack([c[0] + rx * np.cos(t), c[1] + ry * np.sin(t)], 1)
def env(t, t0, t1, dur):
    if t < t0 or t > t1 + 0.3: return 0, 0
    return ease((t - t0) / dur), 1 - min(1, max(0, (t - t1) / 0.3))

# ---------------- sources ----------------
def photo(n):
    im = ImageOps.exif_transpose(Image.open("dl/" + n)).convert("RGB"); w, h = im.size; s = max(W / w, H / h)
    im = im.resize((round(w * s), round(h * s)), Image.LANCZOS); x = (im.width - W) // 2; y = (im.height - H) // 2
    return np.asarray(im.crop((x, y, x + W, y + H))).astype(np.float32) / 255
def kenburns(img, p, z0=1.0, z1=1.08, fx=540, fy=800):
    z = z0 + (z1 - z0) * p
    M = np.float32([[z, 0, fx - z * fx], [0, z, fy - z * fy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
PH = {n: photo(n) for n in ["before_a", "before_b", "d6_b", "m1_b", "m2_a", "m2_b", "m2_c"]}

class Vid:
    def __init__(s, segs): s.segs = list(segs); s.p = None; s.last = None
    def _open(s):
        src, a, e = s.segs.pop(0)
        s.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(a), "-to", str(e), "-i", "dl/" + src, "-vf", "fps=30,scale=1080:1920",
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        s.cut = True
    def next(s):
        s.cut = False
        while True:
            if s.p is None:
                if not s.segs: return s.last
                s._open()
            b = s.p.stdout.read(W * H * 3)
            if len(b) == W * H * 3:
                s.last = np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32) / 255; return s.last
            s.p.kill(); s.p = None
VID = {"after1": [("surgeon", 15.35, 21.10)], "edema": [("surgeon", 50.20, 55.45)], "feel": [("interview", 0.0, 2.5), ("interview", 4.35, 7.85)]}

POSE1 = np.load("pose_b1.npy")
def b1_poles(f):
    f = min(f, len(POSE1) - 1); a = POSE1[max(0, f - 2):f + 3].mean(0); l, r = a[11], a[12]; c = (l + r) / 2; w = np.linalg.norm(l - r)
    return [s + (c - s) * 0.5 + np.array([0, 0.2 * w]) for s in (r, l)], w
KF4 = np.array([[0.2, 120, 860, 270, 850], [2.5, 175, 860, 320, 850], [4.8, 140, 875, 300, 860]], float)
CR4 = (0.0, 255.0, 635.0, 1129.0)
def b4_poles(t):
    p = [np.interp(t, KF4[:, 0], KF4[:, i]) for i in range(1, 5)]
    x0, y0, cw, ch = CR4
    return [np.array([(p[0] - x0) * W / cw, (p[1] - y0) * H / ch]), np.array([(p[2] - x0) * W / cw, (p[3] - y0) * H / ch])]

# ---------------- texts ----------------
HL1 = headline_lines(["А ВДРУГ ГРУДЬ", "ПОЛУЧИЛАСЬ", "СЛИШКОМ БОЛЬШАЯ?"], 84, accent=("БОЛЬШАЯ",))
CAP_6D = pill("6 дней после операции", 40, MAUVE, WHITE)
LBL_TOP = pill("объёмный верх", 46)
HL2 = headline_lines(["ВОТ КАКОЙ", "БЫЛ ЗАПРОС"], 80)
REQ = [pill(s, 50) for s in ["круглая", "наполненная", "но естественная"]]
TAG_BEFORE = pill("ДО", 46, MAUVE, WHITE, 900); TAG_AFTER = pill("ПОСЛЕ", 46, NAVY, POWDER, 900)
HL3 = headline_lines(["СЕЙЧАС ЭТО ЕЩЁ НЕ", "ФИНАЛЬНАЯ ФОРМА"], 80, accent=("ФИНАЛЬНАЯ", "ФОРМА"))
LBL_SLOPE = pill("верхний склон будет ровным", 44)
HL4a = headline_lines(["ОТЁК"], 150, accent=("ОТЁК",))
HL4b = headline_lines(["ТКАНИ ЕЩЁ НЕ", "АДАПТИРОВАЛИСЬ"], 72)
CAP4 = pill("поэтому верх груди сейчас выглядит объёмнее", 36, MAUVE, WHITE)
HL5 = headline_lines(["ЧЕРЕЗ НЕСКОЛЬКО МЕСЯЦЕВ", "ФОРМА БУДЕТ ВЫГЛЯДЕТЬ", "СОВСЕМ ИНАЧЕ"], 64, accent=("ИНАЧЕ",))
CAP_M2 = pill("2 месяца после операции", 40, MAUVE, WHITE); CAP_M1 = pill("месяц после операции", 40, MAUVE, WHITE)
HL6 = headline_lines(["А САМОЧУВСТВИЕ?"], 90)
em = Image.open("emoji.png").crop((19, 17, 136, 129)).resize((84, 80), Image.LANCZOS)
f_ = font(900, 64); qt = "«Как на курорте»"; tw = int(f_.getlength(qt)); asc, desc = f_.getmetrics()
qi = Image.new("RGBA", (tw + 48 + 100, asc + desc + 30), (0, 0, 0, 0)); d_ = ImageDraw.Draw(qi)
d_.rounded_rectangle((0, 0, qi.width - 1, qi.height - 1), radius=qi.height // 2, fill=POWDER + (255,))
d_.text((24, 15 + asc), qt, font=f_, fill=NAVY + (255,), anchor="ls"); qi.alpha_composite(em, (tw + 40, (qi.height - 80) // 2))
QUOTE = to_np(qi)
HL7 = headline_lines(["«ПОЧЕМУ Я НЕ СДЕЛАЛА", "ЭТО РАНЬШЕ?»"], 74, accent=("РАНЬШЕ?»",))
SUBC = {}
def sub_img(w):
    if w not in SUBC: SUBC[w] = pill(w.lower(), 48, NAVY, WHITE, 500)
    return SUBC[w]

def subtitles(frame, t):
    for i, (t0, w) in enumerate(SUBS):
        t1 = SUBS[i + 1][0] if i + 1 < len(SUBS) else DUR
        t1 = min(t1, t0 + 1.0)
        if t0 <= t < t1:
            im = sub_img(w); s = 0.85 + 0.15 * ease((t - t0) / 0.12)
            im2 = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
            blend(frame, im2, 540 - im2.shape[1] / 2, 1690 - im2.shape[0] / 2, min(1, (t - t0) / 0.05))

# ---------------- per-block rendering ----------------
def render_block(b, lt, fi, vid):
    n = b["name"]; d = b["t1"] - b["t0"]; p = lt / d
    if n == "after1":
        fr = vid.next().copy()
        slide_lines(fr, HL1, 540, 190, lt, 0.15)
        cy = 190 + hl_h(HL1) + 40
        pop_img(fr, CAP_6D, 540, cy, lt, 0.7, 1.9)
        pts, w = b1_poles(fi)
        q, al = env(lt, 2.05, 99, 0.45)
        for i, c in enumerate(pts):
            stroke(fr, partial(arc_pts(c + [0, 0.05 * w], 0.2 * w, 0.12 * w, 200, 340), q), al, POWDER, 9)
        q, al = env(lt, 2.35, 99, 0.4)
        if q > 0:
            for i, c in enumerate(pts):
                st = (430 if i == 0 else 650, cy + 30); stroke(fr, partial(bez(st, c + [0, -0.1 * w], -0.2 if i == 0 else 0.2), q), al, WHITE, 6, head=q > 0.95)
        pop_img(fr, LBL_TOP, 540, cy, lt, 2.2)
        return fr
    if n == "before":
        img = PH["before_b"] if lt < 1.9 else PH["before_a"]
        fr = kenburns(img, p, 1.0, 1.1, 540, 650)
        slide_lines(fr, HL2, 540, 200, lt, 0.05)
        pop_img(fr, TAG_BEFORE, 130, 1150, lt, 0.0)
        for i, im in enumerate(REQ):
            pop_img(fr, im, 540, 1200 + i * 95, lt, 0.55 + i * 0.55)
        return fr
    if n == "after_photo":
        fr = kenburns(PH["d6_b"], p, 1.04, 1.12, 540, 600)
        z = 1.04 + 0.08 * p
        def mp_(x, y): return np.array([540 + (x - 540) * z, 600 + (y - 600) * z])
        slide_lines(fr, HL3, 540, 1150, lt, 0.2)
        pop_img(fr, TAG_AFTER, 950, 120, lt, 0.0); pop_img(fr, CAP_6D, 540, 1150 + hl_h(HL3) + 45, lt, 0.9)
        q, al = env(lt, 4.9, 99, 0.6)
        for (x0, y0, x1, y1, x2, y2) in [(120, 440, 238, 372, 380, 420), (600, 425, 745, 366, 880, 425)]:
            pts = bez(mp_(x0, y0), mp_(x2, y2), 0.0, 3); pts = np.array([mp_(x0, y0), mp_(x1, y1), mp_(x2, y2)])
            tt = np.linspace(0, 1, 40)[:, None]; cur = (1 - tt) ** 2 * pts[0] + 2 * (1 - tt) * tt * (2 * pts[1] - (pts[0] + pts[2]) / 2) + tt ** 2 * pts[2]
            stroke(fr, partial(cur, q), al, WHITE, 7, dash=True)
        pop_img(fr, LBL_SLOPE, 540, 250, lt, 4.95)
        return fr
    if n == "edema":
        src = vid.next(); x0, y0, cw, ch = CR4
        fr = cv2.resize(src[int(y0):int(y0 + ch), int(x0):int(x0 + cw)], (W, H), interpolation=cv2.INTER_CUBIC)
        zin = 1 + 0.15 * (1 - ease(lt / 0.5)); fr = kenburns(fr, 0, zin, zin, 540, 1000) if zin > 1.001 else fr
        pls = b4_poles(lt)
        q, al = env(lt, 0.3, 99, 0.4)
        if q > 0:
            pulse = 0.7 + 0.3 * np.sin((lt - 0.3) * 5)
            for i, c in enumerate(pls): glow(fr, c + [0, 10], (150 * q, 85 * q), -10 if i == 0 else 10, 0.42 * al * pulse, POWDER)
            for i, c in enumerate(pls): stroke(fr, partial(arc_pts(c + [0, 25], 150, 85, 190, 350), q), al, WHITE, 6, dash=True)
        pop_img(fr, HL4a[0], 540, 270, lt, 0.3)
        slide_lines(fr, HL4b, 540, 1180, lt, 1.5)
        pop_img(fr, CAP4, 540, 1180 + hl_h(HL4b) + 45, lt, 2.9)
        return fr
    if n == "months":
        seq = [("m2_c", 0, 3.6, CAP_M2), ("m2_b", 3.6, 7.1, CAP_M2), ("m1_b", 7.1, 99, CAP_M1)]
        for (k, a, e, cap) in seq:
            if a <= lt < e:
                pp = (lt - a) / max(0.1, min(e, d) - a)
                fr = kenburns(PH[k], pp, 1.0, 1.06, 540, 960)
                if lt - a < 0.25 and a > 0: fr = kenburns(fr, 0, 1 + 0.12 * (1 - ease((lt - a) / 0.25)), 1, 540, 960)
                if k.startswith("m2"):
                    pop_img(fr, TAG_AFTER, 950, 870, lt, a + 0.1); pop_img(fr, TAG_BEFORE, 950, 1830, lt, a + 0.25)
                pop_img(fr, cap, 540, 1580 if k == "m1_b" else 960, lt, a + 0.3)
        slide_lines(fr, HL5, 540, 120, lt, 0.2)
        return fr
    if n == "feel":
        fr = vid.next().copy()
        cut = 2.5
        if lt >= cut and lt - cut < 3.5:
            fr = kenburns(fr, 0, 1.12, 1.12, 540, 700)
        slide_lines(fr, HL6, 540, 230, lt, 0.05)
        pop_img(fr, QUOTE, 540, 420, lt, 1.35)
        return fr
    if n == "final":
        L = PH["before_b"][:, 561 - 270:561 + 270]; R = PH["d6_b"][:, 523 - 270:523 + 270]
        fr = np.zeros((H, W, 3), np.float32); fr[:, :540] = L
        wp = ease(lt / 0.6); xr = int(540 + 540 * (1 - wp))
        if xr < W: fr[:, xr:] = R[:, :W - xr]
        fr = kenburns(fr, p, 1.0, 1.05, 540, 700)
        cv2.line(fr, (int(540 + (xr - 540) * 1.0), 0), (int(xr), H), (1, 1, 1), 4, cv2.LINE_AA)
        pop_img(fr, TAG_BEFORE, 270, 1080, lt, 0.4); pop_img(fr, TAG_AFTER, 810, 1080, lt, 0.6)
        slide_lines(fr, HL7, 540, 1250, lt, 3.3)
        return fr

dec_rng = [(b, Vid(VID[b["name"]]) if b["name"] in VID else None) for b in B]
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "video_only.mp4"], stdin=subprocess.PIPE)
NF = int(round(DUR * FPS)); bi = 0; bstart = 0
for f in range(NF):
    t = f / FPS
    while bi + 1 < len(B) and t >= B[bi + 1]["t0"]: bi += 1; bstart = f
    b, vid = dec_rng[bi]; lt = t - b["t0"]
    fr = render_block(b, lt, f - bstart, vid)
    if bi > 0 and lt < 0.28:          # punch-in transition
        e = ease(lt / 0.28); z = 1 + 0.1 * (1 - e)
        fr = kenburns(fr, 0, z, z, 540, 960)
        k = int(40 * (1 - e)) | 1
        if k > 2: fr = cv2.blur(fr, (1, k))
    subtitles(fr, t)
    enc.stdin.write((np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
enc.stdin.close(); enc.wait()

# ---------------- SFX ----------------
rng = np.random.default_rng(5)
def lp(x, cut):
    y = np.zeros_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * cut / SR)
    for i in range(len(x)): s += a[i] * (x[i] - s); y[i] = s
    return y
def whoosh(d=0.5, lo=300, hi=5000, pk=0.6):
    n = int(d * SR); tt = np.linspace(0, 1, n)
    e = np.where(tt < pk, (tt / pk) ** 2, ((1 - tt) / (1 - pk)) ** 1.5)
    cut = lo + (hi - lo) * np.sin(np.pi * np.clip(tt / (pk * 2), 0, 1)) ** 2
    x = rng.standard_normal(n); y = (lp(x, cut) - lp(x, cut * 0.25)) * e; return y / np.abs(y).max()
def pop():
    n = int(0.09 * SR); tt = np.arange(n) / SR; y = np.sin(2 * np.pi * np.cumsum(260 + 700 * np.exp(-tt * 60)) / SR) * np.exp(-tt * 45); return y / np.abs(y).max()
def ding():
    n = int(1.4 * SR); tt = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * fq * tt) * np.exp(-tt * k) for fq, a, k in [(1318.5, 1, 3), (1975.5, .5, 4), (2637, .3, 6), (659.25, .4, 2.5)])
    return y * np.minimum(1, tt / 0.004) / np.abs(y).max()
def thump():
    n = int(0.35 * SR); tt = np.arange(n) / SR; y = np.sin(2 * np.pi * np.cumsum(55 + 90 * np.exp(-tt * 25)) / SR) * np.exp(-tt * 9); return y / np.abs(y).max()
trk = np.zeros(int((DUR + 2) * SR))
def put(s, tt, g): i = int(max(0, tt) * SR); seg = trk[i:i + len(s)]; seg += s[:len(seg)] * g
WB, SWS, PP, DG, TH = whoosh(0.6), whoosh(0.25, 1500, 8000, 0.4), pop(), ding(), thump()
for b in B[1:]: put(WB, b["t0"] - 0.35, 0.28)
def at(name, lt): return bt(name)["t0"] + lt
for nm, lt in [("after1", 0.15), ("before", 0.05), ("after_photo", 0.2), ("edema", 1.5), ("months", 0.2), ("feel", 0.05)]:
    put(SWS, at(nm, lt) - 0.05, 0.12)
for nm, lt in [("after1", 0.7), ("after1", 2.2), ("before", 0.55), ("before", 1.1), ("before", 1.65), ("after_photo", 0.9), ("after_photo", 4.95),
               ("edema", 2.9), ("months", 0.4), ("months", 3.9), ("months", 7.4), ("final", 0.4), ("final", 0.6)]:
    put(PP, at(nm, lt), 0.24)
put(TH, at("after1", 0.15), 0.35); put(TH, at("edema", 0.3), 0.45); put(DG, at("feel", 1.35), 0.14); put(DG, at("final", 3.3), 0.16)
trk = trk[:int(DUR * SR)]
with wave.open("sfx.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(trk, -1, 1) * 32767).astype(np.int16).tobytes())
