import subprocess, sys, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
SW, SH = 720, 1280
FONTS = "../fonts2/M%d.ttf"
POWDER = (228, 210, 210); NAVY = (16, 46, 70); MAUVE = (103, 96, 104); WHITE = (255, 255, 255)
DUR = 35.1

# ---------- words (original timeline) ----------
WORDS = [(0.66,"сегодня"),(1.26,"мы"),(1.5,"делаем"),(1.76,"омолаживающую"),(2.78,"операцию"),(3.38,"в"),(3.48,"области"),(3.82,"лица"),
 (4.36,"то"),(4.44,"есть"),(4.58,"у"),(4.64,"нас"),(4.8,"не"),(5.02,"будет"),(5.28,"каких-то"),(5.96,"подтяжек"),(6.52,"SMAS"),
 (7.2,"мы"),(7.4,"делаем"),(7.7,"минимально"),(8.2,"инвазивную"),(8.86,"то"),(8.94,"есть"),(9.1,"мы"),(9.3,"будем"),(9.54,"делать"),
 (9.86,"верхнюю"),(10.38,"блефаропластику"),(11.64,"липофилинг"),(12.24,"нижнего"),(12.7,"века"),(13.04,"и"),(13.24,"скуловой"),(13.68,"области"),
 (14.74,"пациентку"),(15.16,"очень"),(15.4,"беспокоят"),(15.98,"вот"),(16.16,"эти"),(16.32,"марионеточные"),(17.24,"линии"),(17.58,"они"),(17.88,"заломы"),
 (18.76,"обязательно"),(19.14,"их"),(19.34,"заполним"),(20.74,"также"),(21.14,"верхнюю"),(21.7,"и"),(21.82,"нижнюю"),(22.04,"губу"),(22.38,"немножко"),(22.76,"увеличим"),
 (23.28,"и"),(23.46,"сделаем"),(24.38,"из-за"),(24.98,"того"),(25.14,"что"),(25.32,"добавим"),(25.66,"сюда"),(25.86,"немного"),(26.18,"жира"),(26.64,"сделаем"),(26.84,"более"),(27.1,"чёткий"),(27.54,"контур"),
 (28.5,"такие"),(28.86,"у"),(28.9,"нас"),(29.58,"минимальные"),(30.28,"вмешательства"),(31.7,"но"),(31.9,"будут"),(32.22,"достаточно"),(32.66,"неплохие"),(33.28,"изменения")]
WEND = 33.9

# ---------- landmarks ----------
L = np.load("lm.npy").astype(np.float64)
N = len(L)
bad = np.isnan(L[:, 1, 1]) | (L[:, 1, 1] < 450)
L[bad] = np.nan
idx = np.arange(N); good = ~bad
for k in range(478):
    for c in range(2):
        L[:, k, c] = np.interp(idx, idx[good], L[good, k, c])
def gsmooth(a, s):
    r = int(3 * s); k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * s * s)); k /= k.sum()
    pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)])
    return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, pad)
L = gsmooth(L.reshape(N, -1), 1.5).reshape(N, 478, 2)
def P(f, *ks): return L[f, list(ks)].mean(0)
def fw(f): return np.linalg.norm(L[f, 454] - L[f, 234])

# ---------- virtual camera ----------
def lin(t, a, b, va, vb): return va + (vb - va) * min(max((t - a) / (b - a), 0), 1)
raw = []
for f in range(N):
    t = f / FPS; w = fw(f)
    def frac(pt, fr, zmax=2.2): return (pt[0], pt[1], min(max(720 * fr / w, 1.0), zmax))
    if t < 9.3:
        z = lin(t, 0.4, 3.8, 1.0, 1.3)
        cx = lin(t, 0.4, 3.8, 360, max(P(f, 1)[0] + 90, 260)); raw.append((cx, 640, z))
    elif t < 11.6: raw.append(frac(P(f, 159, 386, 168), 0.6))
    elif t < 14.6: raw.append(frac(P(f, 145, 374, 1, 50, 280), 0.6))
    elif t < 20.5: raw.append(frac(P(f, 61, 291, 152, 1), 0.58))
    elif t < 23.4: raw.append(frac(P(f, 0, 17, 13), 0.72))
    elif t < 28.4: raw.append(frac(P(f, 1, 152, 234, 454), 0.52))
    else:
        z = lin(t, 28.4, 29.6, 1.25, 1.0); raw.append((lin(t, 28.4, 29.6, P(f, 1)[0], 360), 640, z))
CAM = gsmooth(np.array(raw), 7)

def crop_of(f):
    cx, cy, z = CAM[f]
    cw, ch = SW / z, SH / z
    x0 = min(max(cx - cw / 2, 0), SW - cw); y0 = min(max(cy - ch / 2, 0), SH - ch)
    return x0, y0, cw, ch
def M(f, p):
    x0, y0, cw, ch = crop_of(f)
    return np.array([(p[0] - x0) * W / cw, (p[1] - y0) * H / ch])

# ---------- text (pills) ----------
def font(w, s): return ImageFont.truetype(FONTS % w, s)
def pill(txt, size=50, bg=NAVY, fg=POWDER, weight=600, rot=0):
    f = font(weight, size); asc, desc = f.getmetrics()
    tw = int(f.getlength(txt)); th = asc + desc
    ph, pw = th + 18, tw + 52
    im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, pw - 1, ph - 1), radius=ph // 2, fill=bg + (255,))
    d.text((pw // 2, ph // 2 + 2), txt, font=f, fill=fg + (255,), anchor="mm")
    a = np.asarray(im).astype(np.float32) / 255
    if rot:
        h, w = a.shape[:2]; R = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1)
        c, s = abs(R[0, 0]), abs(R[0, 1]); nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
        R[0, 2] += nw / 2 - w / 2; R[1, 2] += nh / 2 - h / 2
        a = cv2.warpAffine(a, R, (nw, nh))
    a[..., :3] *= a[..., 3:4]
    return a
def blend(dst, img, x, y, al):
    if al <= 0.003: return
    h, w = img.shape[:2]; x, y = int(x), int(y)
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]; d = dst[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4] * al; d += s[..., :3] * al
def ease(p): p = min(max(p, 0), 1); return 1 - (1 - p) ** 3
def back(p):
    p = min(max(p, 0), 1); c = 1.9; return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2
def draw_pill(dst, img, cx, cy, t, t0, t1, shadow=True):
    """img centred at (cx,cy); pops in at t0, fades at t1"""
    if t < t0 or t > t1 + 0.3: return
    s = max(back((t - t0) / 0.32), 0.02); al = min(1, (t - t0) / 0.08) * (1 - min(1, max(0, (t - t1) / 0.3)))
    im = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s)))) if s != 1 else img
    if shadow:
        sh = cv2.GaussianBlur(im[..., 3], (0, 0), 8) * 0.45
        shi = np.zeros(sh.shape + (4,), np.float32); shi[..., 3] = sh
        blend(dst, shi, cx - im.shape[1] / 2, cy - im.shape[0] / 2 + 6, al)
    blend(dst, im, cx - im.shape[1] / 2, cy - im.shape[0] / 2, al)

# ---------- vector strokes ----------
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
    c = m + np.array([-d[1], d[0]]) * bend
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b
def smooth_curve(pts, n=60):
    pts = np.asarray(pts, float); t = np.linspace(0, 1, len(pts)); tt = np.linspace(0, 1, n)
    k = min(3, len(pts) - 1)
    return np.stack([np.polyval(np.polyfit(t, pts[:, i], k), tt) for i in range(2)], 1)

class Layer:
    """accumulates white/colored strokes into masks for one frame"""
    def __init__(s): s.items = []
    def stroke(s, pts, al, col=WHITE, th=6, dash=False, head=False):
        if al <= 0 or len(pts) < 2: return
        s.items.append(("stroke", pts, al, col, th, dash, head))
    def glow(s, center, axes, ang, al, col=POWDER):
        if al > 0: s.items.append(("glow", center, axes, ang, al, col))
    def render(s, frame):
        for it in s.items:
            m = np.zeros((H // 2, W // 2), np.uint8)
            if it[0] == "glow":
                _, c, ax, ang, al, col = it
                cv2.ellipse(m, (int(c[0] / 2), int(c[1] / 2)), (max(1, int(ax[0] / 2)), max(1, int(ax[1] / 2))), ang, 0, 360, 255, -1, cv2.LINE_AA)
                a = cv2.GaussianBlur(m, (0, 0), max(ax) / 5).astype(np.float32) / 255
                a = cv2.resize(a, (W, H))[..., None] * al
                frame *= 1 - a; frame += np.float32(col)[None, None] / 255 * a
                continue
            _, pts, al, col, th, dash, head = it
            mk = np.zeros((H, W), np.uint8)
            P_ = np.round(pts).astype(np.int32)
            if dash:
                seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
                for d0 in np.arange(0, cum[-1], 26):
                    q = [np.interp(d0, cum, pts[:, 0]), np.interp(d0, cum, pts[:, 1])]
                    cv2.circle(mk, (int(q[0]), int(q[1])), th // 2 + 1, 255, -1, cv2.LINE_AA)
            else:
                cv2.polylines(mk, [P_], False, 255, th, cv2.LINE_AA)
            if head:
                e = pts[-1]; d = pts[-1] - pts[-4 if len(pts) > 4 else 0]; d /= max(np.linalg.norm(d), 1e-6)
                n_ = np.array([-d[1], d[0]]); hl = 28
                tri = np.array([e + d * 6, e - d * hl + n_ * hl * 0.55, e - d * hl - n_ * hl * 0.55]).astype(np.int32)
                cv2.fillPoly(mk, [tri], 255, cv2.LINE_AA)
            sm = cv2.resize(mk, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            sh = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 5), (W, H)).astype(np.float32) / 255 * 0.5 * al
            frame *= 1 - sh[..., None]
            a = mk.astype(np.float32)[..., None] / 255 * al
            frame *= 1 - a; frame += np.float32(col)[None, None] / 255 * a

def env(t, t0, t1, dur):
    """draw progress (0..1) and alpha"""
    if t < t0 or t > t1 + 0.35: return 0, 0
    return ease((t - t0) / dur), 1 - min(1, max(0, (t - t1) / 0.35))

# ---------- landmark groups ----------
UP_R = [33, 246, 161, 160, 159, 158, 157, 173, 133]; UP_L = [263, 466, 388, 387, 386, 385, 384, 398, 362]
LO_R = [33, 7, 163, 144, 145, 153, 154, 155, 133]; LO_L = [263, 249, 390, 373, 374, 380, 381, 382, 362]
BROW_R = [70, 63, 105, 66, 107]; BROW_L = [300, 293, 334, 296, 336]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
JAW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]

def crease(f, up, brow):
    lid = np.array([M(f, L[f, k]) for k in up]); b = np.mean([M(f, L[f, k]) for k in brow], 0)
    off = (b[1] - lid[:, 1].mean()) * 0.38
    c = lid.copy(); c[:, 1] += off
    return smooth_curve(c[1:-1])
def undereye(f, lo, up):
    lid = np.array([M(f, L[f, k]) for k in lo]); top = np.array([M(f, L[f, k]) for k in up])
    h = lid[:, 1].mean() - top[:, 1].mean()
    c = lid.copy(); c[:, 1] += h * 0.9
    return smooth_curve(c[1:-1])
def marionette(f, corner, jaw):
    a = M(f, L[f, corner]); b = M(f, L[f, jaw]); a = a + (b - a) * 0.08
    e = a + (b - a) * 1.0
    return bez(a, e, 0.12 if corner == 61 else -0.12, 30)

LBL = {k: pill(k) for k in ["омолаживающая операция", "подтяжка", "SMAS", "минимально инвазивно", "верхняя блефаропластика",
                            "липофилинг", "нижнее веко + скулы", "марионеточные линии", "заполним", "губы: + объём", "чёткий контур"]}
CHECK = [pill(s, 40, MAUVE, WHITE) for s in ["верхняя блефаропластика", "липофилинг", "губы", "чёткий контур"]]
FINAL = pill("минимальные вмешательства", 44, POWDER, NAVY, 600)
SUB = {}
def sub_img(w):
    if w not in SUB: SUB[w] = pill(w if w == "SMAS" else w.lower(), 50, NAVY, WHITE, 500)
    return SUB[w]

def strike(frame, cx, cy, img, t, t0):
    p, al = env(t, t0, 99, 0.25)
    if p <= 0: return
    w = img.shape[1] * 0.55; a = (cx - w, cy + 8); b = (cx + w, cy - 8)
    lay = Layer(); lay.stroke(partial(np.array([a, b]), p), 1, POWDER, 7); lay.render(frame)

def overlay(frame, f, t):
    lay = Layer()
    # A-C: wide shot labels
    face = M(f, P(f, 454)) + np.array([30, 0])
    draw_pill(frame, LBL["омолаживающая операция"], 540, 300, t, 1.76, 4.2)
    p, al = env(t, 2.0, 4.2, 0.45)
    if p > 0: lay.stroke(partial(bez((420, 345), face, -0.3), p), al, WHITE, 5, head=p > 0.95)
    draw_pill(frame, LBL["подтяжка"], 330, 300, t, 5.9, 9.0); draw_pill(frame, LBL["SMAS"], 640, 300, t, 6.5, 9.0)
    if t < 9.3:
        strike(frame, 330, 300, LBL["подтяжка"], t, 6.3); strike(frame, 640, 300, LBL["SMAS"], t, 6.95)
    draw_pill(frame, LBL["минимально инвазивно"], 520, 400, t, 7.75, 9.0)
    # D: upper blepharoplasty
    p, al = env(t, 9.9, 11.55, 0.6)
    if p > 0:
        cr, cl = crease(f, UP_R, BROW_R), crease(f, UP_L, BROW_L)
        lay.stroke(partial(cr, p), al, WHITE, 8); lay.stroke(partial(cl, p), al, WHITE, 8)
        top = 250
        q, al2 = env(t, 10.3, 11.55, 0.4)
        if q > 0:
            lay.stroke(partial(bez((450, top + 40), cr[len(cr) // 2] + [0, -22], 0.2), q), al2, WHITE, 5, head=q > 0.95)
            lay.stroke(partial(bez((630, top + 40), cl[len(cl) // 2] + [0, -22], -0.2), q), al2, WHITE, 5, head=q > 0.95)
    draw_pill(frame, LBL["верхняя блефаропластика"], 540, 250, t, 10.1, 11.55)
    # E: lipofilling lower lid + cheekbones
    p, al = env(t, 12.2, 14.65, 0.5)
    if p > 0:
        lay.stroke(partial(undereye(f, LO_R, UP_R), p), al, WHITE, 6, dash=True)
        lay.stroke(partial(undereye(f, LO_L, UP_L), p), al, WHITE, 6, dash=True)
    p, al = env(t, 13.2, 14.65, 0.4)
    if p > 0:
        pulse = 0.75 + 0.25 * np.sin((t - 13.2) * 6)
        s = fw(f) * W / crop_of(f)[2]
        for k in (50, 280):
            c = M(f, L[f, k]); ax = (s * 0.13, s * 0.085); ang = np.radians(25 if k == 50 else -25)
            lay.glow(c, (ax[0] * p, ax[1] * p), np.degrees(ang), 0.45 * al * pulse, WHITE)
            th_ = np.linspace(0, 2 * np.pi, 80)
            ell = np.stack([ax[0] * np.cos(th_), ax[1] * np.sin(th_)], 1) @ np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]]) + c
            lay.stroke(partial(ell, p), al, WHITE, 5, dash=True)
    draw_pill(frame, LBL["липофилинг"], 540, 240, t, 11.7, 14.65); draw_pill(frame, LBL["нижнее веко + скулы"], 540, 320, t, 12.3, 14.65)
    # F: marionette lines
    chin = M(f, L[f, 152]); ly = min(chin[1] + 150, 1480)
    p, al = env(t, 16.3, 20.45, 0.5)
    if p > 0:
        mr, ml = marionette(f, 61, 150), marionette(f, 291, 379)
        fill = env(t, 19.34, 99, 0.6)[0]
        if fill > 0:
            lay.stroke(mr, al * fill * 0.8, POWDER, int(6 + 14 * fill)); lay.stroke(ml, al * fill * 0.8, POWDER, int(6 + 14 * fill))
        lay.stroke(partial(mr, p), al * (1 - fill), WHITE, 8); lay.stroke(partial(ml, p), al * (1 - fill), WHITE, 8)
        q, al2 = env(t, 16.7, 20.45, 0.4)
        if q > 0:
            lay.stroke(partial(bez((400, ly - 30), mr[-1] + [-6, 22], 0.1), q), al2, WHITE, 5, head=q > 0.95)
            lay.stroke(partial(bez((680, ly - 30), ml[-1] + [6, 22], -0.1), q), al2, WHITE, 5, head=q > 0.95)
    draw_pill(frame, LBL["марионеточные линии"], 540, ly, t, 16.4, 19.3)
    draw_pill(frame, LBL["заполним"], 540, ly, t, 19.34, 20.45)
    # G: lips
    p, al = env(t, 21.1, 23.35, 0.8)
    if p > 0:
        lips = np.array([M(f, L[f, k]) for k in LIPS])
        c = lips.mean(0); g = 1 + 0.06 * env(t, 22.76, 99, 0.4)[0]
        lay.stroke(partial(c + (lips - c) * (1.12 * g), p), al, WHITE, 5)
        q, al2 = env(t, 21.6, 23.35, 0.4)
        if q > 0: lay.stroke(partial(bez((540, ly - 40), c + [0, (lips[:, 1].max() - c[1]) * 1.4], 0.0), q), al2, WHITE, 5, head=q > 0.95)
    draw_pill(frame, LBL["губы: + объём"], 540, ly, t, 21.2, 23.35)
    # H: jaw contour
    p, al = env(t, 25.3, 28.35, 1.2)
    if p > 0:
        jaw = smooth_curve([M(f, L[f, k]) for k in JAW], 90)
        lay.stroke(partial(jaw, p), al, POWDER, 7)
    draw_pill(frame, LBL["чёткий контур"], 540, min(ly + 20, 1480), t, 27.1, 28.35)
    lay.render(frame)
    # I: summary
    for i, im in enumerate(CHECK):
        draw_pill(frame, im, 80 + im.shape[1] / 2, 280 + i * 78, t, 29.0 + i * 0.3, 34.2)
    draw_pill(frame, FINAL, 80 + FINAL.shape[1] / 2, 280 + 4 * 78 + 20, t, 29.6 + 1.2, 34.2)

def subtitle(frame, t):
    for i, (t0, w) in enumerate(WORDS):
        t1 = WORDS[i + 1][0] if i + 1 < len(WORDS) else WEND
        t1 = min(t1, t0 + 1.1)
        if t0 <= t < t1:
            im = sub_img(w); s = 0.85 + 0.15 * ease((t - t0) / 0.12)
            im2 = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
            blend(frame, im2, 540 - im2.shape[1] / 2, 1700 - im2.shape[0] / 2, min(1, (t - t0) / 0.05))

dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "src.mov", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "video_only.mp4"], stdin=subprocess.PIPE)
T0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0; T1 = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
f = 0
while f < N:
    buf = dec.stdout.read(SW * SH * 3)
    if len(buf) < SW * SH * 3: break
    t = f / FPS
    if T0 <= t <= T1:
        src = np.frombuffer(buf, np.uint8).reshape(SH, SW, 3)
        x0, y0, cw, ch = crop_of(f)
        crop = src[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
        img = cv2.resize(crop, (W, H), interpolation=cv2.INTER_CUBIC)
        if CAM[f][2] > 1.3:
            bl = cv2.GaussianBlur(img, (0, 0), 1.6); img = cv2.addWeighted(img, 1.5, bl, -0.5, 0)
        frame = img.astype(np.float32) / 255
        overlay(frame, f, t); subtitle(frame, t)
        enc.stdin.write((np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    f += 1
enc.stdin.close(); enc.wait(); dec.kill()

# ---------- light SFX ----------
SR = 48000; rng = np.random.default_rng(3)
def lp(x, cut):
    y = np.zeros_like(x); s = 0.0; a = 1 - np.exp(-2 * np.pi * cut / SR)
    for i in range(len(x)): s += a[i] * (x[i] - s); y[i] = s
    return y
def whoosh(d=0.5, lo=300, hi=5000, pk=0.6):
    n = int(d * SR); t = np.linspace(0, 1, n)
    e = np.where(t < pk, (t / pk) ** 2, ((1 - t) / (1 - pk)) ** 1.5)
    cut = lo + (hi - lo) * np.sin(np.pi * np.clip(t / (pk * 2), 0, 1)) ** 2
    x = rng.standard_normal(n); y = (lp(x, cut) - lp(x, cut * 0.25)) * e; return y / np.abs(y).max()
def pop():
    n = int(0.09 * SR); t = np.arange(n) / SR; ph = 2 * np.pi * np.cumsum(260 + 700 * np.exp(-t * 60)) / SR
    y = np.sin(ph) * np.exp(-t * 45); return y / np.abs(y).max()
trk = np.zeros(int((DUR + 2) * SR))
def put(s, t, g): i = int(max(0, t) * SR); seg = trk[i:i + len(s)]; seg += s[:len(seg)] * g
WB, PP, SW_ = whoosh(0.6), pop(), whoosh(0.25, 1500, 8000, 0.4)
for t in [1.76, 5.96, 6.52, 7.75, 10.1, 11.7, 12.3, 16.4, 19.34, 21.2, 27.1, 29.0, 29.3, 29.6, 29.9, 30.8]: put(PP, t, 0.22)
for t in [6.3, 6.95]: put(SW_, t - 0.05, 0.1)
for t in [9.4, 11.6, 14.6, 20.5, 23.4, 28.4]: put(WB, t - 0.35, 0.18)
trk = trk[:int(DUR * SR)]
with wave.open("sfx.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(trk, -1, 1) * 32767).astype(np.int16).tobytes())
