import subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
SP = "/tmp/claude-0/-home-user-design-montazh-so-vkusom/b26b20a4-6ac5-50f8-8404-dcbad88c88d3/scratchpad/"
BLUE = (16, 46, 70); WHITE = (255, 255, 255); POWDER = (228, 210, 210)
def osw(s): return ImageFont.truetype(SP + "v14/fonts/BN-Bold.ttf", s)          # Bebas Neue (Cyrillic)
def mont(w, s): return ImageFont.truetype(SP + f"fonts2/M{w}.ttf", s)
SUBS = []; DUR = 32.5
exec(open("helpers_src.py").read())

# ---------------- timeline (music: "One more", offset 17.44s; hits at 0, 3.27, 7.43, 10.54, drop 13.91) ----------------
BL = [("hook", 0, 3.27), ("mark", 3.27, 7.43), ("task", 7.43, 10.54), ("or", 10.54, 15.42), ("after", 15.42, 19.44),
      ("compare", 19.44, 23.45), ("final", 23.45, 26.96), ("col1", 26.96, 29.72), ("col2", 29.72, DUR)]
HITS = [0.0, 3.27, 7.43, 10.54, 11.19]
BEATS = 13.91 + 0.5017 * np.arange(30)
def block(t):
    for n, a, b in BL:
        if t < b: return n, a, b
    return BL[-1]

# ---------------- sources ----------------
class Src:
    def __init__(s, p): s.cap = cv2.VideoCapture(p); s.pos = 0; s.c = {}
    def get(s, i):
        i = int(np.clip(i, 0, s.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))
        if i in s.c: return s.c[i]
        if i != s.pos: s.cap.set(cv2.CAP_PROP_POS_FRAMES, i); s.pos = i
        ok, f = s.cap.read(); s.pos += 1
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if len(s.c) > 90: s.c.pop(next(iter(s.c)))
        s.c[i] = f; return f
DO, MK, OR, PO = Src("do.mp4"), Src("mark.mp4"), Src("or.mp4"), Src("po.mp4")
def gs(a, s):
    r = int(3 * s); k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * s * s)); k /= k.sum()
    pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)])
    return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, pad)
def fill_nan(a):
    a = a.copy()
    for j in range(a.shape[1]):
        v = a[:, j]; ok = ~np.isnan(v); a[:, j] = np.interp(np.arange(len(v)), np.where(ok)[0], v[ok])
    return a

# face blur for the "before" turnaround (lower face is in frame)
PD = np.load("pose_do.npy")
HEAD = gs(fill_nan(np.c_[PD[:, [0, 9, 10, 2, 5], 0].mean(1), PD[:, [0, 9, 10, 2, 5], 1].mean(1),
                         np.linalg.norm(PD[:, 11, :2] - PD[:, 12, :2], axis=1)]), 3)
def blur_ell(img, c, ax):
    c = np.array(c, float); ax = np.array(ax, float)
    x0, y0 = np.maximum(0, (c - ax * 1.4).astype(int)); x1, y1 = np.minimum([img.shape[1], img.shape[0]], (c + ax * 1.4).astype(int))
    if x1 <= x0 or y1 <= y0: return img
    crop = img[y0:y1, x0:x1].astype(np.float32); bl = cv2.GaussianBlur(cv2.resize(crop, None, fx=0.2, fy=0.2), (0, 0), 6)
    bl = cv2.resize(bl, (crop.shape[1], crop.shape[0]))
    m = np.zeros(crop.shape[:2], np.float32); cv2.ellipse(m, (int(c[0] - x0), int(c[1] - y0)), (int(ax[0]), int(ax[1])), 0, 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(ax) / 7)[..., None]
    img = img.copy(); img[y0:y1, x0:x1] = (crop * (1 - m) + bl * m).astype(np.uint8); return img
def do_frame(ts):
    i = int(round(ts * 30)); f = DO.get(i); hx, hy, sw = HEAD[min(i, len(HEAD) - 1)]
    r = max(sw * 0.42, 70); return blur_ell(f, (hx, hy - 0.25 * r), (r * 0.95, r * 1.3))
TRK = gs(fill_nan(np.load("trk_mark.npy")), 2.5)

def warp(img, z, cx, cy, ox=540, oy=960, size=(W, H)):
    M = np.float32([[z, 0, ox - z * cx], [0, z, oy - z * cy]])
    return cv2.warpAffine(img, M, size, flags=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255, M
def mp(M, p): return np.array([M[0, 0] * p[0] + M[0, 2], M[1, 1] * p[1] + M[1, 2]])

P7446 = cv2.cvtColor(cv2.imread("dl/IMG_7446.JPG"), cv2.COLOR_BGR2RGB); P7444 = cv2.cvtColor(cv2.imread("dl/IMG_7444.JPG"), cv2.COLOR_BGR2RGB)
def photo(img, z, fy=0.5):
    s = H / img.shape[0] * z; return warp(img, s, img.shape[1] / 2, img.shape[0] * fy, 540, 960)[0]

# ---------------- look ----------------
def bw(fr):
    g = fr @ np.float32([0.299, 0.587, 0.114]); g = np.clip((g - 0.5) * 1.18 + 0.48, 0, 1); return np.repeat(g[..., None], 3, 2)
yy_, xx_ = np.mgrid[0:H, 0:W].astype(np.float32)
EDGE = np.clip((((xx_ - 540) / 560) ** 2 + ((yy_ - 900) / 1050) ** 2) ** 0.9, 0, 1)[..., None]
def dark_fx(fr, t):
    """black light: dark pulses on the hits/beats + a soft dark leak drifting around the edges (centre stays clean)"""
    k = 0.0
    for h in HITS:
        if t >= h: k = max(k, 0.55 * np.exp(-(t - h) / 0.35))
    if t >= BEATS[0]:
        b = BEATS[np.searchsorted(BEATS, t) - 1]; k = max(k, 0.3 * np.exp(-(t - b) / 0.17))
    a = 0.6 * t + 1.3; cx, cy = 540 + 700 * np.cos(a), 960 + 1150 * np.sin(a * 0.8)
    leak = np.exp(-(((xx_ - cx) / 520) ** 2 + ((yy_ - cy) / 620) ** 2))[..., None]
    dark = np.clip(k * EDGE + 0.42 * leak * (0.35 + 0.65 * EDGE), 0, 0.62)
    return fr * (1 - dark)

# ---------------- text widgets ----------------
CH_DO, CH_PO = tag("ДО", 46), tag("ПОСЛЕ", 46)
def chip(fr, im, x, y, t, t0, t1=1e9):
    if t < t0 or t > t1 + 0.25: return
    p = ease((t - t0) / 0.35); al = min(1, (t - t0) / 0.1) * (1 - min(1, max(0, (t - t1) / 0.25)))
    blend(fr, shadow_of(im, 8, 0.4), x - (1 - p) * 140, y + 5, al); blend(fr, im, x - (1 - p) * 140, y, al)
TC = {}
def bullet(fr, t, t0, txt, x, y, mark="dot", size=40, t1=1e9):
    """list item: small navy tile + Montserrat text, slides in"""
    if t < t0 or t > t1 + 0.25: return None
    key = (txt, mark, size)
    if key not in TC:
        f = mont(600, size); asc, desc = f.getmetrics(); tw = int(f.getlength(txt)); hh = asc + desc
        im = Image.new("RGBA", (tw + 80, hh + 16), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 8, hh, hh + 8), 8, fill=BLUE + (255,))
        if mark == "plus":
            c = hh // 2; d.rectangle((c - 12, c + 8 - 3, c + 12, c + 8 + 3), fill=WHITE + (255,)); d.rectangle((c - 3, c + 8 - 12, c + 3, c + 8 + 12), fill=WHITE + (255,))
        else:
            c = hh // 2; d.ellipse((c - 7, c + 1, c + 7, c + 15), fill=WHITE + (255,))
        d.text((hh + 18, 8 + asc), txt, font=f, fill=WHITE + (255,), anchor="ls"); TC[key] = (to_np(im), hh + 18 + tw)
    im, tw = TC[key]; p = ease((t - t0) / 0.3); al = min(1, (t - t0) / 0.1) * (1 - min(1, max(0, (t - t1) / 0.25)))
    xx = x - 70 * (1 - p)
    blend(fr, shadow_of(im, 9, 0.75), xx, y + 4, al); blend(fr, im, xx, y, al)
    return np.array([xx + tw + 14, y + im.shape[0] / 2 + 4]), al
def arrow(fr, t, t0, a, b, bend=0.25, al=1.0, dur=0.45):
    if t < t0: return
    pts = bez(a, b, bend, 50); stroke(fr, partial(pts, ease((t - t0) / dur)), al, th=5, head=(t - t0) > dur * 0.6)

T_HOOK = Txt([("ЗДЕСЬ НУЖНА", "H"), ("ПОДТЯЖКА?", "HB"), ("или можно обойтись имплантами?", "SB")], size=150)
T_SEE = Txt([("СМОТРИМ НА НАШЕ ДО", "HB")], size=86, align="L", x=70)
T_NOLIFT = Txt([("ПОДТЯЖКУ", "H"), ("НЕ ПЛАНИРУЕМ", "HB")], size=150)
T_TASK = Txt([("ЗАДАЧА", "HB")], size=96, align="L", x=70)
T_OR = Txt([("ПЕРВЫЙ РЕЗУЛЬТАТ", "HB"), ("увеличение груди имплантами", "SB")], size=104)
T_CMP = Txt([("ИМПЛАНТЫ ИЛИ ПОДТЯЖКА?", "H"), ("решаем только после оценки вашего ДО", "SB")], size=88)
T_FIN = Txt([("НЕ КАЖДОЙ ГРУДИ", "H"), ("ПОСЛЕ ПОТЕРИ ОБЪЁМА", "H"), ("НУЖНА ПОДТЯЖКА", "HB")], size=104)

def trans(t, t0, s=0.07): return 1 + s * (1 - ease((t - t0) / 0.3))
def mblur(fr, lt):
    if lt > 0.1: return fr
    k = int(44 * (1 - lt / 0.1)) | 1
    return cv2.blur(fr, (1, k)) if k > 2 else fr

# compare framing (aligned on the nipple stickers)
def cmp_top(lt):
    f = do_frame(7.6 + lt * 0.25); return warp(f, 2.2, 562, 483, 540, 470, (W, 960))[0]
def cmp_bot(lt):
    f = PO.get(int((7.2 + lt * 0.18) * 30)); return warp(f, 1.2, 620, 690, 540, 470, (W, 960))[0]

PAFT = cv2.cvtColor(cv2.imread("dl/После"), cv2.COLOR_BGR2RGB)
def short_arrow(fr, t, t0, src, tgt, al, L=125, bend=0.25):
    d = np.asarray(src, float) - tgt; d /= max(np.linalg.norm(d), 1e-6)
    arrow(fr, t, t0, tgt + d * L, tgt + d * 12, bend, al, dur=0.3)
def brackets(fr, t, t0, c, hw, hh, al=1.0):
    if t < t0: return
    p = ease((t - t0) / 0.35); s = 1.18 - 0.18 * p; a = min(1, (t - t0) / 0.15) * al
    x0, x1, y0, y1 = c[0] - hw * s, c[0] + hw * s, c[1] - hh * s, c[1] + hh * s; L = 80
    for (x, y, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        stroke(fr, np.array([[x, y + dy * L], [x, y], [x + dx * L, y]], float), a, th=6)
def column(img, z, cx, cy, oy=500):
    return warp(img, z, cx, cy, 270, oy, (540, H))[0]

def render(t):
    n, t0, t1 = block(t); lt = t - t0; flash = 0
    if n == "hook":
        if t < 1.95:
            fr, _ = warp(do_frame(6.6 + t * 0.9), 1.55 + 0.04 * t, 560, 560, 540, 700)
        else:
            fr, _ = warp(do_frame(0.9 + (t - 1.95) * 0.9), 1.7 * trans(t, 1.95), 660, 560, 540, 760); fr = mblur(fr, t - 1.95)
        fr = bw(fr)
        chip(fr, CH_DO, 60, 228, t, 0.1)
        T_HOOK.draw(fr, 1150, t, 0.12, t1 - 0.05, cps=30)
    elif n == "mark":
        st = 3.9 + lt; cx, cy, r = TRK[int(st * 30)]
        fr, M = warp(MK.get(int(st * 30)), 1.0 * trans(t, t0), 540, 960); fr = mblur(fr, lt)
        chip(fr, CH_DO, 60, 228, t, t0 + 0.05)
        s = mp(M, (cx, cy)); rr = r * M[0, 0]
        tgt = [s + np.array([-1.2 * rr, 1.25 * rr]), s + np.array([-0.4 * rr, -2.3 * rr]), s + np.array([5.6 * rr, -0.4 * rr])]
        frm = [s + np.array([-4 * rr, 3 * rr]), s + np.array([-3.5 * rr, -4 * rr]), s + np.array([6.5 * rr, 3 * rr])]
        tb = 6.0
        if t < tb + 0.25:
            T_SEE.draw(fr, 1130, t, t0 + 0.1, tb, cps=34)
            for k, (txt, d0) in enumerate([("небольшой объём", 1.0), ("недостаточная наполненность сверху", 1.6), ("лёгкая асимметрия", 2.2)]):
                res = bullet(fr, t, t0 + d0, txt, 70, 1260 + k * 70, size=36, t1=tb)
                if res is not None and t < tb: short_arrow(fr, t, t0 + d0 + 0.1, frm[k], tgt[k], res[1], bend=0.2 if k != 2 else -0.2)
        T_NOLIFT.draw(fr, 1120, t, tb + 0.05, t1 - 0.05, cps=32)
    elif n == "task":
        st = 8.6 + lt; cx, cy, r = TRK[int(st * 30)]
        z = 1.3 * trans(t, t0)
        fr, M = warp(MK.get(int(st * 30)), z, cx - 60, cy - 40, 540, 760); fr = mblur(fr, lt)
        s = mp(M, (cx, cy)); rr = r * z
        tgt = [s + np.array([-1.3 * rr, 1.0 * rr]), s + np.array([-0.3 * rr, -2.2 * rr]), s + np.array([-3.0 * rr, 0.3 * rr])]
        frm = [s + np.array([-3 * rr, 3.5 * rr]), s + np.array([-3 * rr, -4 * rr]), s + np.array([-6 * rr, 1.5 * rr])]
        T_TASK.draw(fr, 1150, t, t0 + 0.08, t1 - 0.05, cps=30)
        for k, (txt, d0) in enumerate([("добавить объём", 0.55), ("сделать грудь более наполненной", 1.15), ("сохранить естественные пропорции", 1.75)]):
            res = bullet(fr, t, t0 + d0, txt, 70, 1300 + k * 72, size=36, t1=t1 - 0.05)
            if res is not None: short_arrow(fr, t, t0 + d0 + 0.1, frm[k], tgt[k], res[1], bend=0.2)
    elif n == "or":
        if t < 12.9:
            c0 = 11.19 if t >= 11.19 else t0
            fr, _ = warp(OR.get(int((1.0 + lt) * 30)), (1.06 + 0.04 * lt) * trans(t, c0, 0.1), 540, 880, 540, 880)
            fr = np.clip((fr - 0.04) * 1.2, 0, 1); bc = (540, 870)
        elif t < 13.91:
            fr = mblur(photo(P7446, 1.06 * trans(t, 12.9, 0.1) + 0.03 * (t - 12.9)), t - 12.9); fr = np.clip(fr * 1.12, 0, 1); bc = (560, 820)
        else:
            fr = photo(P7444, 1.08 * trans(t, 13.91, 0.14) + 0.025 * (t - 13.91)); fr = np.clip(fr * 1.12, 0, 1); bc = (540, 860)
            flash = 0.6 * max(0, 1 - (t - 13.91) / 0.2)
        fr *= 1 - 0.38 * EDGE                                         # spotlight on the result
        brackets(fr, t, 10.75 if t < 12.9 else (12.9 if t < 13.91 else 13.91), bc, 400, 340, 0.95)
        T_OR.draw(fr, 1350, t, 12.05, t1 - 0.05, cps=34)
    elif n == "after":
        tw = t0
        b_ = warp(PO.get(int((6.4 + lt * 0.9) * 30)), 1.0, 600, 860, 540, 760)[0]
        if t >= 17.94:
            b_ = warp(PO.get(int((3.25 + (t - 17.94) * 0.8) * 30)), 1.05 * trans(t, 17.94), 470, 860, 540, 760)[0]; b_ = mblur(b_, t - 17.94)
        if lt < 0.45:
            a_ = photo(P7444, 1.08 + 0.025 * (t - 13.91)); a_ = np.clip(a_ * 1.12, 0, 1) * (1 - 0.38 * EDGE)
            p = ease(lt / 0.45); edge = (xx_ * 0.55 + yy_) / (W * 0.55 + H)
            m = np.clip((p * 1.25 - edge) / 0.12, 0, 1)[..., None]
            a_ = cv2.resize(a_[int(H * 0.04 * p):H - int(H * 0.04 * p), int(W * 0.04 * p):W - int(W * 0.04 * p)], (W, H))
            fr = a_ * (1 - m) + b_ * m
            band = np.exp(-((p * 1.25 - edge - 0.06) / 0.05) ** 2)[..., None]; fr = fr * (1 - 0.6 * band)
        else: fr = b_
        chip(fr, CH_PO, 60, 228, t, t0 + 0.4)
        for k, (txt, d0) in enumerate([("больше объёма", 16.42), ("больше наполненности", 16.92), ("без подтяжки", 17.42)]):
            bullet(fr, t, d0, txt, 70, 1240 + k * 82, mark="plus", size=44)
    elif n == "compare":
        fr = np.zeros((H, W, 3), np.float32)
        top, bot = cmp_top(lt), cmp_bot(lt)
        o1 = -W * (1 - ease(lt / 0.4)); o2 = W * (1 - ease((lt - 0.1) / 0.4))
        fr[:960] = cv2.warpAffine(top, np.float32([[1, 0, o1], [0, 1, 0]]), (W, 960)); fr[960:] = cv2.warpAffine(bot, np.float32([[1, 0, o2], [0, 1, 0]]), (W, 960))
        chip(fr, CH_DO, 60, 228, t, t0 + 0.3); chip(fr, CH_PO, 60, 1080, t, t0 + 0.45)
        pb = ease((lt - 0.5) / 0.3)
        if pb > 0:
            r_ = np.zeros((190, int(W * pb), 4), np.float32); r_[..., :3] = np.float32(BLUE) / 255; r_[..., 3] = 0.93; blend(fr, r_, 0, 865, 1)
        T_CMP.draw(fr, 870, t, t0 + 0.6, t1 - 0.05, cps=34)
    elif n == "final":
        st = 1.55 + lt * 0.55
        fr, _ = warp(PO.get(int(st * 30)), 1.0 * trans(t, t0) + 0.02 * lt, 540, 900, 540, 820); fr = mblur(fr, lt)
        chip(fr, CH_PO, 60, 228, t, t0 + 0.1)
        T_FIN.draw(fr, 1160, t, t0 + 0.2, t1 - 0.05, cps=30)
    else:
        # result collages: ДО | ПОСЛЕ side by side, held so the result can be examined
        z = (1 + 0.025 * lt) * trans(t, t0, 0.08)
        if n == "col1":
            L = column(do_frame(7.9 + lt * 0.12), 1.56 * z, 562, 483); R = column(PAFT, 0.94 * z, 757, 532)
        else:
            L = column(do_frame(1.0 + lt * 0.1), 3.0 * z, 700, 430, 640); R = column(PO.get(int((3.45 + lt * 0.1) * 30)), 1.0 * z, 360, 860, 640)
        fr = np.zeros((H, W, 3), np.float32)
        o1 = -140 * (1 - ease(lt / 0.3)); o2 = 140 * (1 - ease(lt / 0.3))
        fr[:, :540] = cv2.warpAffine(L, np.float32([[1, 0, o1], [0, 1, 0]]), (540, H)); fr[:, 540:] = cv2.warpAffine(R, np.float32([[1, 0, o2], [0, 1, 0]]), (540, H))
        fr[:, 537:543] = 1.0 * min(1, lt / 0.4)
        chip(fr, CH_DO, 40, 228, t, t0 + 0.3); chip(fr, CH_PO, 580, 228, t, t0 + 0.4)
    fr = dark_fx(fr, t) if n not in ("col1", "col2") else fr * (1 - 0.15 * EDGE)
    blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
    if flash > 0: fr = fr * (1 - flash) + flash
    if t > DUR - 0.7: fr *= 1 - 0.9 * (t - (DUR - 0.7)) / 0.7
    return np.clip(fr, 0, 1)

# typed lines for the keyboard SFX: (start, cps, text)
TYPED = [(0.12, 30, T_HOOK), (3.37, 34, T_SEE), (6.05, 32, T_NOLIFT), (7.51, 30, T_TASK), (12.05, 34, T_OR), (20.04, 34, T_CMP), (23.65, 30, T_FIN)]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for ts in sys.argv[1:]:
            cv2.imwrite(f"f/o_{ts}.jpg", cv2.cvtColor((render(float(ts)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        sys.exit()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "video.mp4"], stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        enc.stdin.write((render(i / FPS) * 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
