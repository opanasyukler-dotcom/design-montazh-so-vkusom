import subprocess, wave, sys
import numpy as np, cv2, librosa
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 48000
SP = "/tmp/claude-0/-home-user-design-montazh-so-vkusom/b26b20a4-6ac5-50f8-8404-dcbad88c88d3/scratchpad/"
BLUE = (16, 46, 70); WHITE = (255, 255, 255); LIME = (214, 236, 46); RED = (226, 38, 64)
def osw(s): return ImageFont.truetype(SP + "v4/fonts/Oswald-600.ttf", s)
def mont(w, s): return ImageFont.truetype(SP + f"fonts2/M{w}.ttf", s)
SUBS = []; DUR = 45.2; M0 = 9.6
exec(open("helpers_src.py").read())
def pulse(t): return 1.0

# ---------------- face blur ----------------
def blur_ell(img, c, ax, sig=None):
    c = np.array(c, float); ax = np.array(ax, float); sig = sig or max(ax) / 3
    x0, y0 = np.maximum(0, (c - ax * 1.5).astype(int)); x1, y1 = (c + ax * 1.5).astype(int)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 <= x0 or y1 <= y0: return img
    crop = img[y0:y1, x0:x1]; bl = cv2.GaussianBlur(cv2.resize(crop, None, fx=0.25, fy=0.25), (0, 0), sig / 4)
    bl = cv2.resize(bl, (crop.shape[1], crop.shape[0]))
    m = np.zeros(crop.shape[:2], np.float32)
    cv2.ellipse(m, (int(c[0] - x0), int(c[1] - y0)), (int(ax[0]), int(ax[1])), 0, 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(ax) / 8)[..., None]
    img[y0:y1, x0:x1] = (crop * (1 - m) + bl * m).astype(img.dtype); return img

# ---------------- photos (aligned on the stickers) ----------------
def rgb(p): return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
DO = blur_ell(rgb("do.jpg"), (1595, 690), (390, 480), 90)
PO = blur_ell(rgb(SP + "v11/dl/После"), (760, 20), (280, 230), 60)
COL = rgb(SP + "v11/dl/Коллаж до_после")
def align_M(L, R, tL=(340, 670), tR=(740, 670)):
    s = (tR[0] - tL[0]) / np.hypot(R[0] - L[0], R[1] - L[1])
    mx, my = (L[0] + R[0]) / 2, (L[1] + R[1]) / 2
    return np.float32([[s, 0, (tL[0] + tR[0]) / 2 - s * mx], [0, s, (tL[1] + tR[1]) / 2 - s * my]])
MDO = align_M((1240, 2135), (2050, 2150)); MPO = align_M((499, 893), (1027, 902))
def zM(M, z, fx=540, fy=800):
    Z = np.float32([[z, 0, fx - z * fx], [0, z, fy - z * fy], [0, 0, 1]]); return (Z @ np.vstack([M, [0, 0, 1]]))[:2]
def photo(img, M, z=1.0, fx=540, fy=800, dx=0, dy=0, size=(W, H)):
    M = zM(M, z, fx, fy); M[0, 2] += dx; M[1, 2] += dy
    return cv2.warpAffine(img, M, size, flags=cv2.INTER_AREA if M[0, 0] < 1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255

# collage: fits the frame; side "after" crop = upper part without the burned-in caption
MCOL = np.float32([[W / COL.shape[1], 0, 0], [0, W / COL.shape[1], 0]])
sc = H / 1090; MSIDE = np.float32([[sc, 0, 540 - sc * 720], [0, sc, 0]])

# ---------------- video ----------------
p = subprocess.run(["ffmpeg", "-v", "error", "-i", "v.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
V = np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)
P = np.load("pose_v.npy")
def gs(a, s):
    r = int(3 * s); k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * s * s)); k /= k.sum()
    pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)])
    return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, pad)
head = P[:, [0, 2, 5, 7, 8], :2].mean(1); shm = P[:, [11, 12], :2].mean(1)
hr = np.maximum(np.linalg.norm(P[:, 11, :2] - P[:, 12, :2], axis=1) * 0.42, (shm[:, 1] - head[:, 1]) * 0.75)
HEAD = gs(np.c_[head, hr], 3)
TORSO = gs(P[:, [11, 12, 23, 24], :2].mean(1), 6)
VC = {}
def vframe(ts, z=1.3, fyo=-120):
    i = int(np.clip(round(ts * 30), 0, len(V) - 1))
    if i not in VC:
        f = V[i].copy(); hx, hy, r = HEAD[i]
        VC[i] = blur_ell(f, (hx, hy - 0.1 * r), (r * 0.95, r * 1.25), 45)
    c = TORSO[i]; M = np.float32([[z, 0, 540 - z * c[0]], [0, z, 960 + fyo - z * c[1]]])
    M[0, 2] = np.clip(M[0, 2], W - z * W, 0); M[1, 2] = np.clip(M[1, 2], H - z * H, 0)
    return cv2.warpAffine(VC[i], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255

# ---------------- text widgets ----------------
def chip(txt): return tag(txt, 40)
CH_DO, CH_PO = chip("ДО"), chip("ПОСЛЕ")
def slide_chip(frame, im, x, y, t, t0, t1=1e9, frm=-1):
    if t < t0 or t > t1 + 0.25: return
    p = ease((t - t0) / 0.35); al = min(1, (t - t0) / 0.1) * (1 - min(1, max(0, (t - t1) / 0.25)))
    blend(frame, shadow_of(im, 8, 0.4), x + frm * (1 - p) * 160, y + 5, al); blend(frame, im, x + frm * (1 - p) * 160, y, al)

def block_img(lines, size, lime_last=False):
    fs = [osw(size) for _ in lines]; asc, desc = fs[0].getmetrics(); lh = int((asc + desc) * 0.98)
    bw = int(max(f.getlength(l) for f, l in zip(fs, lines))); h = lh * len(lines) + 30
    im = Image.new("RGBA", (bw + 40, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for k, l in enumerate(lines):
        d.text(((bw + 40) / 2, 8 + k * lh + asc), l, font=fs[k], fill=WHITE + (255,), anchor="ms")
    last_w = fs[-1].getlength(lines[-1]); uy = 8 + (len(lines) - 1) * lh + asc + 12
    return to_np(im), ((bw + 40 - last_w) / 2, (bw + 40 + last_w) / 2, uy)
BRC = {}
def bracket(frame, t, t0, lines, yc, size=60, t1=1e9):
    """two bars open from the centre and reveal the text, lime underline under the last line"""
    if t < t0 or t > t1 + 0.3: return
    key = (tuple(lines), size)
    if key not in BRC: BRC[key] = block_img(lines, size)
    im, (u0, u1, uy) = BRC[key]; h, w = im.shape[:2]; x0 = 540 - w / 2; y0 = yc - h / 2
    fade = 1 - min(1, max(0, (t - t1) / 0.3)); lt = t - t0
    grow = ease(lt / 0.18); op = ease((lt - 0.15) / 0.45); half = 6 + op * (w / 2 + 10)
    reveal = im.copy(); xs = np.arange(w)[None, :, None] + x0
    reveal *= (np.abs(xs - 540) < half - 4).astype(np.float32)
    blend(frame, shadow_of(reveal, 10, 0.6), x0, y0 + 4, fade); blend(frame, reveal, x0, y0, fade)
    bh = h * grow; al = fade * (1 - 0.35 * op)
    for sx in (-1, 1):
        bx = int(540 + sx * half); r = np.zeros((int(max(bh, 2)), 6, 4), np.float32); r[...] = 1
        blend(frame, r, bx - 3, yc - bh / 2, al)
    up = ease((lt - 0.6) / 0.35)
    if up > 0:
        lw = (u1 - u0) * up; r = np.zeros((7, int(max(lw, 1)), 4), np.float32); r[..., :3] = np.float32(LIME) / 255; r[..., 3] = 1
        blend(frame, r, x0 + u0, y0 + uy, fade)

def items_list(frame, t, entries, x, y, gap=58):
    """thin list items sliding in on the beat, each with a small lime dash"""
    for k, (txt, t0) in enumerate(entries):
        if t < t0: continue
        key = ("li", txt)
        if key not in BRC:
            f = mont(500, 40); asc, desc = f.getmetrics(); tw = int(f.getlength(txt))
            im = Image.new("RGBA", (tw + 70, asc + desc + 12), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
            d.rectangle((0, (asc + desc) // 2 + 2, 36, (asc + desc) // 2 + 8), fill=LIME + (255,))
            d.text((54, 6 + asc), txt, font=f, fill=WHITE + (255,), anchor="ls"); BRC[key] = to_np(im)
        im = BRC[key]; p = ease((t - t0) / 0.3); al = min(1, (t - t0) / 0.12)
        blend(frame, shadow_of(im, 8, 0.6), x - 60 * (1 - p), y + k * gap + 4, al); blend(frame, im, x - 60 * (1 - p), y + k * gap, al)

EMO = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109)
def heart(sz):
    im = Image.new("RGBA", (136, 128), (0, 0, 0, 0)); ImageDraw.Draw(im).text((0, 0), "❤️", font=EMO, embedded_color=True)
    im = im.crop(im.getbbox()).resize((sz, int(sz * 0.9)), Image.LANCZOS); return to_np(im)
HEART = heart(96)

# subtitles in short white chunks (no patient audio in the materials)
def chunks(t0, t1, parts):
    d = (t1 - t0) / len(parts); return [(t0 + i * d, t0 + (i + 1) * d - 0.04, p) for i, p in enumerate(parts)]
SUB = (chunks(0.45, 4.05, ["после двух родов", "грудь моя поменялась", "она стала уже", "не той формы", "размеры стали разные"]) +
       chunks(4.25, 7.95, ["и я уже пришла", "к этому выводу", "что надо что-то", "делать с этим"]) +
       chunks(8.1, 11.45, ["в телеграме мне", "анна романовна попалась", "я очень к ней", "прониклась", "и уже не искала никого"]) +
       chunks(13.1, 19.25, ["волнительно", "так как это", "в первый раз", "у меня такой был опыт", "но всё было нормально", "ничего", "сверхъестественного"]) +
       chunks(19.5, 23.25, ["грудь не болела", "может быть", "как после тренировок", "напряжение какое-то было", "дискомфорт"]) +
       chunks(33.0, 35.9, ["очень довольна", "очень довольна"]))
SC = {}
def subs(frame, t, y=1530):
    for t0, t1, w in SUB:
        if t0 <= t < t1:
            if w not in SC:
                f = mont(500, 44); asc, desc = f.getmetrics(); tw = int(f.getlength(w))
                im = Image.new("RGBA", (tw + 20, asc + desc + 10), (0, 0, 0, 0)); ImageDraw.Draw(im).text((10, 5 + asc), w, font=f, fill=WHITE + (255,), anchor="ls")
                a = to_np(im); SC[w] = (a, shadow_of(a, 7, 0.85))
            im, sh = SC[w]; p = ease((t - t0) / 0.12); al = min(1, (t - t0) / 0.06)
            blend(frame, sh, 540 - im.shape[1] / 2, y - im.shape[0] / 2 + 3 + 14 * (1 - p), al)
            blend(frame, im, 540 - im.shape[1] / 2, y - im.shape[0] / 2 + 14 * (1 - p), al)

def bar(frame, y0, y1, al=1.0, prog=1.0):
    w = int(W * prog)
    if w <= 0: return
    r = np.zeros((y1 - y0, w, 4), np.float32); r[..., :3] = np.float32(BLUE) / 255; r[..., 3] = 1.0
    blend(frame, r, 0, y0, al)

T_HOOK = Txt([("ПОСЛЕ ДВУХ РОДОВ", "H"), ("ГРУДЬ СТАЛА", "H"), ("СОВСЕМ НЕ ТОЙ", "HB")], size=74)
T_KNOW = Txt([("ЗНАКОМОЕ ОЩУЩЕНИЕ?", "HB")], size=64, align="L", x=80)
T_AFT = Txt([("ПОСЛЕ ЛИПОФИЛИНГА ГРУДИ", "HB"), ("собственная жировая ткань, без имплантов", "SB")], size=60)
T_FIRST = Txt([("ПЕРВАЯ ОПЕРАЦИЯ", "HB"), ("И, КОНЕЧНО, БЫЛО ВОЛНИТЕЛЬНО", "H")], size=58)
T_MON = Txt([("2,5 МЕСЯЦА ПОСЛЕ ОПЕРАЦИИ", "H")], size=54)

# ---------------- frame composer ----------------
def trans(t, t0, strong=False):
    """punch-in at a cut: extra zoom + (optional) flash"""
    lt = t - t0
    return 1 + (0.10 if strong else 0.06) * (1 - ease(lt / 0.32)), (0.55 * max(0, 1 - lt / 0.18) if strong else 0)
def mblur(fr, lt):
    if lt > 0.12: return fr
    k = int(40 * (1 - lt / 0.12)) | 1
    return cv2.blur(fr, (k, 1)) if k > 2 else fr

