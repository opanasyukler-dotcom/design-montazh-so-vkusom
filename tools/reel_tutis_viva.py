# Reels: Tutis Viva stroller review. Voice-over driven edit; footage slowed ~0.5x, photos with Ken Burns,
# brand palette (#2F0600 #825D4D #C5A29C #C5B0AD #F7DFDD), Montserrat thin body + CAPS headings.
# usage: python3 reel_tutis_viva.py <src_dir> <work_dir>   (src: video.mov + *.png ; work: voice_clean.wav, mix.wav)
import sys, os, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
SRC, WK = sys.argv[1], sys.argv[2]
W, H, FPS = 1080, 1920, 24
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Montserrat-VariableFont_wght.ttf")
INK, MOCHA, ROSE, TAUPE, BLUSH = (0x2F, 0x06, 0x00), (0x82, 0x5D, 0x4D), (0xC5, 0xA2, 0x9C), (0xC5, 0xB0, 0xAD), (0xF7, 0xDF, 0xDD)

# ---- voice edit: voice source time V -> reel time (two short pauses trimmed) ----
def r(V): return V - 1.05 if V < 39.8 else (V - 1.25 if V < 41.72 else V - 1.50)
VOICE_END = r(74.10); DUR = round(VOICE_END + 1.9, 2)

# ---- video shots (source seconds) ----
SH = [0, 1.81, 3.87, 5.80, 7.68, 9.76, 11.52, 13.97, 15.66, 18.66, 20.66, 23.33, 25.54, 27.71, 29.94, 32.47, 35.80]
def vs(k, off=0.0, sp=None): return ("v", SH[k] + 0.04 + off, SH[k + 1] - 0.04, sp)
PH = {k: os.path.join(SRC, f) for k, f in [("back", "2CEF41E6-B52F-4D66-BB17-2EDFB22BC6D4.png"), ("contrast", "3FF740B6-66D5-483D-A9EB-FB6A74699C31.png"),
      ("mattress", "53CC9E81-02F7-4993-A39F-A0558AEE9A82.png"), ("spring", "7680AC4B-9B6F-47E6-A487-13CBAB711FC4.png"),
      ("cover", "C286C3EB-3333-4FA5-8CD2-2EA45CBE1ED3.png"), ("basket", "D0C1EA1F-2817-4C88-8D73-D05E589C4C68.png"),
      ("mesh", "D3C2B03B-3304-4C78-8EA4-8B86425233F2.png"), ("details", "F06CC0A3-A602-44A0-81BD-A717B119DFC2.png"),
      ("side", "FBE2B737-15F3-4997-9182-3A3FC67B37EC.png")]}
# (voice time start, source, kenburns (z0,z1,cx0,cy0,cx1,cy1) for photos)
SHOTS = [(1.45, vs(0)), (4.95, vs(3)), (8.30, ("p", "side", (1.0, 1.08, .5, .55, .5, .5))), (9.95, ("p", "details", (1.06, 1.0, .5, .5, .5, .5))),
         (11.60, vs(2)), (14.30, vs(7)), (18.50, vs(5)), (21.90, ("p", "mesh", None)), (23.80, ("p", "contrast", (1.0, 1.12, .5, .5, .5, .45))),
         (28.90, vs(1)), (32.10, vs(9)), (32.90, ("p", "spring", (1.12, 1.02, .4, .35, .45, .4))), (33.90, vs(4)), (36.10, vs(8)),
         (39.90, vs(13)), (45.10, vs(15)), (47.95, vs(6)), (52.80, ("p", "mattress", (1.0, 1.1, .5, .55, .5, .5))), (55.40, vs(14)),
         (57.85, vs(3, 0.6)), (61.30, vs(2, 0.9)), (61.90, vs(13, 1.0)), (62.85, ("p", "back", (1.0, 1.06, .5, .45, .5, .45))),
         (63.30, ("p", "cover", None)), (64.00, ("p", "mesh", None)), (64.90, ("p", "details", (1.15, 1.2, .5, .2, .5, .22))),
         (65.70, ("p", "basket", (1.0, 1.06, .5, .5, .5, .55))), (67.10, vs(10)), (70.00, vs(11, 0, 0.36))]
SHOTS = [(r(v), s) for v, s in SHOTS]

# ---- captions: (V_in, V_out, pos, heading, [(V_appear, line)], style) ----
CAPS = [(1.45, 8.30, "T", "TUTIS VIVA", [(2.3, "наша любимая коляска"), (6.12, "в цвете Crystal Soft Grey")], "hero"),
        (8.30, 11.60, "T", "ОБЗОР ФУНКЦИЙ", [(9.2, "что в ней интересного")], ""),
        (11.60, 18.50, "B", "ЛЮЛЬКА", [(12.74, "термоматериал держит"), (15.4, "комфортную температуру внутри")], ""),
        (18.50, 21.90, "B", "UPF 50+", [(19.2, "защита от ультрафиолета")], ""),
        (21.90, 23.80, "T", "КАПЮШОН", [(22.4, "большой, с вентиляцией")], ""),
        (23.80, 28.90, "B", "КОНТРАСТНЫЕ РИСУНКИ", [(25.0, "для развития зрительного"), (26.58, "восприятия малыша")], ""),
        (28.90, 33.90, "T", "ТРОЙНАЯ АМОРТИЗАЦИЯ", [(31.14, "— передние колёса"), (32.18, "— рама"), (32.58, "— задняя подвеска")], ""),
        (33.90, 39.90, "T", "МЯГКИЕ КОЛЁСА", [(34.9, "с гелевым наполнителем"), (36.12, "передние — поворотные"), (37.6, "с фиксацией")], ""),
        (39.90, 45.10, "T", "ПРОГУЛОЧНЫЙ БЛОК", [(41.94, "регулируемый наклон спинки"), (43.58, "есть положение для сна")], ""),
        (45.10, 47.95, "T", "ПОДНОЖКА", [(45.6, "одной рукой, без кнопок")], ""),
        (47.95, 52.80, "B", "КАПЮШОН", [(49.02, "закрывается полностью"), (50.2, "доп. секция и вентиляция")], ""),
        (52.80, 55.40, "T", "100 × 37 СМ", [(53.4, "спальное место")], "hero"),
        (55.40, 57.85, "B", "5-ТОЧЕЧНЫЕ РЕМНИ", [(56.2, "безопасности")], ""),
        (57.85, 67.10, "B", "В КОМПЛЕКТЕ", [(59.06, "уже всё необходимое:"), (61.36, "•  люлька"), (61.94, "•  прогулочный блок"), (62.88, "•  рюкзак"),
                                             (63.36, "•  дождевик"), (64.04, "•  москитная сетка"), (64.92, "•  чехлы"), (65.74, "•  корзина для покупок")], ""),
        (67.10, 74.15, "T", "CRYSTAL SOFT GREY", [(68.5, "спокойный серый оттенок"), (71.44, "увидела — и влюбилась")], "hero")]

# ---------------- text sprites ----------------
_fc = {}
def font(size, wght):
    k = (size, wght)
    if k not in _fc:
        f = ImageFont.truetype(FONT, size); f.set_variation_by_axes([wght]); _fc[k] = f
    return _fc[k]
def glyphs(text, size, wght, track, color):
    """per-character RGBA sprites with x positions (for staggered reveal)"""
    f = font(size, wght); x = 0; out = []
    for ch in text:
        adv = f.getlength(ch)
        if ch != " ":
            bb = f.getbbox(ch); im = Image.new("RGBA", (int(adv) + 8, int(size * 1.4)), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((2, 0), ch, font=f, fill=color + (255,))
            out.append((x, np.asarray(im).astype(np.float32) / 255))
        x += adv + track * size
    return out, x - track * size
def line_sprite(text, size, wght, color, track=0.02):
    f = font(size, wght); wd = int(sum(f.getlength(c) + track * size for c in text)) + 8
    im = Image.new("RGBA", (wd, int(size * 1.45)), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 2
    for c in text:
        col = ROSE if c == "•" else color
        d.text((x, 0), c, font=font(size, 500) if c == "•" else f, fill=col + (255,)); x += f.getlength(c) + track * size
    return np.asarray(im).astype(np.float32) / 255

def ease(x): x = np.clip(x, 0, 1); return 1 - (1 - x) ** 3
def ease_io(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

class Caption:
    def __init__(self, vin, vout, pos, head, lines, style):
        self.t0, self.t1, self.pos = r(vin), min(r(vout), DUR), pos
        hs = 70 if style == "hero" else 50
        self.head, hw = glyphs(head, hs, 500, 0.16, INK)
        self.hh = int(hs * 1.25)
        bs = 35
        self.lines = [(r(a), line_sprite(t, bs, 300, INK)) for a, t in lines]
        self.lh = int(bs * 1.5)
        self.pad = 46
        bodyw = max([s.shape[1] for _, s in self.lines] + [0])
        self.w = int(min(W - 120, max(hw, bodyw) + 2 * self.pad))
        self.h = self.pad + self.hh + 26 + len(self.lines) * self.lh + self.pad - 14
        self.x = 60
        self.y = 230 if pos == "T" else 1530 - self.h
    def alpha(self, t):
        return ease((t - self.t0) / 0.35) * (1 - ease_io((t - (self.t1 - 0.28)) / 0.28))
    def draw(self, fr, t):
        a = self.alpha(t)
        if a <= 0.001: return
        lift = (1 - ease((t - self.t0) / 0.45)) * 40
        x, y, w = self.x, int(self.y + lift), self.w
        # grow card height as lines appear
        nvis = sum(1 for ta, _ in self.lines if t >= ta - 0.05)
        hpart = [ease((t - ta + 0.05) / 0.3) for ta, _ in self.lines]
        h = int(self.pad + self.hh + (26 if self.lines else 0) + sum(hpart) * self.lh + self.pad - 14)
        h = max(h, self.pad * 2 + self.hh - 10)
        y0, y1 = max(0, y), min(H, y + h)
        if y1 <= y0: return
        reg = fr[y0:y1, x:x + w]
        blur = cv2.GaussianBlur(reg, (0, 0), 22)
        m = np.zeros((y1 - y0, w), np.float32)
        cv2.rectangle(m, (0, 0), (w - 1, y1 - y0 - 1), 1.0, -1)
        rad = 30; mm = np.zeros_like(m); cv2.rectangle(mm, (rad, 0), (w - rad, y1 - y0), 1, -1); cv2.rectangle(mm, (0, rad), (w, y1 - y0 - rad), 1, -1)
        for cx, cy in [(rad, rad), (w - rad, rad), (rad, y1 - y0 - rad), (w - rad, y1 - y0 - rad)]: cv2.circle(mm, (cx, cy), rad, 1, -1, cv2.LINE_AA)
        mm = mm[..., None] * a
        card = blur * 0.16 + np.array(BLUSH, np.float32) / 255 * 0.84
        reg[:] = reg * (1 - mm) + card * mm
        # thin rose border
        bd = np.zeros((y1 - y0, w), np.float32)
        cv2.rectangle(bd, (0, 0), (w - 1, y1 - y0 - 1), 0, -1)
        # heading: staggered letters
        hx, hy = x + self.pad, y + self.pad - 6
        n = len(self.head)
        for i, (gx, g) in enumerate(self.head):
            p = ease((t - self.t0 - 0.10 - i * 0.025) / 0.35)
            if p <= 0: continue
            paste(fr, g, int(hx + gx), int(hy + (1 - p) * 18), p * a)
        # rule
        p = ease((t - self.t0 - 0.25) / 0.5)
        ry = hy + self.hh + 8
        if p > 0 and self.lines:
            x1 = int(hx + 90 * p)
            if 0 <= ry < H: fr[ry:ry + 2, hx:x1] = fr[ry:ry + 2, hx:x1] * (1 - a) + np.array(ROSE, np.float32) / 255 * a
        ly = ry + 22
        for (ta, s), hp in zip(self.lines, hpart):
            p = ease((t - ta) / 0.4)
            if p > 0: paste(fr, s, int(hx + (1 - p) * 14), int(ly + (1 - p) * 10), p * a)
            ly += int(hp * self.lh)

def paste(fr, s, x, y, a):
    h, w = s.shape[:2]
    X0, Y0, X1, Y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if X1 <= X0 or Y1 <= Y0: return
    ss = s[Y0 - y:Y1 - y, X0 - x:X1 - x]; al = ss[..., 3:4] * a
    fr[Y0:Y1, X0:X1] = fr[Y0:Y1, X0:X1] * (1 - al) + ss[..., :3] * al

CAPT = [Caption(*c) for c in CAPS]

# ---------------- sources ----------------
_vcache = {}
def clip_frames(a, b):
    k = (a, b)
    if k not in _vcache:
        if len(_vcache) >= 3: _vcache.pop(next(iter(_vcache)))
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a + 0.1:.3f}", "-i", os.path.join(SRC, "video.mov"),
                              "-vf", "scale=1080:1920", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        n = len(raw) // (W * H * 3); _vcache[k] = np.frombuffer(raw[:n * W * H * 3], np.uint8).reshape(n, H, W, 3)
    return _vcache[k]
_pc = {}
def photo(name):
    if name not in _pc:
        im = cv2.cvtColor(cv2.imread(PH[name]), cv2.COLOR_BGR2RGB)
        if im.shape[1] > im.shape[0]:   # landscape -> card on blurred blush background
            bg = cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H)); x0 = (bg.shape[1] - W) // 2; bg = bg[:, x0:x0 + W]
            bg = cv2.GaussianBlur(bg, (0, 0), 40).astype(np.float32) * 0.55 + np.array(BLUSH, np.float32) * 0.45
            cw = W - 100; card = cv2.resize(im, (cw, int(im.shape[0] * cw / im.shape[1])), interpolation=cv2.INTER_AREA)
            ch = card.shape[0]; y0 = (H - ch) // 2 + 80
            sh = np.zeros((H, W), np.float32); cv2.rectangle(sh, (50, y0 + 18), (50 + cw, y0 + ch + 18), 1, -1); sh = cv2.GaussianBlur(sh, (0, 0), 28)
            bg = bg * (1 - 0.35 * sh[..., None])
            m = np.zeros((ch, cw), np.uint8); rad = 26
            cv2.rectangle(m, (rad, 0), (cw - rad, ch), 255, -1); cv2.rectangle(m, (0, rad), (cw, ch - rad), 255, -1)
            for cx, cy in [(rad, rad), (cw - rad, rad), (rad, ch - rad), (cw - rad, ch - rad)]: cv2.circle(m, (cx, cy), rad, 255, -1, cv2.LINE_AA)
            mm = (m.astype(np.float32) / 255)[..., None]
            bg[y0:y0 + ch, 50:50 + cw] = bg[y0:y0 + ch, 50:50 + cw] * (1 - mm) + card * mm
            _pc[name] = ("card", np.clip(bg, 0, 255).astype(np.uint8))
        else:
            _pc[name] = ("full", cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC))
    return _pc[name]

def zoomcrop(img, z, cx, cy):
    M = np.float32([[z, 0, 0], [0, z, 0]])
    M[0, 2] = np.clip(W / 2 - z * cx * W, W - z * W, 0); M[1, 2] = np.clip(H / 2 - z * cy * H, H - z * H, 0)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def shot_frame(i, t):
    t0, s = SHOTS[i]; t1 = SHOTS[i + 1][0] if i + 1 < len(SHOTS) else DUR
    d = t1 - t0; u = (t - t0) / d
    if s[0] == "v":
        _, a, b, sp = s
        if sp is None: sp = min(0.62, max(0.36, (b - a) / d))
        fr = clip_frames(a, b); st = a + (t - t0) * sp
        k = int(np.clip(round((st - a) * 48), 0, len(fr) - 1))
        z = 1.045 + 0.05 * np.clip(u, -0.2, 1.2)        # slow push-in, also hides the thin black edge in shot 2
        return zoomcrop(fr[k], z, 0.5, 0.5)
    kind, img = photo(s[1])
    if kind == "card":
        return zoomcrop(img, 1.0 + 0.05 * np.clip(u, 0, 1), 0.5, 0.52)
    z0, z1, cx0, cy0, cx1, cy1 = s[2] or (1.0, 1.08, .5, .5, .5, .5); q = ease_io(u)
    return zoomcrop(img, z0 + (z1 - z0) * q, cx0 + (cx1 - cx0) * q, cy0 + (cy1 - cy0) * q)

XF = 0.18   # cross-dissolve half width
def base(t):
    i = max(j for j in range(len(SHOTS)) if SHOTS[j][0] <= t + 1e-6) if t >= SHOTS[0][0] else 0
    f = shot_frame(i, t).astype(np.float32) / 255
    if i + 1 < len(SHOTS) and SHOTS[i + 1][0] - t < XF:
        g = shot_frame(i + 1, t).astype(np.float32) / 255; p = ease_io((XF - (SHOTS[i + 1][0] - t)) / (2 * XF)); f = f * (1 - p) + g * p
    elif i > 0 and t - SHOTS[i][0] < XF:
        g = shot_frame(i - 1, t).astype(np.float32) / 255; p = ease_io((XF + t - SHOTS[i][0]) / (2 * XF)); f = g * (1 - p) + f * p
    return f

# warm brand grade: lift blacks toward ink, a touch of blush in highlights
def grade(f):
    f = f * 0.97 + 0.012
    lum = f.mean(2, keepdims=True)
    f = f + (np.array(BLUSH, np.float32) / 255 - 0.93) * 0.18 * lum ** 2
    f = f + (np.array(INK, np.float32) / 255 - f) * 0.05 * (1 - lum) ** 3
    return f

VIG = None
def vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); d = ((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2
    return (1 - 0.14 * np.clip(d, 0, 1) ** 1.5)[..., None]

# end brand plate
END_T = VOICE_END + 0.2
def end_plate(fr, t):
    p = ease((t - END_T) / 0.9)
    if p <= 0: return fr
    ink = np.array(INK, np.float32) / 255
    return fr * (1 - 0.55 * p) + ink * 0.55 * p

def main(out, t_from=0.0, t_to=None, step=1):
    global VIG
    VIG = vignette(); t_to = t_to or DUR
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS // step), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    n0, n1 = int(t_from * FPS), int(t_to * FPS)
    for n in range(n0, n1, step):
        t = n / FPS
        f = grade(base(t)) * VIG
        f = end_plate(f, t)
        for c in CAPT:
            if c.t0 - 0.05 <= t <= c.t1 + 0.05: c.draw(f, t)
        if t > END_T + 0.3: draw_end(f, t)
        enc.stdin.write((np.clip(f, 0, 1) * 255).astype(np.uint8).tobytes())
        if n % 120 == 0: print(f"{t:.1f}/{DUR}", flush=True)
    enc.stdin.close(); enc.wait()

_end = None
def draw_end(f, t):
    global _end
    if _end is None:
        g1, w1 = glyphs("TUTIS VIVA", 64, 400, 0.22, BLUSH)
        s2 = line_sprite("Crystal Soft Grey", 34, 200, TAUPE, 0.06)
        _end = (g1, w1, s2)
    g1, w1, s2 = _end; t0 = END_T + 0.3
    x0 = int((W - w1) / 2); y0 = 860
    for i, (gx, g) in enumerate(g1):
        p = ease((t - t0 - i * 0.04) / 0.5)
        if p > 0: paste(f, g, x0 + int(gx), int(y0 + (1 - p) * 20), p)
    p = ease((t - t0 - 0.5) / 0.6)
    if p > 0:
        f[y0 + 110:y0 + 112, int(W / 2 - 60 * p):int(W / 2 + 60 * p)] = np.array(ROSE, np.float32) / 255
        paste(f, s2, int((W - s2.shape[1]) / 2), y0 + 138, p)

if __name__ == "__main__":
    import json
    if len(sys.argv) > 3 and sys.argv[3] == "info":
        print(json.dumps({"DUR": DUR, "cuts": [round(t, 3) for t, _ in SHOTS], "caps": [(round(c.t0, 2), [round(a, 2) for a, _ in c.lines]) for c in CAPT], "end": END_T}))
    elif len(sys.argv) > 3 and sys.argv[3] == "stills":
        for t in map(float, sys.argv[4].split(",")):
            VIG = vignette(); f = grade(base(t)) * VIG; f = end_plate(f, t)
            for c in CAPT:
                if c.t0 - 0.05 <= t <= c.t1 + 0.05: c.draw(f, t)
            if t > END_T + 0.3: draw_end(f, t)
            cv2.imwrite(os.path.join(WK, f"still_{t:05.1f}.jpg"), cv2.cvtColor((np.clip(f, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    else:
        main(os.path.join(WK, "video_only.mp4"))
