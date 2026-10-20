_b = open("base.py").read()
for a, b in [("DUR = 41.0", "DUR = 35.4"), ("ease((t - t0) / 0.3))\ndef mblur", "ease((t - t0) / 0.6))\ndef mblur"),
             ("if lt > 0.1: return fr", "if lt > 0.18: return fr"), ("(1 - lt / 0.1)", "(1 - lt / 0.18)"),
             ("ti = t0 + k * 0.24; p = ease((t - ti) / 0.65)" if False else "ti = t0 + k * 0.16; p = ease((t - ti) / 0.45)", "ti = t0 + k * 0.24; p = ease((t - ti) / 0.65)"),
             ("out = 1 - ease((t - t1) / 0.3) if t > t1 else 1.0", "out = 1 - ease((t - t1) / 0.4) if t > t1 else 1.0")]:
    assert _b.count(a) == 1, a; _b = _b.replace(a, b)
exec(_b)
exec(open("fx.py").read()); exec(open("draw.py").read())

SEGS = [(70.72, 77.62, 1.60),    # в первый месяц рубец всегда выглядит хорошо… хороший тоненький рубец, хорошо заживший
        (80.55, 88.62, 9.10),    # но со второго месяца образуется больше коллагена… нормальным его состоянием
        (88.60, 91.98, 17.90),   # и только с третьего месяца рубец начинает белеть
        (101.60, 107.05, 21.80)] # если он зажил хорошо, он сам побелеет… к шестому примерно, или к году
def src_of(t):
    best = None
    for si, so, t0 in SEGS:
        if t0 <= t <= t0 + so - si: return si + t - t0
        if t < t0: return max(0.0, si - (t0 - t)) if best is None else min(best[0] + t - best[1], si)
        best = (so, t0 + so - si)
    return best[0] + t - best[1]
BL = [("m1", 0, 8.6), ("m2", 8.6, 17.6), ("m3", 17.6, 21.6), ("m6", 21.6, 27.6), ("post", 27.6, 31.4), ("end", 31.4, DUR)]
XF = 0.35
def block(t):
    for n, a, b in BL:
        if t < b: return n, a, b
    return BL[-1]
CUTS[:] = [b[1] for b in BL[1:]]
V = Src("v/8294.mp4"); NF = int(V.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
NIP = np.load("nip94.npy"); NIP[:, 0] = cv2.GaussianBlur(NIP[:, :1], (1, 0), sigmaX=0, sigmaY=2)[:, 0] if False else NIP[:, 0]
from scipy.ndimage import gaussian_filter1d
NIP[:, 0] = gaussian_filter1d(NIP[:, 0], 2); NIP[:, 1] = gaussian_filter1d(NIP[:, 1], 2)
def _sticker(r):
    S = 4 * r; im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = S // 2; R_ = 2 * r
    d.ellipse((c - R_, c - R_, c + R_, c + R_), fill=BLUE + (255,)); d.ellipse((c - R_ + 7, c - R_ + 7, c + R_ - 7, c + R_ - 7), outline=(255, 255, 255, 200), width=3)
    f = mont(500, int(R_ * 0.36)); d.text((c, c - R_ * 0.2), "ANNA", font=f, fill=(255, 255, 255, 255), anchor="mm"); d.text((c, c + R_ * 0.24), "TVAURI", font=f, fill=(255, 255, 255, 255), anchor="mm")
    return np.asarray(im.resize((2 * r, 2 * r), Image.LANCZOS)).astype(np.float32) / 255
STK = _sticker(46)
def raw(i):
    f = V.get(min(max(i, 0), NF - 1)).copy(); j = i - 66 * 30
    if 0 <= j < len(NIP):                                                # sticker on the breast shown on the tablet
        x, y = NIP[j, :2]; h, w = STK.shape[:2]; x0, y0 = int(x - w / 2), int(y - h / 2)
        a = STK[..., 3:4]; reg = f[y0:y0 + h, x0:x0 + w].astype(np.float32); f[y0:y0 + h, x0:x0 + w] = (reg * (1 - a) + STK[..., :3] * 255 * a).astype(np.uint8)
    return f
def tablet_mask(t, z, cx, cy, oy):
    """soft ellipse over the tablet screen (in output pixels): kept bright, no haze / vignette there"""
    j = int(round(src_of(t) * 30)) - 66 * 30
    if not 0 <= j < len(NIP): return None
    x, y = NIP[j, :2]; ox, oy_ = 540 + z * (x - cx), oy + z * (y + 70 - cy)
    m = np.clip(1 - (((xx_ - ox) / (230 * z)) ** 2 + ((yy_ - oy_) / (330 * z)) ** 2), 0, 1)
    return np.clip(m * 2.2, 0, 1)[..., None]
def look_tablet(fr, t, z, cx, cy, oy):
    plain = fr.copy(); graded = look(fr, t) * (1 - HAZE); m = tablet_mask(t, z, cx, cy, oy)
    if m is None: return graded
    return graded * (1 - m) + (plain * 0.97 + 0.02) * m
def vframe(ts, z, cx, cy, oy): return warp(raw(int(round(ts * 30))), z, cx, cy, 540, oy)
SHOTS = {"m1": (1.35, 560, 800, 1150), "m2": (1.6, 555, 720, 1180), "m3": (1.35, 560, 800, 1150), "m6": (1.55, 555, 730, 1170)}

# ---------------- the scar card: phase label, skin strip with a scar whose colour follows the months, timeline ----------------
PH = [(0.0, "1 МЕСЯЦ", "после операции"), (8.6, "2 МЕСЯЦ", "больше коллагена"), (17.6, "3 МЕСЯЦ", "и дальше"), (21.6, "6 МЕСЯЦЕВ — 1 ГОД", "")]
PHI = {lab: (timg(lab, "H", 76), timg(sub, "S", 0) if sub else None) for _, lab, sub in PH}
# scar colour / width / redness per reel time (key points)
KF = [(0.0, (214, 150, 150), 6), (8.6, (214, 150, 150), 6), (13.2, (196, 62, 70), 11), (17.6, (196, 62, 70), 11),
      (21.0, (214, 138, 138), 8), (25.0, (242, 226, 220), 5), (DUR, (242, 226, 220), 5)]
def scar_state(t):
    for (a, ca, wa), (b, cb, wb) in zip(KF, KF[1:]):
        if a <= t <= b:
            p = ease((t - a) / max(b - a, 1e-6)); return np.float32(ca) * (1 - p) + np.float32(cb) * p, wa + (wb - wa) * p
    return np.float32(KF[-1][1]), KF[-1][2]
SCAR = bez((250, 420), (420, 395), (660, 445), (830, 410), 90)
def scar_card(fr, t, t0card=0.15, y0=215):
    al = min(1, (t - t0card) / 0.4)
    if al <= 0: return
    card_m = np.zeros((H, W), np.float32); cv2.rectangle(card_m, (150, y0), (930, y0 + 360), 1, -1); card_m = cv2.GaussianBlur(card_m, (0, 0), 3)[..., None] * al
    fr[:] = fr * (1 - 0.72 * card_m) + np.float32([8, 22, 38]) / 255 * 0.72 * card_m          # navy card
    skin = np.zeros((H, W), np.float32); cv2.rectangle(skin, (200, y0 + 140), (880, y0 + 270), 1, -1); skin = cv2.GaussianBlur(skin, (0, 0), 6)[..., None] * al
    fr[:] = fr * (1 - skin) + np.float32([226, 186, 164]) / 255 * skin
    col, w = scar_state(t); draw_p = ease((t - t0card - 0.3) / 0.9)
    if t < 17.6 and t > 9.5:                                             # collagen: fine fibres build up around the scar
        rng = np.random.RandomState(7); k = int(40 * ease((t - 9.5) / 3.5))
        for i in range(k):
            x = 260 + rng.rand() * 560; y = 420 + (rng.rand() - 0.5) * 70 + y0 - 215; a = rng.rand() * np.pi
            stroke(fr, np.float32([[x, y], [x + 22 * np.cos(a), y + 10 * np.sin(a)]]), 1, (205, 110, 110), 2, 0.55 * al)
    glow = np.zeros((H, W), np.uint8); cv2.polylines(glow, [np.int32(part(SCAR + [0, y0 - 215], draw_p))], False, 255, int(w * 4), cv2.LINE_AA)
    g = cv2.GaussianBlur(glow.astype(np.float32) / 255, (0, 0), w * 1.4)[..., None] * 0.35 * al * max(0, (col[0] - col[1]) / 140)   # redness halo
    fr[:] = fr * (1 - g) + col / 255 * g
    stroke(fr, SCAR + [0, y0 - 215], draw_p, tuple(col), int(w), al)
    # phase label
    cur = [p for p in PH if t >= p[0]][-1]; ph_t = cur[0]; lab, sub = PHI[cur[1]]
    pa = min(1, (t - max(ph_t, t0card)) / 0.35) * al; dx = 40 * (1 - ease((t - max(ph_t, t0card)) / 0.45))
    ablend(fr, lab, 190 + dx, y0 + 28, pa, WHITE, BLUE)
    if sub is not None: ablend(fr, sub, 200 + lab.shape[1] + 18 + dx, y0 + 62, pa, PINK, ROSE)
    # timeline: 1 · 2 · 3 · 6 · 12 months
    marks = [(1, 0.0), (2, 8.6), (3, 17.6), (6, 21.6), (12, 24.2)]
    xs = np.linspace(240, 840, len(marks)); yl = y0 + 318
    stroke(fr, np.float32([[xs[0], yl], [xs[-1], yl]]), al, (120, 120, 140), 3)
    for (m, tm), x in zip(marks, xs):
        on = t >= tm; r = 13 if on else 8
        cv2.circle(fr, (int(x), yl), r, tuple(float(c) / 255 for c in (PINK if on else (120, 120, 140))), -1, cv2.LINE_AA)
        label_at(fr, f"{m} мес" if m < 12 else "год", x, yl + 34, al * (1 if on else 0.55), WHITE if on else (170, 170, 185), 26, "mm")
    if t >= 0: pass
T_INTRO = Pill([("1 МЕСЯЦ", "H"), ("ПОСЛЕ ОПЕРАЦИИ", "P")], 96)
T_RED = Pop([("КРАСНОТА —", "H"), ("НОРМАЛЬНЫЙ ЭТАП", "P")], 92)
T_POST = [timg(l, "S", 0) for l in ("в этом посте вы сможете наглядно увидеть",)]
T_POST_H = Title([("ПРИМЕР ЗАЖИВЛЕНИЯ", "H"), ("РУБЦОВ НА ГРУДИ", "P")], 88)
def post_freeze(fr_still, t, t0):
    """freeze frame: the whole frame blurs and dims, caption + arrow to an empty slot for the post (added later in Edits)"""
    k = ease((t - t0) / 0.5)
    fr = fr_still * (1 - k) + cv2.GaussianBlur(fr_still, (0, 0), 22) * 0.45 * k
    if t > t0 + 0.3:
        al = min(1, (t - t0 - 0.3) / 0.4); im = T_POST[0]
        ablend(fr, im, 540 - im.shape[1] / 2, 255, al, WHITE, BLUE)
    T_POST_H.draw(fr, 310, t, t0 + 0.45, 1e9)
    pa = ease((t - t0 - 0.9) / 0.5)                                       # arrow down to the slot
    if pa > 0:
        stroke(fr, np.float32([[540, 520], [540, 520 + 110 * pa]]), 1, PINK, 6)
        if pa > 0.9: stroke(fr, np.float32([[515, 605], [540, 632], [565, 605]]), 1, PINK, 6)
    sa = ease((t - t0 - 1.1) / 0.5)                                       # dashed 4:5 slot, left empty for the post
    if sa > 0:
        x0, y0, x1, y1 = 240, 665, 840, 1415; m = np.zeros((H, W), np.uint8)
        for (ax, ay), (bx, by) in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
            L = np.hypot(bx - ax, by - ay); nseg = int(L / 36)
            for q in range(nseg):
                if q % 2 == 0: cv2.line(m, (int(ax + (bx - ax) * q / nseg), int(ay + (by - ay) * q / nseg)), (int(ax + (bx - ax) * (q + 1) / nseg), int(ay + (by - ay) * (q + 1) / nseg)), 255, 4, cv2.LINE_AA)
        a = m.astype(np.float32)[..., None] / 255 * sa * 0.85; fr = fr * (1 - a) + a
    return fr
T_END = Title([("НЕ ОЦЕНИВАЙТЕ РУБЕЦ", "H"), ("СЛИШКОМ РАНО", "P")], 112)

# ---------------- subtitles ----------------
WR = json.load(open("words.json"))["8294"]; SUB = []
for si, so, t0 in SEGS:
    for a, b, w in WR:
        if b > si + 0.1 and a > si - 0.3 and a < so - 0.08:
            w = w.lower().strip(".,!?«»—…").strip()
            if w: SUB.append([t0 + max(0, a - si), t0 + min(b, so) - si, w])
FIX = {"все": "всё"}
SUB = [[a, b, FIX.get(w, w)] for a, b, w in SUB]; SUB.sort()
for i in range(len(SUB)):
    nxt = SUB[i + 1][0] if i + 1 < len(SUB) else DUR
    SUB[i][1] = min(nxt, max(SUB[i][1], SUB[i][0] + 0.3) + 0.25)
KEY = {"первый", "второго", "коллагена", "красными", "нормальным", "третьего", "белеть", "побелеет", "шестому", "году"}
SW_ = {}
def subs2(fr, t, y):
    for t0, t1, w in SUB:
        if t0 <= t < t1:
            if w not in SW_:
                f = mont(300, 52); asc, desc = f.getmetrics(); im = Image.new("RGBA", (int(f.getlength(w)) + 20, asc + desc + 10), (0, 0, 0, 0))
                ImageDraw.Draw(im).text((10, 5 + asc), w, font=f, fill=(255, 255, 255, 255), anchor="ls"); SW_[w] = to_np(im)
            im = SW_[w]; sc = 0.6 + 0.4 * back((t - t0) / 0.28)
            im2 = cv2.resize(im, (max(1, int(im.shape[1] * sc)), max(1, int(im.shape[0] * sc))))
            c = (PINK, ROSE) if w.strip(",") in KEY else (WHITE, BLUE)
            ablend(fr, im2, 540 - im2.shape[1] / 2, y - im2.shape[0] / 2, min(1, (t - t0) / 0.08), *c)

def render_block(t, b=None):
    n, t0, t1 = b or block(t); lt = t - t0; DEFER[0] = True; PENDING.clear(); LIGHT[0] = False
    if n not in ("end", "post"):
        z, cx, cy, oy = SHOTS[n]
        fr = vframe(src_of(t), z * (1 + 0.03 * lt / 6) * trans(t, t0, 0.05), cx, cy, oy)
        if n == "m1": fr = focus(fr, lt, 0.7)
        else: fr = mblur(fr, lt)
        fr = look_tablet(fr, t, z * (1 + 0.03 * lt / 6) * trans(t, t0, 0.05), cx, cy, oy)
        if n == "m1" and t < 1.9: T_INTRO.draw(fr, 1180, t, 0.1, 1.55)
        scar_card(fr, t, 0.2)
        if n == "m2": T_RED.draw(fr, 1190, t, 13.6, 16.9)
        subs2(fr, t, 1440)
    elif n == "post":                                                   # freeze on the last frame of the speech
        z, cx, cy, oy = SHOTS["m6"]; ts = 27.55
        still = look(vframe(src_of(ts), z * (1 + 0.03 * (ts - 21.6) / 6), cx, cy, oy), ts) * (1 - HAZE)
        fr = post_freeze(still, t, t0)
    else:                                                               # end card: the whole timeline + the takeaway
        fr = dark_bg(t); scar_card(fr, t, -10, 620); T_END.draw(fr, 1130, t, t0 + 0.4, 1e9)
    DEFER[0] = False
    for args in PENDING: ablend(fr, *args)
    blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
    return np.clip(fr, 0, 1)
def render(t):
    n, t0, t1 = block(t); i = BL.index((n, t0, t1)); fr = render_block(t)
    if i > 0 and t < t0 + XF:
        a = ease((t - t0) / XF); fr = render_block(t, BL[i - 1]) * (1 - a) + fr * a
    if t > DUR - 0.8: fr = fr * (1 - 0.9 * (t - (DUR - 0.8)) / 0.8)
    return fr

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for ts in sys.argv[1:]:
            cv2.imwrite(f"f/s_{ts}.jpg", cv2.cvtColor((render(float(ts)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        sys.exit()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "video_scar.mp4"], stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        enc.stdin.write((render(i / FPS) * 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
