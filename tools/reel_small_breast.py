_b = open("base.py").read()
for a, b in [("DUR = 41.0", "DUR = 37.6"), ("ease((t - t0) / 0.3))\ndef mblur", "ease((t - t0) / 0.6))\ndef mblur"),
             ("if lt > 0.1: return fr", "if lt > 0.18: return fr"), ("(1 - lt / 0.1)", "(1 - lt / 0.18)"),
             ("ti = t0 + k * 0.16; p = ease((t - ti) / 0.45)", "ti = t0 + k * 0.24; p = ease((t - ti) / 0.65)"),
             ("out = 1 - ease((t - t1) / 0.3) if t > t1 else 1.0", "out = 1 - ease((t - t1) / 0.4) if t > t1 else 1.0"),
             ("p = ease(lt / 0.45); h = int(img.shape[0]", "p = ease(lt / 0.7); h = int(img.shape[0]"),
             ("p = ease((t - t0) / 0.35); al = min(1, (t - t0) / 0.1)", "p = ease((t - t0) / 0.5); al = min(1, (t - t0) / 0.15)")]:
    assert _b.count(a) == 1, a; _b = _b.replace(a, b)
exec(_b)
exec(open("fx.py").read())

# ---------------- voice edit: (source, src_in, src_out, reel_start) ----------------
SEGS = [("surg", 12.48, 17.45, 0.30),    # она хотела ближе к тройке, но… не очень бросалось всем в глаза  (the reel opens here)
        ("surg", 0.00, 5.14, 5.60),      # мы сегодня сделали увеличение груди анатомическими имплантами 295
        ("surg", 23.65, 35.72, 13.60),   # девушки с маленькой грудью… преимущество. …никогда подтяжки
        ("surg", 37.02, 43.82, 25.90),   # ваше страдание… окупаются с лихвой тем, что у вас не будет птоза
        ("surg", 53.88, 55.36, 33.00)]   # будем с вами делиться результатами
def src_of(t):
    best = None
    for s, si, so, t0 in SEGS:
        if t0 <= t <= t0 + so - si: return si + t - t0
        if t < t0: return max(0.0, si - (t0 - t)) if best is None else min(best[0] + t - best[1], si)
        best = (so, t0 + so - si)
    return best[0] + t - best[1]

BL = [("quote", 0, 1.6), ("qsurg", 1.6, 5.4), ("res", 5.4, 11.0), ("turn", 11.0, 13.6), ("main", 13.6, 20.0),
      ("ill1", 20.0, 22.6), ("ill2", 22.6, 25.7), ("final", 25.7, 32.7), ("collage2", 32.7, DUR)]   # schematics illustrate the surgeon's words          # no 3/4 view anywhere
XF = 0.35
def block(t):
    for n, a, b in BL:
        if t < b: return n, a, b
    return BL[-1]
CUTS[:] = [b[1] for b in BL[1:]]
LIGHT_BLOCKS = ()

V = Src("v/surg.mp4"); NF = int(V.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
HEAD = np.load("head.npy")
def raw(ts):
    i = min(max(int(round(ts * 30)), 0), NF - 1); f = V.get(i)
    hx, hy = HEAD[min(i, len(HEAD) - 1)]
    return blur_ell(f, (hx - 30, hy - 15), (260, 185))                  # patient's face, lying on the couch
def vframe(ts, z=1.0, cx=540, cy=960, oy=960): return warp(raw(ts), z, cx, cy, 540, oy)

COLL = load("dl/collage.jpg")
def warm(img, k=1.0):                                                   # gentle warmth on the result
    f = img.astype(np.float32); g = f.mean(2, keepdims=True); f = g + (f - g) * (1 + 0.06 * k)
    f *= np.float32([1 + 0.07 * k, 1 + 0.025 * k, 1 - 0.09 * k]); return np.clip(f, 0, 255).astype(np.uint8)
COLL[560:1660, 552:] = warm(COLL[560:1660, 552:])                        # ПОСЛЕ half of the collage
DO_IMG = COLL[592:1636, 18:548].copy(); PO_IMG = COLL[592:1636, 556:1078].copy()     # the two halves of the collage
def feather(img, e=110):                                                # side photos melt into the black frame at their edges
    h, w = img.shape[:2]; m = (np.clip(np.arange(w) / e, 0, 1) * np.clip((w - 1 - np.arange(w)) / e, 0, 1))[None] * np.clip((h - 1 - np.arange(h)) / e, 0, 1)[:, None]
    return (img * m[..., None] + np.float32([16, 16, 18]) * (1 - m[..., None])).astype(np.uint8)
SDO = feather(load("side_do.png")); SPO = feather(load("side_po.png")); COLL2 = load("collage_side.jpg")   # side view, built by side.py
def side(img, cx, z=1.0): return warp(img, 0.70 * z, cx, 1080, 540, 860)       # torso aligned between ДО and ПОСЛЕ
def morph(a_, b_, p, f=0.35):
    edge = (xx_ / W * 0.6 + yy_ / H * 0.4)[..., None]
    m = np.clip((p * (1 + f) - edge) / f, 0, 1); out = a_ * (1 - m) + b_ * m
    return out
def sweep(fr, t, t0, dur=0.9, k=0.22):
    """a soft light band glides across the photo (draws the eye to the result)"""
    p = (t - t0) / dur
    if not 0 <= p <= 1: return fr
    pos = -0.3 + 1.6 * ease(p); band = np.exp(-(((xx_ / W) * 0.75 + (yy_ / H) * 0.25 - pos) / 0.06) ** 2)[..., None]
    return fr + (1 - fr) * band * k * np.sin(np.pi * p)
def punch(t, hits, amp=0.16, dur=0.5):                                  # quick zoom-in on a key word, eases back
    z = 1.0
    for h in hits:
        if h <= t < h + dur: z += amp * (1 - ease((t - h) / dur))
    return z
exec(open("illus.py").read())
STRIKE = timg("ПОДТЯЖКА", "H", 130)
def strike(fr, t, t0, ts, t1):
    """ПОДТЯЖКА pops in, then a powder line crosses it out"""
    if t < t0 or t > t1 + 0.4: return
    out = 1 - ease((t - t1) / 0.4) if t > t1 else 1.0; sc = 1 + 0.4 * (1 - ease((t - t0) / 0.4))
    im = cv2.resize(STRIKE, (int(STRIKE.shape[1] * sc), int(STRIKE.shape[0] * sc))); y = 830
    ablend(fr, im, 540 - im.shape[1] / 2, y - im.shape[0] / 2, out * min(1, (t - t0) / 0.2), WHITE, BLUE)
    p = ease((t - ts) / 0.35)
    if p > 0:
        w = STRIKE.shape[1] + 60; bar = np.zeros((22, max(1, int(w * p)), 4), np.float32); bar[..., 3] = 1
        M = cv2.getRotationMatrix2D((0, 7), -6, 1.0); big = np.zeros((90, w + 10, 4), np.float32); big[34:56, :bar.shape[1]] = bar
        big = cv2.warpAffine(big, cv2.getRotationMatrix2D((w / 2, 45), 6, 1.0), (big.shape[1], big.shape[0]))
        ablend(fr, big, 540 - w / 2, y - 45 + 6, out, PINK, ROSE)
def photo_full(img, z=1.0, cy_src=None):
    s = 1640 / img.shape[0] * z; cy = img.shape[0] * 0.5 if cy_src is None else cy_src      # whole body fits, black sides
    return warp(img, s, img.shape[1] / 2, cy, 540, 1000)

_RM = {}
def card_video(fr, img, s, cx, cy, r=40):
    w, h = int(W * s), int(H * s)
    if w < 8: return
    im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    if (w, h) not in _RM:
        m = np.zeros((h, w), np.uint8); rr = int(r * min(1, s * 1.6)); cv2.rectangle(m, (rr, 0), (w - rr, h), 255, -1); cv2.rectangle(m, (0, rr), (w, h - rr), 255, -1)
        for x, y in ((rr, rr), (w - rr, rr), (rr, h - rr), (w - rr, h - rr)): cv2.circle(m, (x, y), rr, 255, -1, cv2.LINE_AA)
        _RM[(w, h)] = m.astype(np.float32)[..., None] / 255
    m = _RM[(w, h)]; rgba = np.concatenate([im * m, m], -1).astype(np.float32)
    if s < 0.98: blend(fr, shadow_of(rgba, 30, 0.30), cx - w / 2, cy - h / 2 + 22, 1.0)
    blend(fr, rgba, cx - w / 2, cy - h / 2, 1.0)

# ---------------- text ----------------
CH_DO, CH_PO = label("ДО", WHITE), label("ПОСЛЕ", PINK)
T1 = Pop([("ПОЧТИ НЕ БЫЛО", "H"), ("СВОЕГО ОБЪЁМА", "P")], 104); T1s = timg("но запрос был ближе к тройке", "S", 0)
T2 = Pill([("«ХОЧУ БЛИЖЕ К ТРОЙКЕ,", "H"), ("НО ЧТОБЫ НЕ БРОСАЛОСЬ", "P"), ("В ГЛАЗА»", "P")], 70)
T3 = Title([("АНАТОМИЧЕСКИЕ ИМПЛАНТЫ", "H")], 78); T3s = timg("установка под мышцу", "S", 0)
T4 = Title([("А ТЕПЕРЬ ВАЖНЫЙ МОМЕНТ", "H"), ("ДЛЯ ДЕВУШЕК С ОЧЕНЬ", "P"), ("МАЛЕНЬКОЙ ГРУДЬЮ", "P")], 88)
T5 = Pop([("У МАЛЕНЬКОГО ОБЪЁМА", "H"), ("ЕСТЬ СВОЁ", "H"), ("ПРЕИМУЩЕСТВО", "P")], 104)
T6 = SlideL([("ИНОГДА ТО, ЧТО КАЗАЛОСЬ", "H"), ("НЕДОСТАТКОМ,", "P"), ("СТАНОВИТСЯ", "H"), ("ПРЕИМУЩЕСТВОМ", "P")], 80)
T7 = timg("будем делиться результатом дальше", "S", 0)
NUM = {}
def counter(fr, t, t0, t1, val, y):
    """295 МЛ counts up and lands when the surgeon says it"""
    if t < t0: return
    v = int(round(val * ease((t - t0) / (t1 - t0)))); txt = f"{v} МЛ"
    if txt not in NUM: NUM[txt] = timg(txt, "P", 150)
    im = NUM[txt]; sc = 1 + 0.18 * max(0, 1 - abs(t - t1) / 0.25)        # small punch when it lands
    im2 = cv2.resize(im, (int(im.shape[1] * sc), int(im.shape[0] * sc)))
    ablend(fr, im2, 540 - im2.shape[1] / 2, y - im2.shape[0] / 2, min(1, (t - t0) / 0.2), PINK, ROSE)

# ---------------- animated subtitles ----------------
WR = json.load(open("words.json")); SUB = []
for s, si, so, t0 in SEGS:
    for a, b, w in WR[s]:
        if b > si + 0.1 and a > si - 0.3 and a < so - 0.08:
            w = w.lower().strip(".,!?«»—…").strip()
            if w: SUB.append([t0 + max(0, a - si), t0 + min(b, so) - si, w])
FIX = {"объема": "объёма", "подмышцу": "под мышцу", "лихой": "лихвой", "ее": "её", "еще": "ещё", "295": "295"}
SUB = [[a, b, FIX.get(w, w)] for a, b, w in SUB]; SUB.sort()
for i in range(len(SUB)):
    nxt = SUB[i + 1][0] if i + 1 < len(SUB) else DUR
    SUB[i][1] = min(nxt, max(SUB[i][1], SUB[i][0] + 0.3) + 0.25)
KEY = {"объёма", "тройке", "анатомическими", "имплантами", "295", "маленькой", "преимущество", "никогда", "опустится", "подтяжки", "окупаются", "птоза", "лихвой"}
SW_ = {}
def subs2(fr, t, y):
    for t0, t1, w in SUB:
        if t0 <= t < t1:
            if w not in SW_:
                f = mont(600, 50); asc, desc = f.getmetrics(); im = Image.new("RGBA", (int(f.getlength(w)) + 20, asc + desc + 10), (0, 0, 0, 0))
                ImageDraw.Draw(im).text((10, 5 + asc), w, font=f, fill=(255, 255, 255, 255), anchor="ls"); SW_[w] = to_np(im)
            im = SW_[w]; sc = 0.55 + 0.45 * back((t - t0) / 0.28)
            im2 = cv2.resize(im, (max(1, int(im.shape[1] * sc)), max(1, int(im.shape[0] * sc))))
            c = (PINK, ROSE) if w.strip(",") in KEY else (WHITE, BLUE)
            ablend(fr, im2, 540 - im2.shape[1] / 2, y - im2.shape[0] / 2, min(1, (t - t0) / 0.08), *c)

def render_block(t, b=None):
    n, t0, t1 = b or block(t); lt = t - t0; flash = 0; DEFER[0] = True; PENDING.clear(); LIGHT[0] = n in LIGHT_BLOCKS
    sub_y = 1400
    if n == "do":                                                       # ДО: focus pull + viewfinder on the chest
        fr = focus(photo_full(DO_IMG, 1.0 + 0.05 * lt / 4), lt); chip(fr, CH_DO, 60, 228, t, 0.4)
        brackets(fr, 560, 640, 470, 280, t, 0.3, 2.4)
        T1.draw(fr, 1060, t, 1.0, t1 - 0.3)
        if t > 1.9: ablend(fr, T1s, 540 - T1s.shape[1] / 2, 1290, min(1, (t - 1.9) / 0.4) * (1 - ease((t - t1 + 0.3) / 0.35)), WHITE, BLUE)
    elif n == "quote":                                                  # opens straight on her request: quote in a frame over ДО
        fr = focus(side(SDO, 645, 0.97 + 0.02 * lt / 1.6), lt, 0.7); chip(fr, CH_DO, 60, 228, t, 0.3)
        brackets(fr, 671, 603, 300, 260, t, 0.2, 1.5)                    # viewfinder on the chest
        T2.draw(fr, 1180, t, 0.35, 5.05)
    elif n == "qsurg":                                                  # from "тем не менее": the surgeon, close
        fr = mblur(vframe(src_of(t), 1.5 * (1 - 0.05 * lt / 3.8) * trans(t, t0), 520, 560, 820), lt)
        T2.draw(fr, 1180, t, 0.35, 5.05)
    elif n == "res":                                                    # first result: profile ДО -> ПОСЛЕ, 295 ml counts up
        z = (0.97 + 0.03 * lt / 5.6) * trans(t, t0, 0.03); p = ease((t - 6.4) / 1.6)       # ДО -> ПОСЛЕ in profile while she says "сделали увеличение"
        fr = mblur(morph(side(SDO, 645, z), side(SPO, 702, z), p), lt)
        chip(fr, CH_DO, 60, 228, t, t0 + 0.1, 6.4); chip(fr, CH_PO, 60, 228, t, 7.4)
        T3.draw(fr, 1060, t, 7.6, t1 - 0.3)
        counter(fr, t, 8.4, 9.45, 295, 1205); implant_icon(fr, t, 7.7)
        if t > 9.8: ablend(fr, T3s, 540 - T3s.shape[1] / 2, 1300, min(1, (t - 13.8) / 0.4) * (1 - ease((t - t1 + 0.3) / 0.35)), WHITE, BLUE)
    elif n == "turn":                                                   # beautiful after-shot + the turn of the story
        fr = mblur(side(SPO, 760, (1.18 + 0.04 * lt / 2.8) * trans(t, t0, 0.03)), lt); chip(fr, CH_PO, 60, 228, t, t0 + 0.15)   # profile, closer
        T4.draw(fr, 1000, t, t0 + 0.25, t1 - 0.3)
    elif n == "main":                                                   # close on the surgeon: the main thought
        z = (1.5 - 0.12 * ease(lt / 8)) * punch(t, [17.55, 20.3])       # punch-ins on "преимущество" and "никогда"
        fr = mblur(vframe(src_of(t), z * trans(t, t0), 520, 560, 820), lt)
        T5.draw(fr, 1000, t, 17.4, t1 - 0.35)
    elif n == "wide":                                                   # wide: surgeon and the patient, slow push to the chest
        fr = mblur(vframe(src_of(t), (1.0 + 0.12 * ease(lt / 4.1)) * trans(t, t0), 500, 1000, 960), lt)
    elif n == "ill1":                                                   # "никогда грудь не опустится": implant under the muscle
        fr = dark_bg(t); illus_muscle(fr, t, t0)
    elif n == "ill2":                                                   # "не нужно будет делать подтяжки": ptosis vs no ptosis
        fr = dark_bg(t); illus_ptosis(fr, t, t0)
    elif n == "final":                                                  # final look at the result
        fr = mblur(side(SPO, 702, (0.97 + 0.04 * lt / 7) * trans(t, t0, 0.03)), lt); chip(fr, CH_PO, 60, 228, t, t0 + 0.15)   # the favourable profile
        T6.draw(fr, 1000, t, 26.3, t1 - 0.35)
    else:                                                               # the two collages
        z = 1.04 - 0.04 * ease(lt / 1.2)
        fr = mblur(warp(COLL2, z, 540, 1110, 540, 1110), lt); chip(fr, CH_DO, 40, 610, t, t0 + 0.5); chip(fr, CH_PO, 580, 610, t, t0 + 0.7)
        if t > t0 + 1.0: ablend(fr, T7, 540 - T7.shape[1] / 2, 370, min(1, (t - t0 - 1.0) / 0.5), WHITE, BLUE)
    if not LIGHT[0] and not n.startswith("collage"): fr = look(fr, t) * (1 - HAZE)
    if not n.startswith("collage"): subs2(fr, t, sub_y)
    DEFER[0] = False
    for args in PENDING: ablend(fr, *args)
    blend(fr, LOGO_D if LIGHT[0] else LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
    if flash > 0: fr = fr * (1 - flash) + flash
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
            cv2.imwrite(f"f/o_{ts}.jpg", cv2.cvtColor((render(float(ts)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        sys.exit()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "video.mp4"], stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        enc.stdin.write((render(i / FPS) * 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
