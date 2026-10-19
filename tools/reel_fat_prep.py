_b = open("base.py").read()
for a, b in [("DUR = 41.0", "DUR = 40.4"), ("ease((t - t0) / 0.3))\ndef mblur", "ease((t - t0) / 0.6))\ndef mblur"),
             ("if lt > 0.1: return fr", "if lt > 0.18: return fr"), ("(1 - lt / 0.1)", "(1 - lt / 0.18)"),
             ("ti = t0 + k * 0.16; p = ease((t - ti) / 0.45)", "ti = t0 + k * 0.24; p = ease((t - ti) / 0.65)"),
             ("out = 1 - ease((t - t1) / 0.3) if t > t1 else 1.0", "out = 1 - ease((t - t1) / 0.4) if t > t1 else 1.0")]:
    assert _b.count(a) == 1, a; _b = _b.replace(a, b)
exec(_b)
exec(open("fx.py").read())

# ---------------- voice edit: (source, src_in, src_out, reel_start) ----------------
SEGS = [("surg", 1.42, 7.60, 0.30),      # ну вот, собрали мы жир, перекачали его в специальные шприцы большие
        ("surg", 7.62, 15.36, 6.75),     # дальше будем в десятки шприцы… сколько куда ввели
        ("surg", 15.38, 21.92, 14.80),   # у нас жир хороший, чистый… тогда ещё его и промываем
        ("surg", 24.55, 29.98, 21.45),   # из баночек мы забираем шприцы… чтобы он был чистый
        ("surg", 30.05, 35.36, 27.10),   # и дальше вводим в каждую железу по миллилитрам… в обеих железах
        ("surg", 35.38, 38.34, 32.65)]   # иногда разным вводим, когда бывает асимметрия
def src_of(t, name="surg"):
    best = None
    for s, si, so, t0 in SEGS:
        if t0 <= t <= t0 + so - si: return si + t - t0
        if t < t0: return max(0.0, si - (t0 - t)) if best is None else min(best[0] + t - best[1], si)
        best = (so, t0 + so - si)
    return best[0] + t - best[1]

BL = [("hook", 0, 3.4), ("out", 3.4, 6.75), ("ml", 6.75, 10.8), ("close", 10.8, 14.6), ("prep", 14.6, 21.5),
      ("scheme", 21.5, 27.0), ("hands", 27.0, 30.0), ("equal", 30.0, 35.9), ("outro", 35.9, DUR)]
XF = 0.35
def block(t):
    for n, a, b in BL:
        if t < b: return n, a, b
    return BL[-1]
CUTS[:] = [b[1] for b in BL[1:]]
LIGHT_BLOCKS = ("out", "ml", "scheme", "equal")

V = {"surg": Src("v/surg.mp4")}; NF = int(V["surg"].cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
def raw(ts): return V["surg"].get(min(int(round(ts * 30)), NF - 1))
def vframe(ts, z=1.0, cx=540, cy=960, ox=540, oy=960): return warp(raw(ts), z, cx, cy, ox, oy)
def broll(ts, z=1.0, cy=960): return vframe(min(ts, 55.2), z, 540, cy)

# video as a rounded card on the white background
_RM = {}
def card_video(fr, img, s, cx, cy, r=40):
    w, h = int(W * s), int(H * s)
    if w < 8: return
    im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    key = (w, h)
    if key not in _RM:
        m = np.zeros((h, w), np.uint8); rr = int(r * min(1, s * 1.6)); cv2.rectangle(m, (rr, 0), (w - rr, h), 255, -1); cv2.rectangle(m, (0, rr), (w, h - rr), 255, -1)
        for x, y in ((rr, rr), (w - rr, rr), (rr, h - rr), (w - rr, h - rr)): cv2.circle(m, (x, y), rr, 255, -1, cv2.LINE_AA)
        _RM[key] = m.astype(np.float32)[..., None] / 255
    m = _RM[key]; rgba = np.concatenate([im * m, m], -1).astype(np.float32)
    if s < 0.98: blend(fr, shadow_of(rgba, 30, 0.30), cx - w / 2, cy - h / 2 + 22, 1.0)
    blend(fr, rgba, cx - w / 2, cy - h / 2, 1.0)

# ---------------- text ----------------
T0 = Pop([("ЧТО ПРОИСХОДИТ С ЖИРОМ", "H"), ("ПЕРЕД ЛИПОФИЛИНГОМ ГРУДИ?", "P")], 92)
T1 = Pill([("ОБЪЁМ СЧИТАЕМ", "H"), ("В МИЛЛИЛИТРАХ", "P")], 84); T1s = [timg("чтобы точно понимать, сколько ткани", "S", 0), timg("введено в каждую грудь", "S", 0)]
T2 = SlideL([("ПОДГОТОВКА ЖИРА", "H"), ("ЗАВИСИТ", "P"), ("ОТ ЕГО СОСТОЯНИЯ", "H")], 84)
T3 = Pop([("ОДИНАКОВЫЙ ОБЪЁМ", "H"), ("НЕ ВСЕГДА НУЖЕН", "P")], 92)
EX = [load(f"ex/lipo{i}.jpg") for i in (1, 2, 3)]                     # one patient after breast lipofilling: 3/4, side, front
TXL = Title([("ЛИПОФИЛИНГ ГРУДИ", "P"), ("пример моей пациентки", "S")], 80)
EXAMPLES = [(11.0, 14.6, EX, TXL)]                                     # all three angles in one go: click, click, click
CLICK = 1.2                                                             # each angle stays this long
def flat_card(fr, img, cx, cy, w, al, sc=1.0):
    """straight photo card (no tilt), rounded corners + soft shadow"""
    w = int(w * sc); h = int(img.shape[0] * w / img.shape[1])
    im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    m = np.zeros((h, w), np.uint8); r = 26; cv2.rectangle(m, (r, 0), (w - r, h), 255, -1); cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for x, y in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)): cv2.circle(m, (x, y), r, 255, -1, cv2.LINE_AA)
    rgba = np.concatenate([im * (m[..., None] / 255.0), m[..., None] / 255.0], -1).astype(np.float32)
    blend(fr, shadow_of(rgba, 26, 0.4), cx - w / 2, cy - h / 2 + 16, al); blend(fr, rgba, cx - w / 2, cy - h / 2, al)
def example_overlay(fr, t):
    for a, b, img, tx in EXAMPLES:
        if a <= t <= b + 0.35:
            k = min(1, (t - a) / 0.35) * (1 - min(1, max(0, (t - b) / 0.35)))
            bg = cv2.GaussianBlur(fr, (0, 0), 9) * 0.42; fr[:] = fr * (1 - k) + bg * k          # the video behind dims and softens
            i = min(int((t - a) / CLICK), len(img) - 1); lt = t - a - i * CLICK
            sc = 1.0 + 0.06 * (1 - ease(lt / 0.18))                       # snap-in on every click
            flat_card(fr, img[i], 540, 965, 690, 1 - min(1, max(0, (t - b) / 0.35)), sc)
            if lt < 0.08: fr[:] = fr * (1 - 0.25 * (1 - lt / 0.08)) + 0.25 * (1 - lt / 0.08)   # tiny shutter flash
            tx.draw(fr, 205, t, a + 0.25, b)
T4 = Title([("ПРИ АСИММЕТРИИ ОБЪЁМ", "H"), ("МОЖЕТ БЫТЬ РАЗНЫМ", "P"), ("всё рассчитывается индивидуально", "S")], 92)
STEPS = [("СБОР", 21.85), ("ПОДГОТОВКА", 23.0), ("ШПРИЦЫ", 24.3), ("ВВЕДЕНИЕ", 25.8)]
STEP_IM = {txt: timg(txt, "H", 60) for txt, _ in STEPS}

def step_chip(fr, txt, t, t0, cy, hi):
    """navy outline pill pops in, the current step fills with powder"""
    if t < t0: return
    im = STEP_IM[txt]; w, h = im.shape[1] + 80, im.shape[0] + 26
    key = ("st", txt, hi)
    if key not in TC:
        b = Image.new("RGBA", (w + 8, h + 8), (0, 0, 0, 0)); d = ImageDraw.Draw(b)
        if hi: d.rounded_rectangle((3, 3, w + 4, h + 4), radius=h // 2, fill=PINK + (255,))
        d.rounded_rectangle((3, 3, w + 4, h + 4), radius=h // 2, outline=BLUE + (255,), width=4); TC[key] = to_np(b)
    sc = max(back((t - t0) / 0.45), 0.02); bx = TC[key]
    b2 = cv2.resize(bx, (max(1, int(bx.shape[1] * sc)), max(1, int(bx.shape[0] * sc))))
    blend(fr, b2, 540 - b2.shape[1] / 2, cy - b2.shape[0] / 2, min(1, (t - t0) / 0.15))
    if sc > 0.6:
        i2 = cv2.resize(im, (max(1, int(im.shape[1] * sc)), max(1, int(im.shape[0] * sc))))
        ablend(fr, i2, 540 - i2.shape[1] / 2, cy - i2.shape[0] / 2 + 2, min(1, (t - t0 - 0.1) / 0.2), WHITE, BLUE)
def arrow_down(fr, t, t0, y0, y1):
    p = ease((t - t0) / 0.35)
    if p <= 0: return
    m = np.zeros((H, W), np.uint8); ye = int(y0 + (y1 - y0) * p); cv2.line(m, (540, y0), (540, ye), 255, 5, cv2.LINE_AA)
    if p > 0.8: cv2.line(m, (540, ye), (522, ye - 18), 255, 5, cv2.LINE_AA); cv2.line(m, (540, ye), (558, ye - 18), 255, 5, cv2.LINE_AA)
    a = m.astype(np.float32)[..., None] / 255 * 0.9; fr *= 1 - a; fr += np.float32(ROSE) / 255 * a

GL, GR = timg("ЛЕВАЯ", "H", 52), timg("ПРАВАЯ", "H", 52)
def gauge(fr, cx, y0, y1, level, lab, t, t0):
    """a syringe-like tube that fills with fat to `level` (0..1)"""
    if t < t0: return
    al = min(1, (t - t0) / 0.3); w = 120; up = 40 * (1 - ease((t - t0) / 0.5))
    y0, y1 = y0 + up, y1 + up
    m_out = np.zeros((H, W), np.uint8); cv2.rectangle(m_out, (int(cx - w / 2), int(y0)), (int(cx + w / 2), int(y1)), 255, 5, cv2.LINE_AA)
    for k in range(1, 10):                                              # graduation marks
        yy = int(y1 - (y1 - y0) * k / 10); cv2.line(m_out, (int(cx + w / 2 - 28), yy), (int(cx + w / 2), yy), 255, 3, cv2.LINE_AA)
    cv2.rectangle(m_out, (int(cx - 14), int(y0 - 70)), (int(cx + 14), int(y0)), 255, 5)    # plunger rod
    m_f = np.zeros((H, W), np.uint8); top = y1 - (y1 - y0) * level
    if level > 0.005: cv2.rectangle(m_f, (int(cx - w / 2 + 6), int(top)), (int(cx + w / 2 - 6), int(y1 - 6)), 255, -1)
    fat = np.float32([232, 132, 74]) / 255
    a = m_f.astype(np.float32)[..., None] / 255 * al * 0.9; fr *= 1 - a; fr += fat * a
    a = m_out.astype(np.float32)[..., None] / 255 * al; fr *= 1 - a; fr += np.float32(BLUE) / 255 * a
    ablend(fr, lab, cx - lab.shape[1] / 2, y1 + 24, al, WHITE, BLUE)

# ---------------- animated subtitles: one word, pops in with a bounce; key words in powder ----------------
WR = json.load(open("words.json")); SUB = []
for s, si, so, t0 in SEGS:
    for a, b, w in WR[s]:
        if b > si + 0.1 and a > si - 0.3 and a < so - 0.08:
            w = w.lower().strip(".,!?«»—…").strip()
            if w: SUB.append([t0 + max(0, a - si), t0 + min(b, so) - si, w])
FIX = {"ассиметрия": "асимметрия", "объем": "объём", "еще": "ещё", "шприц": "шприцы", "чтоб": "чтобы", "обоих": "обеих", "десятки": "в десятки"}
SUB = [[a, b, FIX.get(w, w)] for a, b, w in SUB]; SUB.sort()
for i in range(len(SUB)):
    nxt = SUB[i + 1][0] if i + 1 < len(SUB) else DUR
    SUB[i][1] = min(nxt, max(SUB[i][1], SUB[i][0] + 0.3) + 0.25)
KEY = {"жир", "жир,", "шприцы", "миллилитрам", "чистый", "кровью", "промываем", "асимметрия", "одинаковый", "разным", "железу"}
SW_ = {}
def subs2(fr, t, y):
    for t0, t1, w in SUB:
        if t0 <= t < t1:
            if w not in SW_:
                f = mont(300, 52); asc, desc = f.getmetrics(); im = Image.new("RGBA", (int(f.getlength(w)) + 20, asc + desc + 10), (0, 0, 0, 0))
                ImageDraw.Draw(im).text((10, 5 + asc), w, font=f, fill=(255, 255, 255, 255), anchor="ls"); SW_[w] = to_np(im)
            im = SW_[w]; sc = 0.55 + 0.45 * back((t - t0) / 0.28)
            im2 = cv2.resize(im, (max(1, int(im.shape[1] * sc)), max(1, int(im.shape[0] * sc))))
            c = (PINK, ROSE) if w.strip(",") in KEY else (WHITE, BLUE)
            ablend(fr, im2, 540 - im2.shape[1] / 2, y - im2.shape[0] / 2, min(1, (t - t0) / 0.08), *c)

def render_block(t, b=None):
    n, t0, t1 = b or block(t); lt = t - t0; flash = 0; DEFER[0] = True; PENDING.clear(); LIGHT[0] = n in LIGHT_BLOCKS
    sub_y = 1400                                                        # kept above Instagram's bottom UI
    if n == "hook":                                                     # close-up of the prepared fat, focus pull
        fr = focus(broll(47.9 + lt * 0.9, 1.12 - 0.04 * lt / 3.4), lt)
        T0.draw(fr, 1060, t, 0.35, t1 - 0.25)
    elif n == "out":                                                    # full frame pulls back into a card on white
        fr = white_bg(t); p = ease((t - 3.55) / 1.1); s = 1.0 - 0.44 * p
        card_video(fr, raw(src_of(t)), 1.0 - 0.58 * p, 540, 960 - 320 * p); sub_y = 1400
    elif n == "ml":
        fr = white_bg(t); card_video(fr, raw(src_of(t)), 0.42, 540, 640); sub_y = 1440
        T1.draw(fr, 1170, t, t0 + 0.25, t1 - 0.35)
        for k, im in enumerate(T1s):
            if t > t0 + 1.2 + 0.2 * k: ablend(fr, im, 540 - im.shape[1] / 2, 1305 + 46 * k, min(1, (t - t0 - 1.2 - 0.2 * k) / 0.4) * (1 - ease((t - t1 + 0.35) / 0.35)), WHITE, BLUE)
    elif n == "close":                                                  # punch-in on the syringes on the stand
        z = 2.0 + 0.15 * lt / 3.8
        fr = mblur(vframe(src_of(t), z * trans(t, t0), 240, 1150), lt)
        brackets(fr, 560, 920, 520, 640, t, t0 + 0.3, t1 - 0.6)
    elif n == "prep":                                                   # slow push-in on the surgeon
        z = 1.0 + 0.3 * ease(lt / 6.9)
        fr = mblur(vframe(src_of(t), z * trans(t, t0), 600, 820, 540, 820), lt)
        T2.draw(fr, 980, t, 15.1, t1 - 0.35)
    elif n == "scheme":                                                 # сбор -> подготовка -> шприцы -> введение
        fr = white_bg(t); p = ease(lt / 0.8); card_video(fr, raw(src_of(t)), 0.42 - 0.12 * p, 540, 640 - 140 * p); sub_y = 1420
        for k, (txt, ts) in enumerate(STEPS):
            cur = (k == len(STEPS) - 1 and t >= ts) or (ts <= t < (STEPS[k + 1][1] if k + 1 < len(STEPS) else 1e9))
            step_chip(fr, txt, t, ts, 880 + k * 130, cur)
            if k: arrow_down(fr, t, ts - 0.3, 880 + (k - 1) * 130 + 46, 880 + k * 130 - 46)
    elif n == "hands":                                                  # close frame: hands and syringes
        fr = mblur(vframe(src_of(t), 1.55 * (1 + 0.05 * lt / 3) * trans(t, t0), 520, 1080), lt)
    elif n == "equal":                                                  # same volume in both — or not, with asymmetry
        fr = white_bg(t); card_video(fr, raw(src_of(t)), 0.26, 540, 470); sub_y = 1440
        T3.draw(fr, 740, t, 30.2, t1 - 0.35)
        f1 = ease((t - 30.4) / 1.4); asym = ease((t - 33.4) / 1.2)
        gauge(fr, 380, 1030, 1300, 0.68 * f1 - 0.12 * asym, GL, t, 30.3)
        gauge(fr, 700, 1030, 1300, 0.68 * f1 + 0.14 * asym, GR, t, 30.45)
    else:                                                               # outro: prepared fat, asymmetry note
        fr = broll(51.3 + lt * 0.85, 1.05 + 0.05 * lt / 4.5)
        T4.draw(fr, 1020, t, t0 + 0.3, 1e9)
    if not LIGHT[0]: fr = look(fr, t) * (1 - HAZE)
    example_overlay(fr, t)
    subs2(fr, t, 1650 if any(a <= t <= b + 0.35 for a, b, _, _ in EXAMPLES) else sub_y)   # under the collage card
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
