_b = open("base.py").read()
for a, b in [("DUR = 41.0", "DUR = 44.0"), ("ease((t - t0) / 0.3))\ndef mblur", "ease((t - t0) / 0.6))\ndef mblur"),
             ("if lt > 0.1: return fr", "if lt > 0.18: return fr"), ("(1 - lt / 0.1)", "(1 - lt / 0.18)"),
             ("ti = t0 + k * 0.16; p = ease((t - ti) / 0.45)", "ti = t0 + k * 0.24; p = ease((t - ti) / 0.65)"),
             ("out = 1 - ease((t - t1) / 0.3) if t > t1 else 1.0", "out = 1 - ease((t - t1) / 0.4) if t > t1 else 1.0")]:
    assert _b.count(a) == 1, a; _b = _b.replace(a, b)
exec(_b)
exec(open("fx.py").read())
exec(open("draw.py").read())

# ---------------- voice edit (all from IMG_8296): (src_in, src_out, reel_start) ----------------
SEGS = [(7.10, 14.40, 0.30),     # некоторые пациенты представляют себе страшную картину… только стоя жить
        (14.62, 16.30, 7.75),    # нет, это совершенно не так
        (23.28, 27.62, 9.80),    # жир мы вводим гораздо выше. поэтому сидеть можно с первых же дней
        (27.45, 32.10, 14.45),   # единственное, не заваливаться как в кресле, а сидеть просто ровно
        (32.02, 38.60, 19.35),   # садитесь как пианист… всё хорошо будет с вашим жиром
        (90.88, 97.60, 26.30),   # если вы вдруг проснулись… на спине без подушки… катастрофы не произойдёт
        (127.38, 130.95, 33.40), # это не значит, что всё, теперь результат потерян и всё пропало
        (131.12, 135.45, 37.30)] # это всё не так, берегите свои нервы…
def src_of(t):
    best = None
    for si, so, t0 in SEGS:
        if t0 <= t <= t0 + so - si: return si + t - t0
        if t < t0: return max(0.0, si - (t0 - t)) if best is None else min(best[0] + t - best[1], si)
        best = (so, t0 + so - si)
    return best[0] + t - best[1]
BL = [("hook", 0, 7.7), ("myth", 7.7, 9.75), ("seat", 9.75, 14.4), ("chairs", 14.4, 19.3), ("piano", 19.3, 26.2),
      ("sleep", 26.2, 33.3), ("result", 33.3, 37.2), ("nerves", 37.2, DUR)]
XF = 0.35
def block(t):
    for n, a, b in BL:
        if t < b: return n, a, b
    return BL[-1]
CUTS[:] = [b[1] for b in BL[1:]]
V = Src("v/8296.mp4"); NF = int(V.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
def vframe(ts, z=1.0, cx=540, cy=960, oy=960): return warp(V.get(min(max(int(round(ts * 30)), 0), NF - 1)), z, cx, cy, 540, oy)
WIDE = lambda t, lt, d=0.0: vframe(src_of(t), (1.0 + 0.04 * lt / 6) * trans(t, t - lt, 0.05), 540, 960)
CLOSE = lambda t, lt: vframe(src_of(t), (1.6 - 0.07 * lt / 6) * trans(t, t - lt, 0.06), 572, 760, 1010)

# ---------------- text ----------------
TQ = Pop([("ПОСЛЕ ЛИПОФИЛИНГА ЯГОДИЦ", "H"), ("НЕЛЬЗЯ СИДЕТЬ 3 НЕДЕЛИ?", "P")], 92)
TS1 = Title([("ЖИР ВВОДИМ ВЫШЕ", "P")], 96); TS2 = Title([("СИДЕТЬ МОЖНО", "H"), ("С ПЕРВЫХ ДНЕЙ", "P")], 92)
TP = Pop([("САДИТЕСЬ", "H"), ("КАК ПИАНИСТ", "P")], 118)
TL1 = SlideL([("ПРОСНУЛИСЬ НА СПИНЕ", "H"), ("БЕЗ ПОДУШКИ?", "P")], 84, 70)
TL2 = Pop([("КАТАСТРОФЫ", "H"), ("НЕ ПРОИЗОЙДЁТ", "P")], 104)
TR = Pill([("РЕЗУЛЬТАТ", "H"), ("НЕ ПОТЕРЯН", "P")], 96)
TN = Pop([("БЕРЕГИТЕ НЕРВЫ", "P")], 120)
STAMP = timg("МИФ", "H", 230)
def stamp(fr, t, t0, cx, cy):
    """rose stamp slams down over the question"""
    if t < t0: return
    p = ease((t - t0) / 0.22); sc = 1.9 - 0.9 * p; al = min(1, (t - t0) / 0.12)
    w, h = STAMP.shape[1] + 70, STAMP.shape[0] + 30
    box = Image.new("RGBA", (w, h), (0, 0, 0, 0)); ImageDraw.Draw(box).rounded_rectangle((4, 4, w - 5, h - 5), radius=22, outline=(255, 255, 255, 255), width=10)
    lay = to_np(box); lay[15:15 + STAMP.shape[0], 35:35 + STAMP.shape[1]] = np.maximum(lay[15:15 + STAMP.shape[0], 35:35 + STAMP.shape[1]], STAMP)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 9, sc); big = cv2.warpAffine(lay, M, (int(w * 2), int(h * 2.2)), borderValue=0)
    M2 = np.float32([[1, 0, w / 2], [0, 1, h * 0.6]]); big = cv2.warpAffine(big, M2, (big.shape[1], big.shape[0]))
    ablend(fr, big, cx - big.shape[1] / 2, cy - big.shape[0] / 2, al, PINK, ROSE)

# ---------------- line art ----------------
P_ = lambda t, t0: (lambda a, b: ease((t - t0 - a * SLOW) / (b * SLOW)))
def seat_art(fr, t, t0):
    """side view of a seated body: we sit on the sitting bones, the fat goes higher up the buttock"""
    P = P_(t, t0)
    seat = np.float32([[160, 1160], [930, 1160]]); stroke(fr, seat, P(0, 0.5), SKIN, 6, 0.55)
    body = np.vstack([bez((470, 430), (440, 560), (420, 650), (400, 720)),
                      bez((400, 720), (250, 800), (240, 1050), (420, 1150)),
                      bez((420, 1150), (560, 1158), (760, 1150), (900, 1140))])
    front = np.vstack([bez((640, 430), (650, 600), (660, 820), (690, 960)), bez((690, 960), (760, 1000), (860, 1010), (900, 1030))])
    stroke(fr, body, P(0.1, 1.1), SKIN, 7); stroke(fr, front, P(0.2, 1.0), SKIN, 7, 0.7)
    rng = np.random.RandomState(3); pts = np.c_[365 + 40 * rng.randn(46), 870 + 45 * rng.randn(46)]   # fat injected high on the buttock
    for k, (x, y) in enumerate(pts):
        if t > t0 + (1.1 + 0.02 * k) * SLOW: cv2.circle(fr, (int(x), int(y)), 9, tuple(float(c) / 255 for c in MUS), -1, cv2.LINE_AA)
    if t > t0 + 1.0 * SLOW: cv2.circle(fr, (440, 1150), 15, tuple(float(c) / 255 for c in PINK), -1, cv2.LINE_AA)
    label_at(fr, "жир", 560, 820, P(1.8, 0.4), PINK, 48)
    stroke(fr, np.float32([[550, 830], [440, 860]]), P(1.8, 0.4), WHITE, 3)
    label_at(fr, "седалищные бугры —", 470, 1235, P(1.4, 0.4), WHITE, 36); label_at(fr, "на них мы сидим", 470, 1285, P(1.5, 0.4), WHITE, 36)
    stroke(fr, np.float32([[460, 1210], [445, 1170]]), P(1.4, 0.4), WHITE, 3)
    TS1.draw(fr, 250, t, t0 + 0.2, 1e9); TS2.draw(fr, 1390, t, 12.0, 1e9)
def figure(fr, P, d, hip, sh, head, knee, foot, col):
    stroke(fr, np.float32([hip, sh]), P(d, 0.5), col, 12); stroke(fr, np.float32([hip, knee, foot]), P(d + 0.25, 0.6), col, 12)
    al = P(d + 0.1, 0.3)
    if al > 0: cv2.circle(fr, head, 46, tuple(float(c) / 255 for c in col), 10, cv2.LINE_AA)
def chairs_art(fr, t, t0):
    """ПРАВИЛЬНО: upright like at the piano · НЕЖЕЛАТЕЛЬНО: leaning far back as in an armchair"""
    P = P_(t, t0)
    stroke(fr, np.float32([[200, 1030], [400, 1030]]), P(0, 0.4), SKIN, 8, 0.6)
    for x in (215, 385): stroke(fr, np.float32([[x, 1030], [x, 1180]]), P(0.1, 0.3), SKIN, 8, 0.6)
    figure(fr, P, 0.2, (300, 1010), (300, 740), (300, 676), (440, 1010), (440, 1180), PINK)
    stroke(fr, np.float32([[660, 1040], [870, 1040], [870, 1180]]), P(0.4, 0.5), SKIN, 8, 0.6); stroke(fr, np.float32([[680, 1040], [590, 700]]), P(0.5, 0.5), SKIN, 8, 0.6)
    figure(fr, P, 0.6, (720, 1020), (630, 770), (608, 706), (870, 990), (880, 1180), SKIN)
    label_at(fr, "ПРАВИЛЬНО", 320, 1290, P(0.9, 0.4), PINK, 48, "mm"); label_at(fr, "ровно сидим", 320, 1345, P(1.0, 0.4), WHITE, 34, "mm")
    label_at(fr, "НЕЖЕЛАТЕЛЬНО", 760, 1290, P(1.2, 0.4), WHITE, 48, "mm"); label_at(fr, "сильно заваливаться назад", 760, 1345, P(1.3, 0.4), WHITE, 30, "mm")
    stroke(fr, np.float32([[268, 560], [298, 590], [352, 528]]), P(1.0, 0.3), PINK, 9)
    stroke(fr, np.float32([[730, 530], [790, 590]]), P(1.3, 0.2), SKIN, 9); stroke(fr, np.float32([[790, 530], [730, 590]]), P(1.4, 0.2), SKIN, 9)
KEYS = [20.0, 20.45, 20.9, 21.35, 21.8, 22.25, 22.7]                     # one key goes down per note
def piano_keys(fr, t, t0):
    al = min(1, (t - t0 - 0.2) / 0.4)
    if al <= 0: return
    n, kw, x0, y0, kh = 9, 92, 540 - 9 * 92 / 2, 1110, 230
    for k in range(n):
        down = any(0 <= t - s < 0.32 for i, s in enumerate(KEYS) if i % n == k or (i + 1) % n == k + 1 and False) and k == ([i for i, s in enumerate(KEYS) if 0 <= t - s < 0.32] or [-1])[0] % n
        col = (np.float32(PINK) / 255) if down else np.float32([0.97, 0.96, 0.95])
        x = int(x0 + k * kw); y = y0 + (8 if down else 0)
        sub = fr[y:y + kh, x + 3:x + kw - 3]; sub[:] = sub * (1 - al) + col * al
    for k in range(n - 1):
        if k in (2, 6): continue
        x = int(x0 + (k + 1) * kw - 26); sub = fr[y0:y0 + 140, x:x + 52]; sub[:] = sub * (1 - al) + np.float32([0.08, 0.1, 0.14]) * al
def heart(fr, t, t0, cx=540, cy=830, s=1.0):
    p = ease((t - t0) / 0.9); beat = 1 + 0.06 * max(0, np.sin((t - t0 - 0.9) * 2 * np.pi * 1.1)) ** 8 if t > t0 + 0.9 else 1.0
    tt = np.linspace(0, 2 * np.pi, 160); x = 16 * np.sin(tt) ** 3; y = -(13 * np.cos(tt) - 5 * np.cos(2 * tt) - 2 * np.cos(3 * tt) - np.cos(4 * tt))
    pts = np.c_[cx + x * 12 * s * beat, cy + y * 12 * s * beat].astype(np.float32)
    if p > 0.95: fill(fr, pts, 0.85 * min(1, (t - t0 - 0.9) / 0.4), MUS)
    stroke(fr, pts, p, PINK, 9)

# ---------------- subtitles: thin Montserrat, one word, soft pop ----------------
WR = json.load(open("words.json"))["8296"]; SUB = []
for si, so, t0 in SEGS:
    for a, b, w in WR:
        if b > si + 0.1 and a > si - 0.3 and a < so - 0.08:
            w = w.lower().strip(".,!?«»—…").strip()
            if w: SUB.append([t0 + max(0, a - si), t0 + min(b, so) - si, w])
FIX = {"липофиллинг": "липофилинг", "наседалищной": "на седалищные", "бугры": "бугры", "произойдет": "произойдёт", "все": "всё"}
SUB = [[a, b, FIX.get(w, w)] for a, b, w in SUB]; SUB.sort()
for i in range(len(SUB)):
    nxt = SUB[i + 1][0] if i + 1 < len(SUB) else DUR
    SUB[i][1] = min(nxt, max(SUB[i][1], SUB[i][0] + 0.3) + 0.25)
KEY = {"нет", "выше", "пианист", "катастрофы", "ровно", "нервы", "первых", "дней"}
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
    n, t0, t1 = b or block(t); lt = t - t0; DEFER[0] = True; PENDING.clear(); LIGHT[0] = False; sub_y = 1440; vid = True
    if n == "hook":                                                     # wide: the question on the dark wall above her head
        fr = focus(WIDE(t, lt), lt, 0.7); TQ.draw(fr, 250, t, 0.3, 1e9)
    elif n == "myth":                                                   # "нет, это совершенно не так": punch in + МИФ stamp
        fr = mblur(CLOSE(t, lt), lt); TQ.draw(fr, 250, t, -9, 1e9); stamp(fr, t, 7.85, 540, 560)
    elif n == "seat":
        fr = dark_bg(t); seat_art(fr, t, t0); vid = False; sub_y = 1700
    elif n == "chairs":
        fr = dark_bg(t); chairs_art(fr, t, t0); vid = False; sub_y = 1500
        label_at(fr, "КАК СИДЕТЬ ПОСЛЕ ОПЕРАЦИИ", 540, 300, min(1, lt / 0.5), WHITE, 46, "mm")
    elif n == "piano":                                                  # "садитесь как пианист" + keys pressed one by one
        fr = mblur(WIDE(t, lt), lt); fr = fr * (1 - 0.35 * np.clip((yy_[..., None] - 900) / 500, 0, 1) * 0)
        TP.draw(fr, 220, t, 19.95, 1e9); piano_keys(fr, t, t0)
    elif n == "sleep":
        fr = mblur(CLOSE(t, lt) if t < 30.1 else WIDE(t, t - 30.1), min(lt, abs(t - 30.1) + (0 if t >= 30.1 else 1)))
        if t < 30.1: TL1.draw(fr, 300, t, 26.5, 29.75)
        else: TL2.draw(fr, 230, t, 30.2, 1e9)
    elif n == "result":
        fr = mblur(CLOSE(t, lt), lt); TR.draw(fr, 330, t, 35.0, 1e9)
    else:                                                               # берегите нервы + a beating heart
        fr = mblur(WIDE(t, lt), lt) if t < 39.4 else dark_bg(t)
        if t >= 39.4: heart(fr, t, 39.45); TN.draw(fr, 1160, t, 39.9, 1e9); vid = False; sub_y = 1500
        else: TN.draw(fr, 250, t, 37.9, 39.2)
    if vid: fr = look(fr, t) * (1 - HAZE)
    subs2(fr, t, sub_y)
    DEFER[0] = False
    for args in PENDING: ablend(fr, *args)
    blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
    return np.clip(fr, 0, 1)
def render(t):
    n, t0, t1 = block(t); i = BL.index((n, t0, t1)); fr = render_block(t)
    if i > 0 and t < t0 + XF:
        a = ease((t - t0) / XF); fr = render_block(t, BL[i - 1]) * (1 - a) + fr * a
    if 39.4 <= t < 39.75: a = ease((t - 39.4) / 0.35); fr = render_block(39.39) * (1 - a) + fr * a
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
