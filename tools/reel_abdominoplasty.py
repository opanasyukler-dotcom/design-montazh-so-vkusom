_b = open("base.py").read()
for a, b in [("DUR = 41.0", "DUR = 50.6"), ("ease((t - t0) / 0.3))\ndef mblur", "ease((t - t0) / 0.6))\ndef mblur"),
             ("if lt > 0.1: return fr", "if lt > 0.18: return fr"), ("(1 - lt / 0.1)", "(1 - lt / 0.18)"),
             ("ti = t0 + k * 0.16; p = ease((t - ti) / 0.45)", "ti = t0 + k * 0.24; p = ease((t - ti) / 0.65)"),
             ("out = 1 - ease((t - t1) / 0.3) if t > t1 else 1.0", "out = 1 - ease((t - t1) / 0.4) if t > t1 else 1.0"),
             ("p = ease(lt / 0.45); h = int(img.shape[0]", "p = ease(lt / 0.7); h = int(img.shape[0]"),
             ("p = ease((t - t0) / 0.35); al = min(1, (t - t0) / 0.1)", "p = ease((t - t0) / 0.5); al = min(1, (t - t0) / 0.15)")]:
    assert _b.count(a) == 1, a; _b = _b.replace(a, b)
exec(_b)

# ---------------- voice edit: (source, src_in, src_out, reel_start) ----------------
SEGS = [("iv", 3.10, 5.64, 0.40), ("iv", 7.28, 11.66, 3.20),                  # я очень рада… что сделала эту операцию
        ("surg", 3.58, 6.00, 8.20), ("surg", 9.44, 10.95, 10.80),              # четверо детишек, одни близняшки… растянут живот
        ("surg", 12.18, 15.80, 12.85),                                          # два месяца назад абдоминопластику с липосакцией
        ("surg", 24.55, 26.40, 16.75), ("surg", 27.05, 31.22, 18.75),          # фигурка отточенная, хорошая. фартука нет, диастаз ушили
        ("surg", 47.25, 48.70, 23.25),                                          # шикарная талия
        ("iv", 12.06, 16.70, 25.00),                                            # шестого числа началась новая жизнь, модельная
        ("iv", 19.92, 24.62, 30.20), ("iv", 26.10, 30.40, 35.30)]              # надо было просто давно прийти… жить здесь и сейчас
def src_of(t, name):
    best = None                                                         # in the gaps the picture keeps playing (no freeze)
    for s, si, so, t0 in SEGS:
        if s != name: continue
        if t0 <= t <= t0 + so - si: return si + t - t0
        if t < t0: return max(0.0, si - (t0 - t)) if best is None else min(best[0] + t - best[1], si)
        best = (so, t0 + so - si)
    return None if best is None else best[0] + t - best[1]

BL = [("open", 0, 3.2), ("iv1", 3.2, 8.0), ("do", 8.0, 12.6), ("morph", 12.6, 17.0), ("zones", 17.0, 20.2),
      ("belly", 20.2, 23.0), ("waist", 23.0, 24.8), ("life", 24.8, 30.0), ("circ", 30.0, 35.1), ("here", 35.1, 40.0),
      ("col1", 40.0, 41.8), ("col2", 41.8, 43.6), ("col3", 43.6, 45.4), ("col4", 45.4, 47.2), ("cta", 47.2, DUR)]
XF = 0.35
def block(t):
    for n, a, b in BL:
        if t < b: return n, a, b
    return BL[-1]
CUTS[:] = [b[1] for b in BL[1:]]

# ---------------- sources ----------------
V = {k: Src(f"v/{k}.mp4") for k in ["iv", "surg", "walk"]}
NF = {k: int(V[k].cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2 for k in V}
def vframe(k, ts, z=1.0, cx=540, cy=960, ox=540, oy=960):
    f = V[k].get(min(int(round(ts * 30)), NF[k] - 1))
    if k == "walk":                                                     # chin/mouth peeks in at the top edge: soft blur band
        top = f[:420].astype(np.float32); bl = cv2.GaussianBlur(cv2.resize(top, None, fx=0.1, fy=0.1), (0, 0), 3); bl = cv2.resize(bl, (W, 420))
        m = np.clip((330 - np.arange(420)) / 120, 0, 1)[:, None, None]; f = f.copy(); f[:420] = (top * (1 - m) + bl * m).astype(np.uint8)
    return warp(f, z, cx, cy, ox, oy)
def vsync(name, t, z=1.0, cx=540, cy=960, oy=960): return vframe(name, src_of(t, name), z, cx, cy, 540, oy)

def photo(n):
    im = load(f"dl/IMG_{n}.JPG"); im[3300:, 1500:] = im[3250:3260, 1500:].mean((0, 1)).astype(np.uint8)   # hide the app watermark
    return im
BF = {k: photo(n) for k, n in [("f", 7728), ("q", 7729), ("s", 7730), ("b", 7731)]}
AF = {k: photo(n) for k, n in [("f", 7732), ("q", 7733), ("s", 7734), ("b", 7735)]}
AF["f"] = blur_ell(AF["f"], (1040, 900), (260, 210)); AF["q"] = blur_ell(AF["q"], (1090, 1000), (240, 270)); AF["s"] = blur_ell(AF["s"], (1180, 990), (200, 240))
# body anchors: (shoulder y, crotch y, centre x) in source pixels
ANC = {("B", "f"): (900, 2120, 1088), ("B", "q"): (960, 2230, 1050), ("B", "b"): (960, 2250, 1010),
       ("A", "f"): (1130, 2140, 1025), ("A", "q"): (1278, 2200, 1080)}
def pv(img, key, z=1.0, Y0=430, Y1=1120):
    top, bot, cx = ANC[key]; s = (Y1 - Y0) / (bot - top) * z
    return warp(img, s, cx, (top + bot) / 2, 540, (Y0 + Y1) / 2)
COL = [load(f"dl/IMG_{n}.PNG") for n in (7736, 7737, 7738, 7739)]

# ---------------- text ----------------
T0 = Title([("НАДО БЫЛО ПРОСТО", "H"), ("ДАВНО ПРИЙТИ", "P"), ("2 месяца после операции", "S")], 96)
T1 = Title([("4 ДЕТЕЙ", "P"), ("ОДНИ РОДЫ С БЛИЗНЯШКАМИ", "H")], 92)
T2 = Title([("АБДОМИНОПЛАСТИКА", "P"), ("+ ЛИПОСАКЦИЯ РУК", "H"), ("+ ЛИПОСАКЦИЯ СПИНЫ", "H")], 84)
T3 = Title([("УШИЛИ ДИАСТАЗ", "P"), ("рубцу сейчас только 2 месяца", "S")], 104)
T4 = Title([("НАЧАЛАСЬ", "H"), ("НОВАЯ ЖИЗНЬ", "P")], 104)
T5 = Title([("«ОКАЗЫВАЕТСЯ, ВЕСЬ СЕКРЕТ —", "H"), ("ПРОСТО ПРИЙТИ»", "P")], 80)
T6 = Title([("НАДО ЖИТЬ", "H"), ("ЗДЕСЬ И СЕЙЧАС", "P")], 110)
TC = Title([("ЗАПИСЬ НА КОНСУЛЬТАЦИЮ", "H"), ("ПИШИТЕ «КОНСУЛЬТАЦИЯ»", "P"), ("в Direct", "S")], 92)
CH_DO, CH_PO = label("ДО", WHITE), label("ПОСЛЕ", PINK)

ZL = {}
def zone(fr, txt, t, t0, pt, lx, ly, side):
    """straight arrow from a label to a body zone, drawn in"""
    if t < t0: return
    p = ease((t - t0) / 0.6)
    sx, sy = (lx + (12 if side < 0 else -12), ly)
    ex, ey = sx + (pt[0] - sx) * p, sy + (pt[1] - sy) * p
    m = np.zeros((H, W), np.uint8); cv2.line(m, (int(sx), int(sy)), (int(ex), int(ey)), 255, 4, cv2.LINE_AA)
    if p > 0.85:
        ang = np.arctan2(pt[1] - sy, pt[0] - sx); L = 26
        for da in (2.6, -2.6):
            cv2.line(m, (int(ex), int(ey)), (int(ex + L * np.cos(ang + da)), int(ey + L * np.sin(ang + da))), 255, 4, cv2.LINE_AA)
    a = m.astype(np.float32)[..., None] / 255; fr *= 1 - a; fr += a
    if txt not in ZL:
        f = osw(68); asc, desc = f.getmetrics(); tw = int(f.getlength(txt))
        im = Image.new("RGBA", (tw + 8, asc + desc + 4), (0, 0, 0, 0)); ImageDraw.Draw(im).text((4, 2 + asc), txt, font=f, fill=WHITE + (255,), anchor="ls"); ZL[txt] = to_np(im)
    im = ZL[txt]; al = min(1, (t - t0) / 0.3)
    x = lx - im.shape[1] if side < 0 else lx; ablend(fr, im, x + 30 * (1 - ease((t - t0) / 0.5)) * side, ly - im.shape[0] / 2 - 4, al, WHITE, BLUE)

# ---------------- subtitles ----------------
WR = json.load(open("words.json")); SUB = []
for s, si, so, t0 in SEGS:
    toks = []
    for a, b, w in WR[s]:
        if w.startswith("-") and toks: toks[-1][2] += w; continue
        toks.append([a, b, w])
    for a, b, w in toks:
        if b > si + 0.1 and a > si - 0.3 and a < so - 0.08:
            w = w.lower().strip(".,!?«»—…").strip()
            if w: SUB.append([t0 + max(0, a - si), t0 + min(b, so) - si, w])
FIX = {"пришел": "причём", "блестяшки": "близняшки", "обдоменную": "абдомино-", "пластику": "пластику", "отточенная": "точёная", "фартуга": "фартука",
       "сушили": "ушили", "тесла": "числа", "6": "шестого", "все": "всё"}
SUB = [[a, b, FIX.get(w, w)] for a, b, w in SUB]; SUB.sort()
for i in range(len(SUB)):
    nxt = SUB[i + 1][0] if i + 1 < len(SUB) else DUR
    SUB[i][1] = min(nxt, max(SUB[i][1], SUB[i][0] + 0.3) + 0.25)

def face_disc_iv(t, r):
    f = V["iv"].get(int(round(src_of(t, "iv") * 30))); half = 430
    crop = f[int(560 - half):int(560 + half), int(540 - half):int(540 + half)]
    return cv2.resize(crop, (2 * int(r), 2 * int(r)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
def circle_iv(fr, t, t0, cx, cy, r):
    lt = t - t0; s = max(back(lt / 0.6), 0.02); rr = int(r * s)
    if rr < 4: return
    yo = 6 * np.sin(t * 1.8)
    put_disc(fr, face_disc_iv(t, rr), cx, cy + yo); ring(fr, cx, cy + yo, rr + 14, ease((lt - 0.2) / 0.9))
    pul = (lt * 0.7) % 1
    ring(fr, cx, cy, rr + 24 + 60 * pul, 1.0, col=tuple(int(c) for c in np.float32(PINK) * (1 - pul) + np.float32([250, 247, 246]) * pul), th=4)

def morph(a_, b_, p, f=0.35):
    edge = (xx_ / W * 0.6 + yy_ / H * 0.4)[..., None]
    m = np.clip((p * (1 + f) - edge) / f, 0, 1); return a_ * (1 - m) + b_ * m

LIGHT_BLOCKS = ("circ", "cta")
def render_block(t, b=None):
    n, t0, t1 = b or block(t); lt = t - t0; flash = 0; DEFER[0] = True; PENDING.clear(); LIGHT[0] = n in LIGHT_BLOCKS
    if n == "open":                                                     # ДО, slow push
        fr = pv(BF["f"], ("B", "f"), 1.0 + 0.05 * lt / 3.2); chip(fr, CH_DO, 60, 228, t, 0.2)
    elif n == "iv1":
        fr = mblur(vsync("iv", t, 1.12 * (1 + 0.03 * lt / 4.8) * trans(t, t0), 540, 900, 900), lt)
    elif n == "do":                                                     # ДО: front -> 3/4 -> back, soft dissolves
        seq = [("f", 8.0), ("q", 9.5), ("b", 11.0)]; fr = None
        for k, (key, s0) in enumerate(seq):
            if t < s0: break
            img = pv(BF[key], ("B", key), 1.0 + 0.04 * (t - s0))
            a = ease((t - s0) / 0.45) if k else 1.0; fr = img if fr is None else fr * (1 - a) + img * a
        chip(fr, CH_DO, 60, 228, t, t0 + 0.1); T1.draw(fr, 1210, t, t0 + 0.3, t1 - 0.05)
    elif n == "morph":                                                  # smooth ДО -> ПОСЛЕ (shoulders + crotch aligned)
        z = 1.0 + 0.04 * lt / 4.4
        fr = morph(pv(BF["f"], ("B", "f"), z), pv(AF["f"], ("A", "f"), z), ease((t - 13.1) / 1.6))
        chip(fr, CH_DO, 60, 228, t, t0, 13.1); chip(fr, CH_PO, 60, 228, t, 14.0)
        T2.draw(fr, 1190, t, 14.2, t1 - 0.05)
    elif n == "zones":                                                  # where the figure changed
        fr = pv(AF["q"], ("A", "q"), 1.0 + 0.03 * lt / 3.2); chip(fr, CH_PO, 60, 228, t, t0 + 0.1)
        z = 1.0 + 0.03 * lt / 3.2; P = lambda x, y: (540 + (x - 1080) * 690 / 922 * z, 775 + (y - 1739) * 690 / 922 * z)
        zone(fr, "РУКИ", t, t0 + 0.3, P(700, 1450), 200, 560, -1)
        zone(fr, "ТАЛИЯ", t, t0 + 0.75, P(1370, 1760), 870, 640, 1)
        zone(fr, "ЖИВОТ", t, t0 + 1.2, P(1080, 1900), 850, 1010, 1)
        zone(fr, "СПИНА", t, t0 + 1.65, P(740, 1800), 200, 940, -1)
    elif n == "belly":                                                  # close-up: belly, scar
        fr = mblur(vframe("surg", 35.9 + lt * 0.85, 1.05 * (1 + 0.03 * lt / 2.8) * trans(t, t0), 540, 820, 540, 800), lt)
        chip(fr, CH_PO, 60, 228, t, t0 + 0.1); T3.draw(fr, 1210, t, t0 + 0.35, t1 - 0.05)
    elif n == "waist":
        fr = mblur(vsync("surg", t, 1.1 * trans(t, t0), 540, 980, 900), lt); chip(fr, CH_PO, 60, 228, t, t0 + 0.1)
    elif n == "life":                                                   # the figure now: she turns
        fr = mblur(vframe("walk", 0.2 + lt, 1.0 * trans(t, t0), 540, 900, 540, 900), lt); chip(fr, CH_PO, 60, 228, t, t0 + 0.1)
        T4.draw(fr, 1230, t, 26.7, t1 - 0.05)
    elif n == "circ":
        fr = white_bg(t); circle_iv(fr, t, t0, 540, 720, 300)
        T5.draw(fr, 1150, t, t0 + 0.5, t1 - 0.05)
    elif n == "here":
        fr = mblur(vframe("walk", 5.0 + lt * 0.9, 1.0 * trans(t, t0), 540, 900, 540, 900), lt); chip(fr, CH_PO, 60, 228, t, t0 + 0.1)
        T6.draw(fr, 1230, t, 37.5, t1 - 0.05)
    elif n.startswith("col"):                                           # ready ДО/ПОСЛЕ collages: ПОСЛЕ top, ДО bottom
        img = COL[int(n[3]) - 1]; z = (1 + 0.02 * lt) * trans(t, t0, 0.04)
        fr = mblur(warp(img, W / img.shape[1] * z, img.shape[1] / 2, img.shape[0] / 2), lt)
        chip(fr, CH_PO, 40, 228, t, t0 + 0.3); chip(fr, CH_DO, 40, 990, t, t0 + 0.5)
    else:                                                               # CTA on white
        fr = white_bg(t); TC.draw(fr, 760, t, t0 + 0.15, 1e9)
    if n in ("open", "iv1"): T0.draw(fr, 1180, t, 0.3, 7.7)
    dark = not LIGHT[0] and not n.startswith("col")
    if not LIGHT[0]: fr = look(fr, t)
    if dark: fr = fr * (1 - HAZE)
    if not n.startswith("col") and n != "cta": subs(fr, t)
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
