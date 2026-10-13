_b = open("base.py").read()
for a, b in [("DUR = 41.0", "DUR = 48.6"), ("ease((t - t0) / 0.3))\ndef mblur", "ease((t - t0) / 0.6))\ndef mblur"),
             ("if lt > 0.1: return fr", "if lt > 0.18: return fr"), ("(1 - lt / 0.1)", "(1 - lt / 0.18)"),
             ("ti = t0 + k * 0.16; p = ease((t - ti) / 0.45)", "ti = t0 + k * 0.24; p = ease((t - ti) / 0.65)"),
             ("out = 1 - ease((t - t1) / 0.3) if t > t1 else 1.0", "out = 1 - ease((t - t1) / 0.4) if t > t1 else 1.0"),
             ("p = ease(lt / 0.45); h = int(img.shape[0]", "p = ease(lt / 0.7); h = int(img.shape[0]"),
             ("p = ease((t - t0) / 0.35); al = min(1, (t - t0) / 0.1)", "p = ease((t - t0) / 0.5); al = min(1, (t - t0) / 0.15)")]:
    assert _b.count(a) == 1, a; _b = _b.replace(a, b)
exec(_b)
import json

# ---------------- voice edit: (source, src_in, src_out, reel_start) ----------------
SEGS = [("IMG_3119", 0.00, 3.35, 0.30), ("IMG_3119", 5.25, 8.62, 4.00),            # год назад… не готова к имплантам
        ("IMG_3119", 8.81, 10.90, 7.85),                                              # что-то чужеродное
        ("IMG_3116", 3.70, 5.50, 10.85), ("IMG_3116", 5.69, 9.15, 12.95),            # хирург: делали подтяжку… объёма не хватает
        ("IMG_3119", 18.24, 20.80, 17.45),                                            # из-за объёма решилась на импланты
        ("IMG_3144", 1.81, 2.40, 22.15), ("IMG_3144", 4.03, 6.72, 22.95),           # ну, хорошо. тогда поставим 325 (whole number)
        ("IMG_3138", 5.33, 7.30, 29.05),                                              # как будто и было всегда
        ("IMG_3138", 11.29, 13.69, 31.70), ("IMG_3138", 14.10, 16.00, 34.35),       # тот вид… в своих мечтах
        ("IMG_1360", 2.50, 4.90, 36.85)]                                              # мечты сбываются. это точно
VID = {"IMG_3119": "iv", "IMG_3116": "surg", "IMG_3144": "impl", "IMG_3138": "ward", "IMG_1360": "a60"}
def src_of(t, name):
    best = None                                                         # in the gaps the picture keeps playing (no freeze)
    for s, si, so, t0 in SEGS:
        if s != name: continue
        if t0 <= t <= t0 + so - si: return si + t - t0
        if t < t0: return max(0.0, si - (t0 - t)) if best is None else min(best[0] + t - best[1], si)
        best = (so, t0 + so - si)
    return None if best is None else best[0] + t - best[1]

BL = [("do", 0, 3.9), ("iv1", 3.9, 7.6), ("cards", 7.6, 10.6), ("surg", 10.6, 17.0), ("circ", 17.0, 20.4),
      ("impl", 20.4, 26.6), ("or", 26.6, 28.8), ("after", 28.8, 31.4), ("cmp", 31.4, 36.6), ("dream", 36.6, 39.8),
      ("col1", 39.8, 42.6), ("col2", 42.6, 45.4), ("cta", 45.4, DUR)]
XF = 0.35                                                               # soft dissolve between blocks
def block(t):
    for n, a, b in BL:
        if t < b: return n, a, b
    return BL[-1]
CUTS[:] = [b[1] for b in BL[1:]]

# ---------------- sources ----------------
V = {k: Src(f"v/{k}.mp4") for k in ["do", "a59", "a60", "surg", "iv", "ward", "impl", "or1", "or2", "or3"]}
def gs(a, s):
    r = int(3 * s); k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * s * s)); k /= k.sum()
    pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)])
    return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, pad)
def fill_nan(a):
    a = a.copy()
    for j in range(a.shape[1]):
        v = a[:, j]; ok = ~np.isnan(v); a[:, j] = np.interp(np.arange(len(v)), np.where(ok)[0], v[ok])
    return a
HEADS = {}
for k in ("do", "a59", "a60"):
    P = np.load(f"pose_{k}.npy")
    HEADS[k] = gs(fill_nan(np.c_[P[:, [0, 2, 5, 9, 10], 0].mean(1), P[:, [0, 2, 5, 9, 10], 1].mean(1), np.linalg.norm(P[:, 11, :2] - P[:, 12, :2], axis=1)]), 3)
def vframe(k, ts, z=1.0, cx=540, cy=960, ox=540, oy=960):
    i = int(round(ts * 30)); f = V[k].get(i)
    if k in HEADS:                                                       # patient face blurred
        hx, hy, sw = HEADS[k][min(i, len(HEADS[k]) - 1)]; r = max(sw * 0.4, 80)
        f = blur_ell(f, (hx, hy - 0.2 * r), (r * 0.95, r * 1.3)) if k != "a60" else blur_ell(f, (hx, hy), (r * 1.25, r * 1.6))
    return warp(f, z, cx, cy, ox, oy)
def vsync(name, t, z=1.0, cx=540, cy=960, oy=960): return vframe(VID[name], src_of(t, name), z, cx, cy, 540, oy)

AF = {n: load(p) for n, p in [("f", "dl/IMG_7678.JPG"), ("q", "dl/IMG_7679.JPG"), ("s", "dl/IMG_7680.JPG"), ("bra", "dl/IMG_7681.JPG")]}
BF = {"f": load("IMG_7690.jpg"), "q": load("IMG_7691.jpg")}
for d, k, c, ax in [(AF, "f", (765, 60), (220, 190)), (AF, "q", (960, 190), (160, 180)), (AF, "s", (930, 185), (160, 180)),
                    (AF, "bra", (800, 50), (210, 160)), (BF, "f", (1560, 130), (440, 360)), (BF, "q", (1900, 200), (350, 380))]:
    d[k] = blur_ell(d[k], c, ax)
SHO = {("A", "f"): ((1078, 388), (439, 378)), ("A", "q"): ((904, 614), (623, 626)), ("B", "f"): ((2505, 913), (661, 907)), ("B", "q"): ((1818, 1157), (835, 1170))}
def body_view(img, sh, width_sh=600, y_sh=520, size=(W, H), z=1.0):
    (x1, y1), (x2, y2) = sh; sw = abs(x1 - x2); s = width_sh / sw * z
    return warp(img, s, (x1 + x2) / 2, (y1 + y2) / 2, size[0] / 2, y_sh, size)
def _icon(p):
    im = load(p); d = im.shape[1]; c = im[:d, :d]; m = np.zeros((d, d), np.uint8); cv2.circle(m, (d // 2, d // 2 + 2), d // 2 - 6, 255, -1, cv2.LINE_AA)
    return np.concatenate([c.astype(np.float32) / 255 * (m[..., None] / 255.0), m[..., None] / 255.0], -1).astype(np.float32)
# collage framing: from the neck down to the fingertips (top, bottom, centre x in source pixels)
SPAN = {("A", "f"): (202, 1671, 758), ("B", "f"): (416, 4432, 1583), ("A", "q"): (441, 1895, 900), ("B", "q"): (654, 4796, 1650)}
def span_view(img, top, bot, cx, z=1.0):
    s = 930 / (bot - top) * z; return warp(img, s, cx, (top + bot) / 2, 540, 480, (W, 960))
ICONS = [_icon(f"dl/IMG_769{i}.PNG") for i in (2, 3, 4)]; ICON_LB = ["Липофилинг", "Тест", "Импланты"]

# ---------------- text ----------------
T1 = Title([("ГОД НАЗАД Я НЕ БЫЛА ГОТОВА", "H"), ("К ИМПЛАНТАМ", "P")], 84)
T2 = Title([("ПОЭТОМУ СНАЧАЛА —", "H"), ("ПОДТЯЖКА + ЛИПОФИЛИНГ", "P")], 78)
T3 = Title([("НО ЧЕРЕЗ ГОД", "H"), ("ЗАПРОС ИЗМЕНИЛСЯ", "P")], 84)
T3b = Title([("«ХОЧУ БОЛЬШЕ ОБЪЁМА»", "P")], 104)
T4 = Title([("ИМЕННО ИЗ-ЗА ОБЪЁМА", "H"), ("РЕШИЛАСЬ НА ИМПЛАНТЫ", "P")], 78)
T5 = Title([("ПЛАНИРУЕМ УВЕЛИЧЕНИЕ", "H"), ("КРУГЛЫМИ ИМПЛАНТАМИ", "P")], 76)
T5b = Title([("ВЫБРАЛИ 325 МЛ", "P")], 110)
T7 = Title([("«КАК БУДТО ТАК", "H"), ("И БЫЛО ВСЕГДА»", "P")], 88)
T8 = Title([("ИМЕННО ТОТ ОБЪЁМ,", "H"), ("КОТОРОГО ЕЙ НЕ ХВАТАЛО", "P")], 78)
T9 = Title([("МЕЧТЫ СБЫВАЮТСЯ", "P")], 100)
TC1 = Title([("ЛИПОФИЛИНГ", "H"), ("ИЛИ ИМПЛАНТЫ?", "P"), ("иногда ответ меняется вместе с вашим запросом", "S")], 104)
CH_DO, CH_PO = label("ДО", WHITE), label("ПОСЛЕ", PINK)
def vchip(fr, txt, t, t0, cx, cy, hi=False):
    """volume chip that pops in (navy outline pill, filled when chosen)"""
    if t < t0: return
    key = ("vc", txt, hi)
    if key not in TC:
        f = osw(64); tw = int(f.getlength(txt)); w, h = tw + 64, 92
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        if hi: d.rounded_rectangle((2, 2, w - 3, h - 3), 44, fill=PINK + (255,))
        else: d.rounded_rectangle((3, 3, w - 4, h - 4), 44, outline=WHITE + (255,), width=4)
        d.text((w / 2, h / 2 + 2), txt, font=f, fill=(BLUE if hi else WHITE) + (255,), anchor="mm"); TC[key] = to_np(im)
    im = TC[key]; s = max(back((t - t0) / 0.35), 0.02)
    im2 = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
    blend(fr, im2, cx - im2.shape[1] / 2, cy - im2.shape[0] / 2, min(1, (t - t0) / 0.1))

# ---------------- subtitles from all voice sources ----------------
WR = json.load(open("words.json")); SUB = []
for s, si, so, t0 in SEGS:
    toks = []
    for a, b, w in WR[s]:
        if w.startswith("-") and toks: toks[-1][2] += w; continue
        toks.append([a, b, w])
    for a, b, w in toks:
        if b > si + 0.1 and a > si - 0.3 and a < so - 0.08:
            w = w.lower().strip(".,!?«»—…").strip()
            if w: SUB.append([t0 + max(0, a - si), t0 + min(b, so) - si, {"все": "всё", "-таки": "-таки"}.get(w, w)])
SUB.sort()
for i in range(len(SUB)):
    nxt = SUB[i + 1][0] if i + 1 < len(SUB) else DUR
    SUB[i][1] = min(nxt, max(SUB[i][1], SUB[i][0] + 0.3) + 0.25)

def face_disc_iv(t, r):
    f = V["iv"].get(int(round(src_of(t, "IMG_3119") * 30))); half = 470
    crop = f[int(700 - half):int(700 + half), int(570 - half):int(570 + half)]
    return cv2.resize(crop, (2 * int(r), 2 * int(r)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
def circle_iv(fr, t, t0, cx, cy, r):
    lt = t - t0; s = max(back(lt / 0.45), 0.02); rr = int(r * s)
    if rr < 4: return
    yo = 6 * np.sin(t * 1.8)
    put_disc(fr, face_disc_iv(t, rr), cx, cy + yo); ring(fr, cx, cy + yo, rr + 14, ease((lt - 0.15) / 0.7))
    pul = (lt * 0.8) % 1
    ring(fr, cx, cy, rr + 24 + 60 * pul, 1.0, col=tuple(int(c) for c in np.float32(PINK) * (1 - pul) + np.float32([250, 247, 246]) * pul), th=4)

LIGHT_BLOCKS = ("cards", "circ", "cta")
def render_block(t, b=None):
    n, t0, t1 = b or block(t); lt = t - t0; flash = 0; DEFER[0] = True; PENDING.clear(); LIGHT[0] = n in LIGHT_BLOCKS
    if n == "do":                                                       # ДО turnaround, front
        fr = vframe("do", 20.0 + t * 0.8, 1.25 + 0.05 * t / 3.5, 540, 820, 540, 900); chip(fr, CH_DO, 60, 228, t, 0.1)
    elif n == "iv1":
        fr = mblur(vsync("IMG_3119", t, 1.2 * (1 + 0.03 * lt / 3) * trans(t, t0), 570, 820, 900), lt)
    elif n == "cards":                                                  # after the first operation: lift + lipofilling
        fr = white_bg(t)
        card(fr, BF["f"], t, t0, 330, 780, 520, rot0=-7, frm=(-700, 0)); card(fr, BF["q"], t, t0 + 0.35, 760, 820, 520, rot0=6, frm=(700, 0))
        T2.draw(fr, 1300, t, t0 + 0.2, t1 - 0.05)
    elif n == "surg":                                                   # surgeon with the patient, lips in sync
        fr = mblur(vsync("IMG_3116", t, 1.15 * (1 + 0.04 * lt / 5) * trans(t, t0), 540, 900, 900), lt)
        T3.draw(fr, 1170, t, 11.3, 14.6); T3b.draw(fr, 1200, t, 14.9, t1 - 0.05)
    elif n == "circ":
        fr = white_bg(t); circle_iv(fr, t, t0, 540, 720, 300)
        T4.draw(fr, 1150, t, t0 + 0.45, t1 - 0.05)
    elif n == "impl":                                                   # implant choice
        if t < 22.0: fr = mblur(vframe("impl", 14.0 + lt, 1.15 * trans(t, t0), 540, 900), lt)
        else: fr = mblur(vframe("impl", src_of(t, "IMG_3144") if src_of(t, "IMG_3144") else 1.6, 1.1 * trans(t, 22.0), 540, 900), t - 22.0)
        T5.draw(fr, 1050, t, t0 + 0.2, 25.0)
        for k, (txt, d0) in enumerate([("275", 0.8), ("300", 1.25), ("325", 1.7)]):
            vchip(fr, txt + " мл", t, t0 + d0, 250 + k * 290, 1290, hi=(txt == "325" and t >= 25.05))
        T5b.draw(fr, 1060, t, 25.2, t1 - 0.05)
    elif n == "or":                                                     # quick operating-room cuts
        clips = [(26.6, "or3", 1.5, 1.1), (27.3, "or1", 0.5, 1.15), (28.0, "or3", 6.0, 1.3)]
        s0, k, st, z = [c for c in clips if c[0] <= t][-1]
        fr = mblur(vframe(k, st + (t - s0), z * trans(t, s0, 0.1), 540, 960), t - s0); flash = 0.25 * max(0, 1 - (t - s0) / 0.25)
    elif n == "after":
        fr = mblur(vframe("a59", 18.3 + lt * 0.6, 1.35 * trans(t, t0, 0.1), 540, 760, 540, 860), lt); flash = 0.3 * max(0, 1 - lt / 0.35)
        chip(fr, CH_PO, 60, 228, t, t0 + 0.1); T7.draw(fr, 1160, t, t0 + 0.2, t1 - 0.05)
    elif n == "cmp":                                                    # smooth ДО -> ПОСЛЕ (shoulders aligned)
        z = 1.0 + 0.04 * lt / 5.2
        a_ = body_view(BF["f"], SHO[("B", "f")], z=z); b_ = body_view(AF["f"], SHO[("A", "f")], z=z)
        p = ease((t - 32.2) / 1.6); f = 0.35; edge = (xx_ / W * 0.6 + yy_ / H * 0.4)[..., None]
        m = np.clip((p * (1 + f) - edge) / f, 0, 1); fr = a_ * (1 - m) + b_ * m
        chip(fr, CH_DO, 60, 228, t, t0, 32.2); chip(fr, CH_PO, 60, 228, t, 33.1)
        T8.draw(fr, 1180, t, 33.4, t1 - 0.05)
    elif n == "dream":
        fr = mblur(vframe("a60", 8.2 + lt * 0.75, 1.3 * trans(t, t0), 540, 760, 540, 860), lt)     # side → ¾, never the back; chip(fr, CH_PO, 60, 228, t, t0 + 0.1)
        T9.draw(fr, 1200, t, 36.85, t1 - 0.05)
    elif n in ("col1", "col2"):                                         # ДО/ПОСЛЕ collage: ПОСЛЕ top, ДО bottom
        k = "f" if n == "col1" else "q"; z = (1 + 0.02 * lt) * trans(t, t0, 0.05)
        fr = np.zeros((H, W, 3), np.float32)
        fr[:960] = span_view(AF[k], *SPAN[("A", k)], z); fr[960:] = span_view(BF[k], *SPAN[("B", k)], z)
        fr[957:963] = min(1, lt / 0.6); fr = mblur(fr, lt)
        chip(fr, CH_PO, 40, 228, t, t0 + 0.3); chip(fr, CH_DO, 40, 990, t, t0 + 0.55)
    else:                                                               # final CTA on white
        fr = white_bg(t)
        TC1.draw(fr, 760, t, t0 + 0.1, 1e9)
    if n == "do" or n == "iv1": T1.draw(fr, 1150, t, 0.3, 7.45)
    dark = not LIGHT[0] and n not in ("col1", "col2")
    if not LIGHT[0]: fr = look(fr, t)
    if dark: fr = fr * (1 - HAZE)
    if n not in ("col1", "col2", "cta"): subs(fr, t)
    DEFER[0] = False
    for args in PENDING: ablend(fr, *args)
    blend(fr, LOGO_D if LIGHT[0] else LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
    if flash > 0: fr = fr * (1 - flash) + flash
    return np.clip(fr, 0, 1)
def render(t):
    n, t0, t1 = block(t); i = BL.index((n, t0, t1)); fr = render_block(t)
    if i > 0 and t < t0 + XF:                                           # dissolve from the previous block
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
