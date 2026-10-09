import json
exec(open("reel2_base.py").read())

# ---------------- voice edit: (src_in, src_out, reel_start) ----------------
SEGS = [(7.84, 14.25, 0.6), (14.62, 24.12, 7.06), (24.62, 29.95, 16.71), (30.40, 33.66, 25.0),
        (35.16, 38.45, 28.46), (38.90, 46.24, 32.0), (49.95, 52.70, 41.1)]
def src_of(t):
    best = None
    for si, so, t0 in SEGS:
        if t0 <= t <= t0 + so - si: return si + t - t0
        if t < t0: return si if best is None else best
        best = so
    return best

BL = [("hook", 0, 4.6), ("talk1", 4.6, 7.05), ("before", 7.05, 11.65), ("talk2", 11.65, 13.6), ("front", 13.6, 15.3),
      ("cut", 15.3, 23.7), ("after", 23.7, 27.0), ("split", 27.0, 31.0), ("talk3", 31.0, 33.8), ("side", 33.8, 36.6),
      ("collage", 36.6, 38.6), ("slider", 38.6, 40.9), ("final", 40.9, DUR)]
def block(t):
    for n, a, b in BL:
        if t < b: return n, a, b
    return BL[-1]

# ---------------- interview frames (face already blurred in the source) ----------------
TALK = {n: (a, b) for n, a, b in BL if n.startswith("talk")}
need = sorted({int(round(src_of(i / FPS) * 30)) for n, (a, b) in TALK.items() for i in range(int(a * FPS), int(b * FPS) + 2)})
IV = {}
pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", SP + "v13/dl/i", "-vf", "fps=30,scale=1080:1920", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
k = 0; ns = set(need)
while k <= need[-1]:
    buf = pr.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    if k in ns: IV[k] = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    k += 1
pr.kill()
def iframe(t, z, cy=900):
    i = int(round(src_of(t) * 30)); i = min(IV, key=lambda j: abs(j - i)) if i not in IV else i
    M = np.float32([[z, 0, 540 - z * 540], [0, z, 960 - z * cy]]); M[1, 2] = np.clip(M[1, 2], H - z * H, 0)
    return cv2.warpAffine(IV[i], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255

# ---------------- word-by-word subtitles from the interview ----------------
WR = json.load(open("words.json")); toks = []
for a, b, w in WR:
    if w.startswith("-") and toks: toks[-1][2] += w; continue
    toks.append([a, b, w])
SUB = []
for si, so, t0 in SEGS:
    for a, b, w in toks:
        if b > si + 0.12 and a > si - 0.4 and a < so - 0.1:
            w = w.lower().strip(".,!?«»"); w = "всё" if w == "все" else w
            SUB.append([t0 + max(0, a - si), t0 + min(b, so) - si, w])
for i in range(len(SUB)):
    nxt = SUB[i + 1][0] if i + 1 < len(SUB) else DUR
    SUB[i][1] = min(nxt, max(SUB[i][1], SUB[i][0] + 0.3) + 0.25)

T_NAME = Txt([("ИРИНА", "H"), ("пациентка, мама двоих детей", "S")], size=50, align="L", x=70)

def render(t):
    n, t0, t1 = block(t); lt = t - t0; flash = 0
    if n == "hook":
        fr = grade(vframe(15.55 + t, 1.32 + 0.05 * t / 4.6))
        slide_chip(fr, CH_DO, 60, 228, t, 0.15)
        T_HOOK.draw(fr, 1090, t, 0.45, t1 - 0.1, cps=24)
    elif n.startswith("talk"):
        tz, _ = trans(t, t0); z = dict(talk1=1.08, talk2=1.38, talk3=1.2)[n] * (1 + 0.03 * lt / 2.5)
        fr = grade(mblur(iframe(t, z * tz, 820 if n == "talk2" else 900), lt))
    elif n == "before":
        tz, _ = trans(t, t0)
        fr = grade(mblur(vframe(11.0 + lt, (1.22 + 0.05 * lt / 4.6) * tz), lt))
        slide_chip(fr, CH_DO, 60, 228, t, t0 + 0.05)
        T_KNOW.draw(fr, 1140, t, t0 + 0.15, t1 - 0.1, cps=28)
        items_list(fr, t, [("форма изменилась", t0 + 0.85), ("появилась разница в объёме", t0 + 1.9), ("хочется вернуть наполненность", t0 + 2.95)], 80, 1250)
    elif n == "front":
        tz, _ = trans(t, t0)
        fr = grade(mblur(vframe(17.2 + lt, (1.75 + 0.06 * lt / 1.7) * tz, fyo=230), lt))
        slide_chip(fr, CH_DO, 60, 228, t, t0 + 0.05)
    elif n == "cut":
        segs = [(15.3, 9.3, 1.22), (16.98, 10.4, 1.5), (18.66, 12.3, 1.3), (20.34, 14.6, 1.45), (22.02, 18.4, 1.6)]
        s0, src, z = [s for s in segs if s[0] <= t][-1]; tz, _ = trans(t, s0)
        fr = grade(mblur(vframe(src + (t - s0) * 0.8, z * tz), t - s0))
        bracket(fr, t, t0 + 0.3, ["И ДАЛЬШЕ РЕШЕНИЕ", "ПРИШЛО БЫСТРО"], 1250, 62, t1 - 0.15)
    elif n == "after":
        tz, flash = trans(t, t0, True)
        fr = grade(photo(PO, MPO, (1.12 - 0.08 * ease(lt / 3.3)) * tz))
        slide_chip(fr, CH_PO, 60, 228, t, t0 + 1.0)
        T_AFT.draw(fr, 1215, t, t0 + 1.0, t1 - 0.1, cps=32)
    elif n == "split":
        fr = np.zeros((H, W, 3), np.float32); z = 1.0 + 0.03 * lt / 4
        top = np.ascontiguousarray(vframe(16.0 + lt, 1.3 * z, fyo=-60)[400:1360]); bot = photo(PO, MPO, z, dy=-330, size=(W, 960))
        ox1 = -W * (1 - ease(lt / 0.4)); ox2 = W * (1 - ease((lt - 0.12) / 0.4))
        fr[:960] = cv2.warpAffine(top, np.float32([[1, 0, ox1], [0, 1, 0]]), (W, 960))
        fr[960:] = cv2.warpAffine(bot, np.float32([[1, 0, ox2], [0, 1, 0]]), (W, 960))
        fr = grade(fr); fr[957:963] = 1.0 * min(1, lt / 0.5)
        slide_chip(fr, CH_DO, 60, 228, t, t0 + 0.3); slide_chip(fr, CH_PO, 60, 1088, t, t0 + 0.45)
        T_FIRST.draw(fr, 862, t, t0 + 0.3, t1 - 0.1, cps=32)
    elif n == "side":
        tz, _ = trans(t, t0)
        fr = grade(mblur(photo(COL, MSIDE, (1.0 + 0.07 * lt / 2.8) * tz, fx=540, fy=700), lt))
        slide_chip(fr, CH_PO, 60, 228, t, t0 + 0.1)
        bracket(fr, t, t0 + 0.1, ["«КАК ПОСЛЕ ТРЕНИРОВОК»"], 1300, 66, t1 - 0.1)
    elif n == "collage":
        tz, _ = trans(t, t0)
        fr = grade(mblur(photo(COL, MCOL, (1.0 + 0.03 * lt / 2) * tz, fy=960), lt))
        bar(fr, 872, 1036, 1.0, ease(lt / 0.3)); T_MON.draw(fr, 908, t, t0 + 0.2, 1e9, cps=32)
        slide_chip(fr, CH_PO, 60, 228, t, t0 + 0.3); slide_chip(fr, CH_DO, 60, 1090, t, t0 + 0.45)
    elif n == "slider":
        z = 1.0 + 0.04 * lt / 2.3; a = photo(DO, MDO, z); b = photo(PO, MPO, z)
        sx = int(W * (1 - ease((lt - 0.2) / 1.7))); fr = a.copy(); fr[:, sx:] = b[:, sx:]; fr = grade(fr)
        if 0 < sx < W: fr[:, max(0, sx - 3):sx + 3] = 1.0
        slide_chip(fr, CH_DO, 60, 228, t, t0 + 0.05, t0 + 1.2)
        slide_chip(fr, CH_PO, W - 60 - CH_PO.shape[1], 228, t, t0 + 1.0, 1e9, frm=1)
        bracket(fr, t, t0 + 0.3, ["РЕЗУЛЬТАТ", "СПУСТЯ 2,5 МЕСЯЦА"], 1270, 64, t1 - 0.1)
    else:
        tz, flash = trans(t, t0, True)
        fr = grade(photo(PO, MPO, (1.10 - 0.06 * ease(lt / 4)) * tz))
        for k, (wd, tw) in enumerate([("ОЧЕНЬ", 0.25), ("ДОВОЛЬНА", 0.85)]):
            if ("fw", wd) not in BRC:
                f = osw(78); asc, desc = f.getmetrics(); im = Image.new("RGBA", (int(f.getlength(wd)) + 40, asc + desc + 20), (0, 0, 0, 0))
                ImageDraw.Draw(im).text((20, 10 + asc), wd, font=f, fill=WHITE + (255,), anchor="ls"); BRC[("fw", wd)] = to_np(im)
            pop_img(fr, BRC[("fw", wd)], 540, 1200 + k * 96, t, t0 + tw)
        fr *= 1 - 0.85 * max(0, (t - (DUR - 0.7)) / 0.7)
    subs(fr, t)
    blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
    if flash > 0: fr = fr * (1 - flash) + flash
    return np.clip(fr, 0, 1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for ts in sys.argv[1:]:
            cv2.imwrite(f"f/p_{ts}.jpg", cv2.cvtColor((render(float(ts)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        sys.exit()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "video2.mp4"], stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        enc.stdin.write((render(i / FPS) * 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
