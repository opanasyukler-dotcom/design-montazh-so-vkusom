"""@mcg.va — «возвращение в блог»: 5 talking stories + 1 photo story (1080x1920, 25 fps).
Talking parts cut from IMG_8334 / IMG_8335 (pauses tightened, off-plan lines dropped), short text inserts that appear
line by line in sync with the speech, white type on the dark coat at the bottom (Instagram safe zone).
Fonts (stories_kit): main = Inter Bold (IG Classic look), second = thin Montserrat, detail = Montserrat Italic.
Story 3 ends on the «tongue» photo; story 6 is the spoons photo as a card with room for a native poll sticker.
Work dir: dl/IMG_8334.MOV, dl/IMG_8335.MOV, dl/DSC01327.JPG, heic_full.jpg (IMG_8332), w8334.json, w8335.json.
  python3 -I stories_return.py all out_dir
"""
import json, os, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import W, H, INK, font, wrap

FPS = 25
WHITE = (255, 255, 255)
GREY = (105, 98, 102)
TH, MINSIL, KEEPSIL = -40, 0.24, 0.07       # pauses: quiet below TH for ≥ MINSIL → keep KEEPSIL on each side

# (source, [(t0, t1), ...] spoken ranges in source seconds, text groups)
# text group: (show_from_src, hide_at_src or None, [(role, text, appear_src), ...])
STORIES = [
    ("IMG_8334", [(1.45, 11.95)], [
        (1.45, None, [("second", "Если вы думаете, куда я опять пропала, то всё максимально банально.", 2.0),
                     ("main", "Я просто нихрена не успеваю.", 6.4)])]),
    ("IMG_8334", [(12.22, 16.85), (20.95, 26.62)], [
        (12.2, None, [("second", "Из новостей: сделали очередную прививку, готовимся к холодам и обновляем гардероб.", 12.5),
                      ("main", "У ребёнка вещей скоро будет больше, чем у меня.", 22.7),
                      ("detail", "Хотя, кажется, уже.", 24.9)])]),
    ("IMG_8334", [(30.95, 34.0)], [
        (30.9, None, [("second", "А ещё у нас новая умелка.", 31.0),
                      ("main", "Теперь всем показываем язык.", 32.5)])]),
    ("IMG_8334", [(36.85, 54.75)], [
        (36.8, 49.05, [("second", "А вообще, я очень хочу вернуться к более регулярному контенту.", 37.0),
                       ("second", "Тем более у меня столько всего накопилось, что хочется вам рассказать.", 42.2)]),
        (49.05, None, [("main", "И да, скоро наконец-то будут видео про роды и роддом.", 49.2),
                       ("detail", "Я до них всё-таки добралась!", 52.9)])]),
    ("IMG_8335", [(0.0, 15.0), (15.95, 24.25)], [
        (0.0, None, [("second", "Кажется, у меня официально началась новая стадия материнства.", 0.4),
                     ("detail", "Муж предлагает купить что-нибудь себе, а я уже выбираю очередную штуку для ребёнка.", 9.5),
                     ("main", "Детский шопоголизм.", 16.3)])]),
]
TONGUE_HOLD = 3.0                            # story 3: the tongue photo after her line (source audio keeps running)

# ---------------------------------------------------------------- audio analysis / cut
def loud_db(src):
    a = subprocess.run(["ffmpeg", "-v", "error", "-i", f"dl/{src}.MOV", "-ac", "1", "-ar", "48000", "-f", "s16le", "-"],
                       capture_output=True).stdout
    x = np.frombuffer(a, np.int16).astype(np.float32) / 32768
    n = len(x) // 480
    return 20 * np.log10(np.sqrt((x[:n * 480].reshape(n, 480) ** 2).mean(1) + 1e-12))

GAP, PAD_OUT, PAD_IN = 0.30, 0.10, 0.12     # word gaps longer than GAP are tightened (wind noise defeats a loudness gate)

def intervals(src, ranges):
    Wd = json.load(open(f"w{src[-4:]}.json")); iv = []
    for x0, y0 in ranges:
        ws = [w for w in Wd if w[0] >= x0 - 0.05 and w[1] <= y0 + 0.05]; c = x0
        for w, nx in zip(ws, ws[1:]):
            if nx[0] - w[1] > GAP:
                iv.append((c, w[1] + PAD_OUT)); c = nx[0] - PAD_IN
        iv.append((c, y0))
    return [(round(a * FPS) / FPS, round(b * FPS) / FPS) for a, b in iv if b - a > 1.5 / FPS]

def src_to_cut(iv, t):
    acc = 0.0
    for a, b in iv:
        if t < a: return acc
        if t <= b: return acc + t - a
        acc += b - a
    return acc

def cut_video(src, iv, out, tail=None):
    """concat the kept intervals (video + audio); tail=(still_png, seconds, audio_from_src) appends a still"""
    parts = []
    for k, (a, b) in enumerate(iv):
        parts.append(f"[0:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS,fps={FPS},scale={W}:{H},setsar=1[v{k}];"
                      f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS,afade=t=in:d=0.012,afade=t=out:st={b - a - 0.012:.3f}:d=0.012[a{k}];")
    n = len(iv); ins = ["-i", f"dl/{src}.MOV"]
    chain = "".join(parts) + "".join(f"[v{k}][a{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=1[cv][ca]"
    if tail:
        png, dur, a0 = tail
        ins += ["-loop", "1", "-t", f"{dur}", "-i", png]
        chain += (f";[1:v]fps={FPS},scale={W}:{H},setsar=1,format=yuv420p[tv];"
                  f"[0:a]atrim={a0:.3f}:{a0 + dur:.3f},asetpts=PTS-STARTPTS,volume=0.35,afade=t=out:st=0.15:d=0.45[ta];"
                  f"[cv][ca][tv][ta]concat=n=2:v=1:a=1[cv2][ca2]")
        vo, ao = "[cv2]", "[ca2]"
    else: vo, ao = "[cv]", "[ca]"
    chain += f";{ao}highpass=f=90,afftdn=nr=8:nf=-32,loudnorm=I=-16:TP=-1.5:LRA=9[ao]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", chain, "-map", vo, "-map", "[ao]",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", out], check=True)

# ---------------------------------------------------------------- text layer
SIZES = {"main": 56, "second": 44, "detail": 40}
LEAD = {"main": 70, "second": 58, "detail": 52}
MAXW, BOTTOM, GAP_B = 940, 1575, 30
CARD_TOP, CARD_PADX, CARD_PADY, CARD_GAP, CARD_MAXW, MAIN_Y = 262, 46, 30, 18, 820, 1505

def layout(items, mode="bottom"):
    """bottom: white lines stacked up from BOTTOM (over the dark coat);
    top: second/detail lines in white cards stacked down from CARD_TOP (close-up framing), main in white at MAIN_Y.
    returns [(role, [lines], y_first_line_centre, appear, card_box or None)]"""
    out = []
    if mode == "bottom":
        blocks = [(r, wrap(t, font(r, SIZES[r]), MAXW), ap) for r, t, ap in items]
        hgt = sum(LEAD[r] * len(ls) for r, ls, _ in blocks) + GAP_B * (len(blocks) - 1)
        y = BOTTOM - hgt + LEAD[blocks[0][0]] / 2
        for r, ls, ap in blocks:
            out.append((r, ls, y, ap, None)); y += LEAD[r] * len(ls) + GAP_B
        return out
    y0 = CARD_TOP
    for r, t, ap in items:
        f = font(r, SIZES[r])
        if r == "main":
            out.append((r, wrap(t, f, MAXW), MAIN_Y, ap, None)); continue
        ls = wrap(t, f, CARD_MAXW); w = max(f.getlength(l) for l in ls) + 2 * CARD_PADX
        h = LEAD[r] * len(ls) + 2 * CARD_PADY - (LEAD[r] - f.size)
        box = ((W - w) / 2, y0, (W + w) / 2, y0 + h)
        out.append((r, ls, y0 + CARD_PADY + f.size / 2, ap, box)); y0 += h + CARD_GAP
    return out

def text_layer(groups_cut, t):
    """RGBA overlay at time t (cut seconds)"""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sh = Image.new("L", (W, H), 0); csh = Image.new("L", (W, H), 0)
    d, ds, dc = ImageDraw.Draw(im), ImageDraw.Draw(sh), ImageDraw.Draw(csh)
    any_ = False
    for g0, g1, blocks in groups_cut:
        if t < g0 or (g1 is not None and t >= g1 + 0.25): continue
        out = 0.0 if g1 is None or t < g1 else (t - g1) / 0.25
        for role, ls, y, appear, box in blocks:
            p = (t - appear) / 0.35
            if p <= 0: continue
            e = 1 - (1 - min(1, p)) ** 3; a = e * (1 - out)
            f = font(role, SIZES[role]); dy = (1 - e) * 18
            if box:
                bx = (box[0], box[1] + dy, box[2], box[3] + dy); rad = min(34, (box[3] - box[1]) / 2)
                dc.rounded_rectangle((bx[0], bx[1] + 8, bx[2], bx[3] + 8), rad, fill=int(70 * a))
                d.rounded_rectangle(bx, rad, fill=(255, 255, 255, int(255 * a)))
                col = INK if role == "second" else GREY
            for k, l in enumerate(ls):
                yy = y + k * LEAD[role] + dy
                if box: d.text((W / 2, yy), l, font=f, fill=col + (int(255 * a),), anchor="mm")
                else:
                    d.text((W / 2, yy), l, font=f, fill=WHITE + (int(255 * a),), anchor="mm")
                    ds.text((W / 2, yy + 3), l, font=f, fill=int(150 * a), anchor="mm")
            any_ = True
    if not any_: return None
    shadow = Image.new("RGBA", (W, H), (20, 16, 18, 0))
    shadow.putalpha(Image.fromarray(np.maximum(np.asarray(sh.filter(ImageFilter.GaussianBlur(8))),
                                               np.asarray(csh.filter(ImageFilter.GaussianBlur(22))))))
    return Image.alpha_composite(shadow, im)

def overlay(cut_mp4, groups_cut, out):
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", cut_mp4],
                               capture_output=True, text=True).stdout)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", cut_mp4, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-i", cut_mp4, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium",
                            "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", "-shortest", out],
                           stdin=subprocess.PIPE)
    n = 0; cache = {}
    while True:
        b = dec.stdout.read(W * H * 3)
        if len(b) < W * H * 3: break
        t = n / FPS
        key = round(t * FPS) if any(abs(t - bl[3]) < 0.4 or (g1 and abs(t - g1) < 0.3) for g0, g1, bls in groups_cut for bl in bls) else \
              tuple(bl[3] <= t for g0, g1, bls in groups_cut for bl in bls) + tuple(g1 is not None and t >= g1 + 0.25 for g0, g1, _ in groups_cut)
        if key not in cache: cache = {key: text_layer(groups_cut, t)}
        lay = cache[key]
        fr = Image.frombuffer("RGB", (W, H), b)
        if lay is not None: fr = Image.alpha_composite(fr.convert("RGBA"), lay).convert("RGB")
        enc.stdin.write(fr.tobytes()); n += 1
    enc.stdin.close(); enc.wait(); dec.wait()

# ---------------------------------------------------------------- photo stories
def tongue_png(path):
    im = Image.open("dl/DSC01327.JPG").convert("RGB").rotate(90, expand=True)     # camera held sideways → upright
    im = im.resize((W, H), Image.LANCZOS)
    t = "В общем, растёт очень воспитанная девочка."; f = font("detail", 40); tw = f.getlength(t)
    x0, y0 = (W - tw) / 2 - 44, 1440; box = (x0, y0, W - x0, y0 + 96)
    shadowed(im, box, 48)
    ImageDraw.Draw(im).text((W / 2, y0 + 49), t, font=f, fill=INK, anchor="mm")
    im.save(path)

def shadowed(img, box, radius, blur=28, off=10, alpha=0.22, fill=(255, 255, 255)):
    x0, y0, x1, y1 = box
    sh = Image.new("L", img.size, 0); ImageDraw.Draw(sh).rounded_rectangle((x0, y0 + off, x1, y1 + off), radius, fill=int(255 * alpha))
    img.paste(Image.new("RGB", img.size, (40, 30, 35)), (0, 0), sh.filter(ImageFilter.GaussianBlur(blur)))
    if fill: ImageDraw.Draw(img).rounded_rectangle(box, radius, fill=fill)

def spoons_story(out):
    ph = Image.open("heic_full.jpg").convert("RGB")                       # 4032x3024 landscape
    bgk = ph.resize((int(ph.width * H / ph.height), H), Image.LANCZOS)
    x = (bgk.width - W) // 2; bgk = bgk.crop((x, 0, x + W, H)).filter(ImageFilter.GaussianBlur(40))
    a = np.asarray(bgk, np.float32); a = a * 0.55 + 255 * 0.45                       # light, soft backdrop for dark type
    img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    y = 285
    for l in wrap("Вот, например, уже начинаю закупаться к прикорму.", font("second", 42), 860):
        d.text((W / 2, y), l, font=font("second", 42), fill=INK, anchor="mm"); y += 56
    cw, ch = 840, 630; cx0, cy0 = (W - cw) // 2, y + 22
    card = ph.resize((cw, ch), Image.LANCZOS)
    shadowed(img, (cx0, cy0, cx0 + cw, cy0 + ch), 30, fill=None)
    m = Image.new("L", (cw, ch), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, cw - 1, ch - 1), 30, fill=255)
    img.paste(card, (cx0, cy0), m)
    y = cy0 + ch + 62; d = ImageDraw.Draw(img)
    for l in wrap("Казалось бы, просто ложечки.", font("main", 46), 900):
        d.text((W / 2, y), l, font=font("main", 46), fill=INK, anchor="mm"); y += 58
    for l in wrap("Но почему мне теперь хочется скупить вообще всё детское?!", font("main", 46), 900):
        d.text((W / 2, y), l, font=font("main", 46), fill=INK, anchor="mm"); y += 58
    d.text((W / 2, y + 16), "Потом покажу, что в итоге набрала.", font=font("detail", 38), fill=GREY, anchor="mm")
    img.save(out, quality=95, subsampling=0)
    return y + 16                                                         # poll sticker goes below this line

def build(k, od, tmp):
    src, ranges, groups = STORIES[k]
    iv = intervals(src, ranges)
    tail = None
    if k == 2:
        png = os.path.join(tmp, "tongue.png"); tongue_png(png)
        tail = (png, TONGUE_HOLD, ranges[-1][1])
    cut = os.path.join(tmp, f"cut{k + 1}.mp4"); cut_video(src, iv, cut, tail)
    gc = []
    for g0, g1, items in groups:
        blocks = layout(items, "top" if k == 4 else "bottom")            # story 5 is a close-up: cards on top
        blocks = [(r, ls, y, src_to_cut(iv, ap), box) for r, ls, y, ap, box in blocks]
        gc.append((src_to_cut(iv, g0), None if g1 is None else src_to_cut(iv, g1), blocks))
    if k == 2:   # the talking text hides on the cut to the tongue photo (its aside card is baked into the still)
        tcut = sum(b - a for a, b in iv)
        gc[0] = (gc[0][0], tcut - 0.25, gc[0][2])
    overlay(cut, gc, os.path.join(od, f"story_{k + 1}.mp4"))
    print(k + 1, "cut", round(sum(b - a for a, b in iv), 2), "s,", len(iv), "pieces")

if __name__ == "__main__":
    od = sys.argv[2]; tmp = os.path.join(od, "_tmp"); os.makedirs(tmp, exist_ok=True)
    which = range(5) if sys.argv[1] == "all" else [int(x) - 1 for x in sys.argv[1].split(",")]
    for k in which:
        if k < 5: build(k, od, tmp)
    if sys.argv[1] == "all" or "6" in sys.argv[1].split(","):
        print("poll below y", spoons_story(os.path.join(od, "story_6.jpg")))
