"""@mcg.va — «возвращение в блог»: 5 talking stories + 2 photo stories (1080x1920, 25 fps).
  1 куда я пропала · 2 новости · 3 новая умелка · 4 фото «язык» · 5 работа и роды · 6 детский шопоголизм · 7 ложечки + опрос
Talking parts cut from IMG_8334 / IMG_8335: long pauses tightened to a natural breath, off-plan lines dropped; the audio
is overlap-added with 60 ms raised-cosine crossfades at every join and faded at the end, so nothing clicks or cuts off.
Text inserts sit in white cards at the top (the sky) from the first frame; thin Manrope subtitles run at the bottom.
Fonts (stories_kit): main = Inter Bold (IG Classic look), second = thin Montserrat, detail = Montserrat Italic;
subtitles = Manrope Light.
Work dir: dl/IMG_8334.MOV, dl/IMG_8335.MOV, dl/DSC01327.JPG, heic_full.jpg (IMG_8332), w8334.json, w8335.json,
Manrope-Light.ttf.
  python3 -I stories_return.py all out_dir
"""
import json, os, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import W, H, INK, font, wrap

FPS, SR = 25, 48000
WHITE = (255, 255, 255)
GREY = (105, 98, 102)
SUB_FONT = "Manrope-Light.ttf"

# (out name, source, spoken ranges in source seconds, text groups (show_from_src or None, hide_at_src or None, [(role, text)]))
STORIES = [
    ("story_1", "IMG_8334", [(1.45, 12.15)], [
        (None, None, [("second", "Если вы думаете, куда я опять пропала, то всё максимально банально."),
                      ("main", "Я просто нихрена не успеваю.")])]),
    ("story_2", "IMG_8334", [(12.22, 16.95), (20.95, 26.62)], [
        (None, None, [("second", "Из новостей: сделали очередную прививку, готовимся к холодам и обновляем гардероб."),
                      ("main", "У ребёнка вещей скоро будет больше, чем у меня."),
                      ("detail", "Хотя, кажется, уже.")])]),
    ("story_3", "IMG_8334", [(30.80, 34.25)], [
        (None, None, [("second", "А ещё у нас новая умелка."),
                      ("main", "Теперь всем показываем язык.")])]),
    ("story_5", "IMG_8334", [(36.85, 54.72)], [
        (None, 49.0, [("second", "А вообще, я очень хочу вернуться к более регулярному контенту."),
                      ("second", "Тем более у меня столько всего накопилось, что хочется вам рассказать.")]),
        (49.0, None, [("main", "И да, скоро наконец-то будут видео про роды и роддом."),
                      ("detail", "Я до них всё-таки добралась!")])]),
    ("story_6", "IMG_8335", [(0.0, 15.0), (15.95, 24.6)], [
        (None, None, [("second", "Кажется, у меня официально началась новая стадия материнства."),
                      ("detail", "Муж предлагает купить что-нибудь себе, а я уже выбираю очередную штуку для ребёнка."),
                      ("main", "Детский шопоголизм.")])]),
]
FIXES = {"шипоголизм": "шопоголизм", "пойдёт": "обойдёт", "чё": "что"}     # whisper slips in the subtitles

# ---------------------------------------------------------------- cut: word gaps → kept intervals
GAP, PAD_OUT, PAD_IN = 0.45, 0.18, 0.13     # gaps longer than GAP shrink to PAD_OUT + PAD_IN (≈ 0.3 s, a natural breath)
XF = 0.06                                    # crossfade at each join

def words(src): return json.load(open(f"w{src[-4:]}.json"))

def intervals(src, ranges):
    Wd = words(src); iv = []
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

def load_audio(src):
    a = subprocess.run(["ffmpeg", "-v", "error", "-i", f"dl/{src}.MOV", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                       capture_output=True).stdout
    return np.frombuffer(a, np.int16).astype(np.float32) / 32768

def cut_audio(src, iv, out):
    """overlap-add the kept pieces with raised-cosine crossfades centred on each join (length = sum of pieces)"""
    x = load_audio(src); n = int(round(sum(b - a for a, b in iv) * SR)); y = np.zeros(n + SR, np.float32)
    h = int(XF * SR / 2); pos = 0
    for k, (a, b) in enumerate(iv):
        s0, s1 = int(a * SR), int(b * SR)
        lo = h if k > 0 else 0; hi = h if k < len(iv) - 1 else 0
        seg = x[max(0, s0 - lo):s1 + hi].copy()
        if lo: seg[:2 * lo] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 2 * lo))
        if hi: seg[-2 * hi:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, 2 * hi))
        st = pos - lo; y[max(0, st):st + len(seg)] += seg[max(0, -st):]
        pos += s1 - s0
    y = y[:n]
    fi, fo = int(0.04 * SR), int(0.28 * SR)                       # soft start, gentle tail fade so the last word rings out
    y[:fi] *= np.linspace(0, 1, fi); y[-fo:] *= np.cos(np.linspace(0, np.pi / 2, fo)) ** 2
    with wave.open(out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())

def cut_video(src, iv, wav, out):
    parts = "".join(f"[0:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS,fps={FPS},scale={W}:{H},setsar=1[v{k}];" for k, (a, b) in enumerate(iv))
    chain = parts + "".join(f"[v{k}]" for k in range(len(iv))) + f"concat=n={len(iv)}:v=1:a=0[cv];" \
            "[1:a]highpass=f=90,afftdn=nr=8:nf=-32,loudnorm=I=-16:TP=-1.5:LRA=9[ao]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"dl/{src}.MOV", "-i", wav, "-filter_complex", chain,
                    "-map", "[cv]", "-map", "[ao]", "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-shortest", out], check=True)

# ---------------------------------------------------------------- subtitles
def subtitles(src, iv):
    ws = []
    for a, b, t in words(src):
        if any(x - 0.02 <= a and b <= y + 0.05 for x, y in iv):
            t = " ".join(FIXES.get(p, p) for p in t.split())
            if t.startswith("-") and ws: ws[-1][2] += t; ws[-1][1] = b; continue
            ws.append([a, b, t])
    chunks, cur = [], []
    for i, w in enumerate(ws):
        cur.append(w); n = sum(len(x[2]) + 1 for x in cur)
        if w[2][-1] in ",.!?" or len(cur) >= 4 or n > 24 or i == len(ws) - 1: chunks.append(cur); cur = []
    out = []
    for i, c in enumerate(chunks):
        t0 = src_to_cut(iv, c[0][0]) - 0.05
        t1 = src_to_cut(iv, chunks[i + 1][0][0]) - 0.05 if i + 1 < len(chunks) else 1e9
        t1 = min(t1, src_to_cut(iv, c[-1][1]) + 0.5)
        out.append((max(0, t0), t1, " ".join(x[2] for x in c).rstrip(".,").replace("...", "").replace("…", "")))
    return out

# ---------------------------------------------------------------- text layer
SIZES = {"main": 48, "second": 40, "detail": 36}
LEAD = {"main": 62, "second": 54, "detail": 48}
CARD_TOP, CARD_PADX, CARD_PADY, CARD_GAP, CARD_MAXW = 262, 46, 30, 18, 880
SUB_Y, SUB_SIZE = 1470, 44

def layout(items):
    """each line block in its own white rounded card, stacked down from CARD_TOP"""
    out = []; y0 = CARD_TOP
    for r, t in items:
        f = font(r, SIZES[r]); ls = wrap(t, f, CARD_MAXW)
        w = max(f.getlength(l) for l in ls) + 2 * CARD_PADX
        h = LEAD[r] * len(ls) + 2 * CARD_PADY - (LEAD[r] - f.size)
        box = ((W - w) / 2, y0, (W + w) / 2, y0 + h)
        out.append((r, ls, y0 + CARD_PADY + f.size / 2, box)); y0 += h + CARD_GAP
    return out

def text_layer(groups, subs, t):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sh = Image.new("L", (W, H), 0); csh = Image.new("L", (W, H), 0)
    d, ds, dc = ImageDraw.Draw(im), ImageDraw.Draw(sh), ImageDraw.Draw(csh)
    for g0, g1, blocks in groups:
        if t < g0 or (g1 is not None and t >= g1): continue
        a = 1.0 if g0 == 0 else min(1, (t - g0) / 0.2)             # story 5's second group fades in; the rest is there from frame 1
        for role, ls, y, box in blocks:
            rad = min(34, (box[3] - box[1]) / 2)
            dc.rounded_rectangle((box[0], box[1] + 8, box[2], box[3] + 8), rad, fill=int(70 * a))
            d.rounded_rectangle(box, rad, fill=(255, 255, 255, int(255 * a)))
            col = GREY if role == "detail" else INK
            for k, l in enumerate(ls):
                d.text((W / 2, y + k * LEAD[role]), l, font=font(role, SIZES[role]), fill=col + (int(255 * a),), anchor="mm")
    fs = ImageFont.truetype(SUB_FONT, SUB_SIZE)
    for s0, s1, txt in subs:
        if s0 <= t < s1:
            ls = wrap(txt, fs, 900); y = SUB_Y - (len(ls) - 1) * 28
            for l in ls:
                d.text((W / 2, y), l, font=fs, fill=WHITE + (255,), anchor="mm")
                ds.text((W / 2, y + 2), l, font=fs, fill=190, anchor="mm"); y += 56
    shadow = Image.new("RGBA", (W, H), (18, 14, 16, 0))
    shadow.putalpha(Image.fromarray(np.maximum(np.asarray(sh.filter(ImageFilter.GaussianBlur(6))),
                                               np.asarray(csh.filter(ImageFilter.GaussianBlur(22))))))
    return Image.alpha_composite(shadow, im)

def overlay(cut_mp4, groups, subs, out):
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
        fading = any(g0 > 0 and 0 <= t - g0 < 0.2 for g0, _, _ in groups)
        key = n if fading else (tuple(t >= g0 and (g1 is None or t < g1) for g0, g1, _ in groups),
                                next((i for i, (s0, s1, _) in enumerate(subs) if s0 <= t < s1), -1))
        if key not in cache: cache = {key: text_layer(groups, subs, t)}
        fr = Image.alpha_composite(Image.frombuffer("RGB", (W, H), b).convert("RGBA"), cache[key]).convert("RGB")
        enc.stdin.write(fr.tobytes()); n += 1
    enc.stdin.close(); enc.wait(); dec.wait()

# ---------------------------------------------------------------- photo stories
def shadowed(img, box, radius, blur=28, off=10, alpha=0.22, fill=(255, 255, 255)):
    x0, y0, x1, y1 = box
    sh = Image.new("L", img.size, 0); ImageDraw.Draw(sh).rounded_rectangle((x0, y0 + off, x1, y1 + off), radius, fill=int(255 * alpha))
    img.paste(Image.new("RGB", img.size, (40, 30, 35)), (0, 0), sh.filter(ImageFilter.GaussianBlur(blur)))
    if fill: ImageDraw.Draw(img).rounded_rectangle(box, radius, fill=fill)

def tongue_story(out):
    """separate story: the tongue photo with the aside in a white card"""
    im = Image.open("dl/DSC01327.JPG").convert("RGB").rotate(90, expand=True)     # camera held sideways → upright
    im = im.resize((W, H), Image.LANCZOS)
    t = "В общем, растёт очень воспитанная девочка."; f = font("detail", 40); tw = f.getlength(t)
    x0, y0 = (W - tw) / 2 - 44, 1440
    shadowed(im, (x0, y0, W - x0, y0 + 96), 48)
    ImageDraw.Draw(im).text((W / 2, y0 + 49), t, font=f, fill=INK, anchor="mm")
    im.save(out, quality=95, subsampling=0)

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
    for txt in ["Казалось бы, просто ложечки.", "Но почему мне теперь хочется скупить вообще всё детское?!"]:
        for l in wrap(txt, font("main", 46), 900):
            d.text((W / 2, y), l, font=font("main", 46), fill=INK, anchor="mm"); y += 58
    d.text((W / 2, y + 16), "Потом покажу, что в итоге набрала.", font=font("detail", 38), fill=GREY, anchor="mm")
    img.save(out, quality=95, subsampling=0)

def build(k, od, tmp):
    name, src, ranges, groups = STORIES[k]
    iv = intervals(src, ranges)
    wav = os.path.join(tmp, f"{name}.wav"); cut_audio(src, iv, wav)
    cut = os.path.join(tmp, f"{name}_cut.mp4"); cut_video(src, iv, wav, cut)
    gc = [(0.0 if g0 is None else src_to_cut(iv, g0), None if g1 is None else src_to_cut(iv, g1), layout(items))
          for g0, g1, items in groups]
    overlay(cut, gc, subtitles(src, iv), os.path.join(od, f"{name}.mp4"))
    print(name, round(sum(b - a for a, b in iv), 2), "s,", len(iv), "pieces")

if __name__ == "__main__":
    od = sys.argv[2]; tmp = os.path.join(od, "_tmp"); os.makedirs(tmp, exist_ok=True)
    sel = sys.argv[1]
    for k, s in enumerate(STORIES):
        if sel == "all" or s[0] in sel.split(","): build(k, od, tmp)
    if sel == "all" or "story_4" in sel: tongue_story(os.path.join(od, "story_4.jpg"))
    if sel == "all" or "story_7" in sel: spoons_story(os.path.join(od, "story_7.jpg"))
