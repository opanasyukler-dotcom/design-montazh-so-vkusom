"""@mcg.va — «первая посуда для прикорма», minimal review edit, 16:9 like the source (3840x2160, 25 fps).
Voice-over = the six narration mp3s in script order; shots cut to the words (faster-whisper timings in words.json).
Subtitles: thin Montserrat (wght 300), white with a soft glyph shadow over a long black spray at the very bottom.
Pieces the narration does not name get small pills (Montserrat Italic, INK on white).
  python3 posuda_reel.py SRC_DIR PHOTO_DIR WORK_DIR [--frames]
"""
import json, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import font, INK

W, H, FPS = 3840, 2160, 25
GAP = 0.5                                   # pause between narration files
SCRIPT = [
    "К первому прикорму в этой семье начали готовиться заранее. Причём мама, кажется, ждала этого момента даже больше, чем сам ребёнок. И, конечно же, первым делом она решила накупить красивой посуды.",
    "В наборе оказалось всё необходимое: две ложечки, вилочка, тарелка с секциями, глубокая тарелочка, ниблер и поильник. Обе тарелки на присосках, чтобы крепить их к столику для кормления.",
    "Хотя мама прекрасно понимала, что рано или поздно еда всё равно окажется где угодно, только не во рту ребёнка. Тарелка с секциями предназначалась для знакомства с разными продуктами, а глубокая для каш и пюре.",
    "Ниблер тоже ждал своего часа, как и поильник, из которого маленькой хозяйке предстояло научиться пить. Но больше всего маму, конечно, радовал цвет. Ведь даже детская посуда должна была вписываться в её представления о прекрасном.",
    "А тем временем близкие решили, что одного набора будет недостаточно, и подарили ещё одну тарелочку в виде сердечка, кружечку, ложечку и красивую подстилку. Таким образом, к пяти месяцам у маленькой девочки уже собралась собственная коллекция посуды.",
    "Оставалось выяснить лишь одно: оценит ли всё это сама хозяйка? Впрочем, маму этот вопрос пока волновал меньше всего. Ведь она уже была в полном восторге от покупок.",
]
# (start on the timeline, clip or photo layout, in-point)
SHOTS = [
    (0.0, "C0934", 0.3), (4.6, "C0936", 0.0), (7.6, "C0922", 0.0),
    (10.3, "C0914", 3.0), (12.7, "C0914", 15.5), (15.1, "C0914", 36.0),
    (17.25, "C0932", 0.0), (20.2, "C0926", 0.0), (22.9, "C0928", 8.0), (24.75, "C0919", 4.6),
    (26.5, "C0941", 0.5), (27.6, "C0917", 11.2), (29.1, "C0919", 17.0), (32.1, "C0928", 13.3),
    (35.05, "C0929", 1.0), (39.25, "C0930", 0.3), (41.75, "C0931", 0.0),
    (43.8, "C0928", 1.5), (46.3, "C0928", 23.0), (48.8, "C0919", 0.5), (51.1, "C0919", 11.0),
    (53.4, "C0939", 1.0), (55.0, "C0940", 37.5), (56.6, "C0915", 1.0), (59.4, "C0917", 3.5),
    (62.3, "C0933", 0.0), (65.2, "C0937", 0.0), (68.0, "C0938", 0.0), (70.8, "C0927", 7.0),
    (74.4, "P:set", 0), (81.9, "P:plate|cupside", 0), (84.9, "P:cuptop|spoon", 0), (87.4, "P:mat", 0),
    (90.0, "C0934", 5.2), (94.0, "C0923", 0.0),
    (99.0, "C0940", 27.0), (102.2, "C0941", 2.5), (105.2, "C0924", 0.0), (107.5, "C0925", 0.0),
    (110.3, "C0935", 0.0),
]
END = 115.1
PILLS = [                                   # things the voice does not say
    (10.6, 17.0, "набор Kuchertim"),
    (20.4, 22.8, "+ металлическая ложечка"),
    (35.5, 43.6, "+ силиконовый нагрудник с кармашком"),
    (46.5, 53.2, "тарелки с крышками"),
    (53.6, 56.5, "ниблер edda"),
    (59.6, 62.2, "поильник с трубочкой и грузиком"),
    (71.0, 74.3, "+ прорезыватель-звёздочка"),
    (76.0, 89.8, "подарок от близких"),
]
TITLE = (0.3, 4.4, "первая посуда", "для прикорма")

def offsets(durs):
    o, t = [], 0.0
    for d in durs: o.append(t); t += d + GAP
    return o

def cues(words, offs):
    """script words on whisper timings; phrases break at punctuation or ~38 chars"""
    out = []
    for k, text in enumerate(SCRIPT):
        toks, ws = text.split(), words[str(k + 1)]
        assert len(toks) == len(ws), (k, len(toks), len(ws))
        cur, st = [], None
        for i, (t, w) in enumerate(zip(toks, ws)):
            if st is None: st = w[1]
            cur.append(t); line = " ".join(cur)
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            weak = len(t) <= 3 and t[-1].isalpha()           # never leave «для», «о», «у», «от» hanging at a line end
            brk = nxt is None or t[-1] in ".?!:" or (t[-1] in ",;" and len(line) > 14) or \
                  (len(line + " " + (nxt or "")) > 38 and not weak) or len(line) > 46
            if brk:
                out.append([offs[k] + st, offs[k] + w[2], line.rstrip(",:;")]); cur, st = [], None
    for a, b in zip(out, out[1:]):                    # hold each phrase until the next one (if the pause is short)
        if b[0] - a[1] < 0.6: a[1] = b[0]
        else: a[1] += 0.25
    out[-1][1] += 0.6
    return out

def text_png(path, lines, size, role, y, col=(255, 255, 255), shadow=True):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sh = Image.new("L", (W, H), 0)
    d, ds = ImageDraw.Draw(img), ImageDraw.Draw(sh); f = font(role, size)
    for l in lines:
        d.text((W / 2, y), l, font=f, fill=col + (255,), anchor="mm")
        ds.text((W / 2, y + 5), l, font=f, fill=190, anchor="mm"); y += size * 1.3
    if shadow:
        s = Image.new("RGBA", (W, H), (20, 14, 12, 0)); s.putalpha(sh.filter(ImageFilter.GaussianBlur(14)))
        img = Image.alpha_composite(s, img)
    img.save(path)

def pill_png(path, text):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); f = font("detail", 66)
    tw = f.getlength(text); w, h = tw + 100, 124; x0, y0 = 170, 170
    m = Image.new("L", (W, H), 0); ImageDraw.Draw(m).rounded_rectangle((x0, y0 + 14, x0 + w, y0 + h + 14), h // 2, fill=70)
    s = Image.new("RGBA", (W, H), (60, 40, 40, 0)); s.putalpha(m.filter(ImageFilter.GaussianBlur(26)))
    img = Image.alpha_composite(img, s); d = ImageDraw.Draw(img)
    d.rounded_rectangle((x0, y0, x0 + w, y0 + h), h // 2, fill=(255, 255, 255, 245))
    d.text((x0 + w / 2, y0 + h / 2 + 2), text, font=f, fill=INK + (255,), anchor="mm")
    img.save(path)

def spray_png(path):
    y = np.linspace(0, 1, H, dtype=np.float32)
    a = (np.clip((y - 0.62) / 0.38, 0, 1) ** 1.6 * 0.62 * 255).astype(np.uint8)
    img = np.zeros((H, W, 4), np.uint8); img[..., 3] = a[:, None]
    Image.fromarray(img, "RGBA").save(path)

def cover(ph, w, h, zoom=1.0, cx=0.5, cy=0.5):
    s = max(w / ph.width, h / ph.height) * zoom
    p = ph.resize((round(ph.width * s), round(ph.height * s)), Image.LANCZOS)
    x = int(np.clip(cx * p.width - w / 2, 0, p.width - w)); y = int(np.clip(cy * p.height - h / 2, 0, p.height - h))
    return p.crop((x, y, x + w, y + h))

def photo_frame(name, ph):
    if name == "set": return cover(ph["set"], W, H, 1.0, 0.47, 0.56)
    if name == "mat": return cover(ph["set"], W, H, 1.25, 0.40, 0.62)
    a, b = name.split("|"); img = Image.new("RGB", (W, H), (255, 255, 255)); g = 16
    img.paste(cover(ph[a], W // 2 - g // 2, H), (0, 0)); img.paste(cover(ph[b], W // 2 - g // 2, H), (W // 2 + g // 2, 0))
    return img

def run(cmd): subprocess.run(cmd, check=True)

def main(src, photos, work, frames_only=False):
    os.makedirs(work, exist_ok=True)
    ph = {k: Image.open(os.path.join(photos, f"{i}.png")).convert("RGB")
          for i, k in enumerate(["spoon", "plate", "sweater", "jacket", "set", "cupside", "cuptop"], 1)}
    starts = [s[0] for s in SHOTS] + [END]
    if frames_only:
        for i, (t, c, a) in enumerate(SHOTS):
            mid = (starts[i + 1] - t) / 2
            if c.startswith("P:"): photo_frame(c[2:], ph).resize((640, 360)).save(f"{work}/f{i:02d}.jpg")
            else: run(["ffmpeg", "-v", "error", "-y", "-ss", str(a + mid), "-i", f"{src}/{c}.MP4", "-frames:v", "1", "-vf", "scale=640:-1", f"{work}/f{i:02d}.jpg"])
        return
    # 1. shots → 4K intermediates (frame-exact durations)
    parts = []
    for i, (t, c, a) in enumerate(SHOTS):
        n = round(starts[i + 1] * FPS) - round(t * FPS); out = f"{work}/s{i:02d}.mp4"; parts.append(out)
        if os.path.exists(out): continue
        enc = ["-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an"]
        if c.startswith("P:"):
            p = f"{work}/p{i:02d}.png"; photo_frame(c[2:], ph).save(p)
            vf = f"scale=8*{W}:-1,zoompan=z='1+0.05*on/{n}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS}"
            run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", p, "-vf", vf, "-frames:v", str(n)] + enc + [out])
        else:
            run(["ffmpeg", "-v", "error", "-y", "-ss", str(a), "-i", f"{src}/{c}.MP4", "-frames:v", str(n), "-vf", f"fps={FPS},setsar=1"] + enc + [out])
    with open(f"{work}/list.txt", "w") as f:
        for p in parts: f.write(f"file '{os.path.abspath(p)}'\n")
    # 2. narration: the six files with short pauses, a tiny fade at each edge
    durs = [float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f"{src}/a{k}.mp3"])) for k in range(1, 7)]
    offs = offsets(durs)
    ins = sum([["-i", f"{src}/a{k}.mp3"] for k in range(1, 7)], [])
    fl = ";".join(f"[{k}:a]aresample=48000,afade=t=in:d=0.02,afade=t=out:st={durs[k]-0.06:.3f}:d=0.06,adelay={int(offs[k]*1000)}|{int(offs[k]*1000)}[v{k}]" for k in range(6))
    fl += ";" + "".join(f"[v{k}]" for k in range(6)) + f"amix=inputs=6:normalize=0,apad,atrim=0:{END},loudnorm=I=-15:TP=-1.5:LRA=9[a]"
    run(["ffmpeg", "-v", "error", "-y"] + ins + ["-filter_complex", fl, "-map", "[a]", "-ac", "2", "-c:a", "pcm_s16le", f"{work}/voice.wav"])
    # 3. overlays: spray, subtitles, pills, title
    words = json.load(open(f"{work}/words.json")); ov = []
    spray_png(f"{work}/spray.png")
    for j, (a, b, t) in enumerate(cues(words, offs)):
        p = f"{work}/sub{j:03d}.png"; text_png(p, [t], 84, "second", H - 230); ov.append((a, b, p))
    for j, (a, b, t) in enumerate(PILLS):
        p = f"{work}/pill{j}.png"; pill_png(p, t); ov.append((a, b, p))
    a, b, t1, t2 = TITLE; p = f"{work}/title.png"
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    d.text((240, 330), t1, font=font("main", 120), fill=INK + (255,), anchor="lm")             # the wall left of the flowers
    d.text((244, 480), t2, font=font("detail", 92), fill=INK + (255,), anchor="lm"); img.save(p); ov.append((a, b, p))
    # one RGBA overlay track: a composed PNG per interval where the set of active layers is constant (low memory)
    spray = Image.open(f"{work}/spray.png"); cuts = sorted({0.0, END} | {round(x, 2) for a, b, _ in ov for x in (a, b) if x < END})
    lst = []
    for k, (t0, t1) in enumerate(zip(cuts, cuts[1:])):
        fr = spray.copy()
        for a, b, p in ov:
            if a <= t0 < b: fr = Image.alpha_composite(fr, Image.open(p))
        q = f"{work}/ov{k:03d}.png"; fr.save(q); lst.append((q, t1 - t0))
    with open(f"{work}/ov.txt", "w") as f:
        for q, d in lst: f.write(f"file '{os.path.abspath(q)}'\nduration {d:.3f}\n")
        f.write(f"file '{os.path.abspath(lst[-1][0])}'\n")
    ins = ["-f", "concat", "-safe", "0", "-i", f"{work}/list.txt", "-i", f"{work}/voice.wav",
           "-f", "concat", "-safe", "0", "-i", f"{work}/ov.txt"]
    fl = (f"[2:v]fps={FPS},format=rgba[o];[0:v][o]overlay=0:0:eof_action=repeat,"
          f"fade=t=in:d=0.5,fade=t=out:st={END-1.0}:d=1.0,format=yuv420p[v]")
    run(["ffmpeg", "-v", "error", "-y"] + ins + ["-filter_complex", fl, "-map", "[v]", "-map", "1:a",
         "-t", str(END), "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-profile:v", "high", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", f"{work}/posuda_4k.mp4"])

if __name__ == "__main__":
    main(*sys.argv[1:4], frames_only="--frames" in sys.argv)
