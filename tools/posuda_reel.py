"""@mcg.va — «первая посуда для прикорма», minimal review edit, 16:9 like the source (3840x2160, 25 fps).
Voice-over = the six narration mp3s in script order; shots cut to the words (faster-whisper timings in words.json).
Subtitles: thin Montserrat (wght 300), white with a soft glyph shadow over a long black spray at the very bottom.
Pieces the narration does not name get small pills (Montserrat Italic, INK on white).
Dynamic cut: long shots are split into ~1.4 s jump cuts, hand footage runs up to 1.35x; steady pieces get slow eased\npush-ins / pull-outs, in-points avoid the moments the camera is being re-aimed (motion.json, phase correlation).
The relatives' gifts: the video freezes, the still blurs, and the gift photos snap in one by one (flash + shutter)
with a heading and a label under each card.
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
SHOTS = [                                   # (start, clip, in-point[, speed]) — in-points sit in steady stretches
    (0.0, "C0934", 0.3), (4.6, "C0936", 0.0), (7.6, "C0922", 0.0),
    (10.3, "C0914", 2.4, 1.0), (12.7, "C0914", 19.6, 1.3), (15.1, "C0914", 41.2, 0.85),
    (17.25, "G:set3", 0), (20.2, "C0926", 0.0), (22.9, "C0928", 5.0, 1.3), (24.75, "C0919", 3.4, 1.3),
    (26.5, "G:nib2", 0), (29.1, "C0919", 10.2, 1.0), (31.3, "C0919", 13.6, 1.0), (33.5, "C0928", 19.4, 1.3),
    (35.05, "C0929", 1.0), (39.25, "C0930", 0.3), (41.75, "C0931", 0.0),
    (43.8, "C0928", 1.5), (46.3, "C0928", 23.0), (48.8, "C0919", 0.5), (51.1, "C0919", 24.0, 1.1),
    (53.4, "C0939", 7.2, 1.0), (55.0, "C0940", 37.5), (56.6, "C0915", 24.0, 1.0), (59.4, "C0917", 3.6, 0.9),
    (62.3, "C0929", 16.0, 1.0), (65.2, "C0937", 0.0), (68.0, "C0938", 0.0), (70.8, "C0927", 9.0, 1.0), (73.2, "C0937", 0.6, 1.0),
    (74.4, "C0921", 16.0), (76.7, "C0934", 2.0), (79.0, "F:C0934", 4.3),
    (90.0, "G:grid4", 0), (94.0, "C0923", 0.0),
    (99.0, "C0940", 21.0, 0.9), (102.2, "C0941", 2.5), (105.2, "C0921", 6.0, 1.0), (107.5, "C0925", 0.0),
    (110.3, "C0935", 0.0),
]
END = 115.1
# video collages: panels (clip, in-point, speed, appears at) in a layout of (x, y, w, h) boxes over a blurred backdrop
COLLAGES = {
    "set3": ([("C0932", 0.0, 1.0, 0.0), ("C0929", 13.6, 1.0, 0.45), ("C0915", 17.0, 1.0, 0.9)],
             [(160, 170, 1120, 1560), (1360, 170, 1120, 1560), (2560, 170, 1120, 1560)]),
    "nib2": ([("C0941", 0.5, 1.0, 0.0), ("C0917", 11.2, 1.0, 1.15)],
             [(180, 170, 1700, 1560), (1960, 170, 1700, 1560)]),
    "grid4": ([("C0934", 5.2, 1.0, 0.0), ("C0923", 0.0, 1.0, 0.35), ("C0929", 19.4, 1.0, 0.7), ("C0937", 0.0, 0.7, 1.05)],
              [(190, 110, 1700, 860), (1950, 110, 1700, 860), (190, 1030, 1700, 860), (1950, 1030, 1700, 860)]),
}
PILLS = [                                   # things the voice does not say
    (10.6, 17.0, "набор Kuchertim"),
    (20.3, 22.85, "ошибочка: ложечек три — силиконовая, прозрачная и металлическая"),
    (35.5, 43.6, "+ силиконовый нагрудник с кармашком"),
    (46.5, 53.2, "тарелки с крышками"),
    (53.6, 56.5, "ниблер edda"),
    (59.6, 62.2, "поильник с трубочкой и грузиком"),
    (71.0, 74.3, "+ прорезыватель-звёздочка"),
] 
FREEZE = (79.0, 90.0)
GIFTS = [                                   # (snap time, photo, label, label time, card size, centre, tilt)
    (81.05, "set", "подстилка", 87.6, (1100, 825), (3000, 1000), 5),
    (82.41, "plate", "тарелочка-сердечко", 82.41, (720, 960), (680, 1000), -6),
    (84.95, "cupside", "кружечка", 84.95, (720, 960), (1370, 960), 4),
    (86.29, "spoon", "ложечка", 86.29, (720, 960), (2060, 1030), -3),
]
SNAPS = [g[0] for g in GIFTS]
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

def collage(src, work, name, d, out):
    """video panels with white rounded frames sliding up one after another over a blurred, dimmed backdrop"""
    panels, boxes = COLLAGES[name]; n = round(d * FPS); ins, fl = [], []
    for k, ((c, a, sp, _), (x, y, w, h)) in enumerate(zip(panels, boxes)):
        ins += ["-ss", f"{a:.3f}", "-t", f"{d * sp + 0.2:.3f}", "-i", f"{src}/{c}.MP4"]
        m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), 44, fill=255); m.save(f"{work}/cm_{w}x{h}.png")
        fr = Image.new("RGBA", (w + 28, h + 28), (0, 0, 0, 0)); ImageDraw.Draw(fr).rounded_rectangle((0, 0, w + 27, h + 27), 56, fill=(255, 255, 255, 255))
        fr.save(f"{work}/cf_{w}x{h}.png")
    for k, (_, _, _, _) in enumerate(panels):
        w, h = boxes[k][2:]; ins += ["-loop", "1", "-i", f"{work}/cm_{w}x{h}.png", "-loop", "1", "-i", f"{work}/cf_{w}x{h}.png"]
    np_ = len(panels)
    fl.append(f"[0:v]setpts=PTS/{panels[0][2]},fps={FPS},scale=480:270,boxblur=18:2,scale={W}:{H},eq=brightness=-0.10:saturation=0.9,setsar=1[b0]")
    for k, ((c, a, sp, at), (x, y, w, h)) in enumerate(zip(panels, boxes)):
        mi, fi = np_ + 2 * k, np_ + 2 * k + 1
        fl.append(f"[{k}:v]setpts=(PTS-STARTPTS)/{sp},fps={FPS},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},format=rgba[v{k}]")
        fl.append(f"[{mi}:v]format=gray,fps={FPS}[m{k}];[v{k}][m{k}]alphamerge[r{k}]")
        fl.append(f"[{fi}:v]fps={FPS},format=rgba[f{k}]")
        yy = f"'{y}+90*pow(max(0,1-(t-{at})/0.35),3)'"
        fl.append(f"[b{k}][f{k}]overlay=x={x - 14}:y={yy.replace(str(y) + '+', str(y - 14) + '+')}:enable='gte(t,{at})':shortest=1[c{k}]")
        fl.append(f"[c{k}][r{k}]overlay=x={x}:y={yy}:enable='gte(t,{at})':shortest=1[b{k + 1}]")
    fl.append(f"[b{np_}]format=yuv420p[o]")
    run(["ffmpeg", "-v", "error", "-y"] + ins + ["-filter_complex", ";".join(fl), "-map", "[o]", "-frames:v", str(n),
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-r", str(FPS), "-an", out])

_len = {}
def clip_len(src, c):
    if c not in _len:
        _len[c] = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f"{src}/{c}.MP4"]))
    return _len[c]

ZOOM = False                                # push-ins / pull-outs: switched off at the blog's request
ZOOM_STEADY = ("in", 1.0, "out", 1.0)

def steady(src, c, a, dur):
    """mean camera shift (px per 0.2 s at 192 px wide) over a source window, from motion.json if present"""
    p = os.path.join(src, "..", "motion.json")
    if not os.path.exists(p): return 99
    m = np.array(json.load(open(p))[c]); seg = m[int(a * 5):max(int(a * 5) + 1, int((a + dur) * 5)), 0]
    return float(seg.mean()) if len(seg) else 99

def piece_file(work, p):
    """intermediate named by its content, so edits to SHOTS never reuse a stale file"""
    t, d, c, a, sp, z = p
    return f"{work}/p_{t:07.3f}_{d:.3f}_{c.replace(':', '-').replace('|', '+')}_{a:.3f}_{sp:.4f}_{z}.mp4"

def pieces(src):
    """SHOTS → (start, dur, clip, in, speed, zoom). Long shots become ~1.4 s jump cuts, hand footage runs up to 1.35x.
    zoom (only with ZOOM): steady pieces take turns with a slow eased push-in / pull-out ("in" / "out")"""
    starts = [s[0] for s in SHOTS] + [END]; out = []; cyc = 0
    for i, sh in enumerate(SHOTS):
        t, c, a = sh[:3]; d = starts[i + 1] - t
        if c[:2] in ("F:", "G:"): out.append((t, d, c, a, 1.0, 1.0)); continue
        sp = sh[3] if len(sh) > 3 else float(np.clip((clip_len(src, c) - a - 0.12) / d, 0.8, 1.35))   # short clips slow down a touch rather than run out
        k = max(1, round(d / 1.4)) if d > 2.2 else 1
        for j in range(k):
            tj = t + d * j / k; aj = a + (tj - t) * sp
            out.append((tj, d / k, c, aj, sp, ZOOM_STEADY[cyc % 4] if ZOOM and steady(src, c, aj, d / k * sp) < 1.2 else 1.0))
            cyc += ZOOM and steady(src, c, aj, d / k * sp) < 1.2
    return out

def polaroid(ph, size, tilt):
    """photo card with a white border, rotated; returns (card RGBA, shadow L)"""
    w, h = size; b = 30; im = Image.new("RGBA", (w + 2 * b, h + 2 * b), (255, 255, 255, 255))
    im.paste(cover(ph, w, h), (b, b)); m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, im.width - 1, im.height - 1), 18, fill=255); im.putalpha(m)
    im = im.rotate(tilt, Image.BICUBIC, expand=True)
    sh = im.getchannel("A").point(lambda v: int(v * 0.45)).filter(ImageFilter.GaussianBlur(34))
    return im, sh

def fade(img, a):
    if a >= 1: return img
    img = img.copy(); img.putalpha(img.getchannel("A").point(lambda v: int(v * max(a, 0)))); return img

def freeze(src, ph, clip, at, t0, t1, out):
    """still of the last frame, blur ramps in, gift cards snap in one by one with a flash, heading + labels"""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", f"{src}/{clip}.MP4", "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    still = Image.frombytes("RGB", (W, H), raw)
    blurs = [still] + [Image.fromarray((np.asarray(still.filter(ImageFilter.GaussianBlur(4 * k)), np.float32) * (1 - 0.018 * k))
                       .astype(np.uint8)) for k in range(1, 11)]
    cards = [(g, *polaroid(ph[g[1]], g[4], g[6])) for g in GIFTS]
    text_png(f"{out}.head.png", ["подарок от близких"], 116, "main", 250)
    head = Image.open(f"{out}.head.png")
    labels = []
    for g, im, _ in cards:
        q = f"{out}.lab{len(labels)}.png"; lab = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        f = font("detail", 70); cx, cy = g[5]; y = cy + im.height // 2 + 20
        sh = Image.new("L", (W, H), 0); ImageDraw.Draw(sh).text((cx, y + 4), g[2], font=f, fill=200, anchor="mm")
        s2 = Image.new("RGBA", (W, H), (20, 14, 12, 0)); s2.putalpha(sh.filter(ImageFilter.GaussianBlur(12)))
        lab = Image.alpha_composite(lab, s2); ImageDraw.Draw(lab).text((cx, y), g[2], font=f, fill=(255, 255, 255, 255), anchor="mm")
        labels.append(lab)
    n = round(t1 * FPS) - round(t0 * FPS)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    ease = lambda x: 1 - (1 - min(max(x, 0), 1)) ** 3
    for k in range(n):
        t = t0 + k / FPS; lt = t - t0
        fr = blurs[min(10, int(lt / 0.5 * 10))].convert("RGBA")
        if lt > 0.2: fr = Image.alpha_composite(fr, fade(head, (lt - 0.2) / 0.4))
        flash = 0.0
        for (g, im, sh), lab in zip(cards, labels):
            dt = t - g[0]
            if dt < 0: continue
            z = 1.22 - 0.22 * ease(dt / 0.16); a = min(1, dt / 0.05)
            ci, cs = im, sh
            if z != 1.0:
                ci = im.resize((round(im.width * z), round(im.height * z)), Image.BILINEAR)
                cs = sh.resize(ci.size, Image.BILINEAR)
            ci = fade(ci, a)
            cx, cy = g[5]; x, y = cx - ci.width // 2, cy - ci.height // 2
            fr.paste((40, 25, 25, 255), (x + 10, y + 26), cs)
            fr.alpha_composite(ci, (max(x, 0), max(y, 0)), (max(-x, 0), max(-y, 0)))
            flash = max(flash, 0.45 * max(0, 1 - dt / 0.12))
            lt2 = t - g[3]
            if lt2 >= 0: fr = Image.alpha_composite(fr, fade(lab, lt2 / 0.3))
        fr = fr.convert("RGB")
        if flash > 0: fr = Image.blend(fr, Image.new("RGB", (W, H), (255, 255, 255)), flash)
        enc.stdin.write(fr.tobytes())
    enc.stdin.close(); enc.wait()

def main(src, photos, work, frames_only=False):
    os.makedirs(work, exist_ok=True)
    ph = {k: Image.open(os.path.join(photos, f"{i}.png")).convert("RGB")
          for i, k in enumerate(["spoon", "plate", "sweater", "jacket", "set", "cupside", "cuptop"], 1)}
    starts = [s[0] for s in SHOTS] + [END]
    if frames_only == "collages":
        for t, c, a in [x[:3] for x in SHOTS if x[1].startswith("G:")]:
            i = [s[0] for s in SHOTS].index(t); collage(src, work, c[2:], starts[i + 1] - t, f"{work}/test_{c[2:]}.mp4")
        return
    if frames_only == "freeze":
        (t, c, a), = [x for x in SHOTS if x[1].startswith("F:")]
        t0, t1 = FREEZE
        freeze(src, ph, c[2:], a, t0, t1, f"{work}/freeze_test.mp4"); return
    if frames_only:
        for i, (t, c, a) in enumerate(SHOTS):
            mid = (starts[i + 1] - t) / 2
            if c.startswith("P:"): photo_frame(c[2:], ph).resize((640, 360)).save(f"{work}/f{i:02d}.jpg")
            else: run(["ffmpeg", "-v", "error", "-y", "-ss", str(a + mid), "-i", f"{src}/{c}.MP4", "-frames:v", "1", "-vf", "scale=640:-1", f"{work}/f{i:02d}.jpg"])
        return
    # 1. shots → 4K intermediates (frame-exact durations)
    parts = []
    for i, pc in enumerate(pieces(src)):
        t, d, c, a, sp, z = pc
        n = round((t + d) * FPS) - round(t * FPS); out = piece_file(work, pc); parts.append(out)
        if os.path.exists(out): continue
        enc = ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an"]
        if c.startswith("F:"):
            freeze(src, ph, c[2:], a, t, t + d, out)
        elif c.startswith("G:"):
            collage(src, work, c[2:], (round((t + d) * FPS) - round(t * FPS)) / FPS, out)
        else:
            vf = f"setpts=PTS/{sp:.4f},fps={FPS}"
            if isinstance(z, str):                                    # slow eased push-in / pull-out
                e = f"(0.5-0.5*cos(PI*min(t/{d:.3f},1)))"; zz = f"(1+0.13*{e})" if z == "in" else f"(1.13-0.13*{e})"
                vf += f",scale=w='trunc({W}*{zz}/2)*2':h='trunc({H}*{zz}/2)*2':eval=frame:flags=bicubic,crop={W}:{H}"
            elif z != 1.0: vf += f",crop=iw/{z}:ih/{z},scale={W}:{H}:flags=lanczos"
            vf += ",setsar=1"
            run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", f"{src}/{c}.MP4", "-frames:v", str(n), "-vf", vf] + enc + [out])
    with open(f"{work}/list.txt", "w") as f:
        for p in parts: f.write(f"file '{os.path.abspath(p)}'\n")
    # 2. narration: the six files with short pauses, a tiny fade at each edge
    durs = [float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f"{src}/a{k}.mp3"])) for k in range(1, 7)]
    offs = offsets(durs)
    ins = sum([["-i", f"{src}/a{k}.mp3"] for k in range(1, 7)], [])
    fl = ";".join(f"[{k}:a]aresample=48000,afade=t=in:d=0.02,afade=t=out:st={durs[k]-0.06:.3f}:d=0.06,adelay={int(offs[k]*1000)}|{int(offs[k]*1000)}[v{k}]" for k in range(6))
    fl += ";" + "".join(f"[v{k}]" for k in range(6)) + f"amix=inputs=6:normalize=0,apad,atrim=0:{END},loudnorm=I=-15:TP=-1.5:LRA=9[a]"
    run(["ffmpeg", "-v", "error", "-y"] + ins + ["-filter_complex", fl, "-map", "[a]", "-ac", "2", "-c:a", "pcm_s16le", f"{work}/voice.wav"])
    # 3. frame-exact compositor: transitions, shrink moments, spray, word-by-word subtitles, pills, title
    import posuda_compose
    posuda_compose.render(work, offs, f"{work}/posuda_4k.mp4")

if __name__ == "__main__":
    main(*sys.argv[1:4], frames_only="freeze" if "--freeze-test" in sys.argv else "collages" if "--collage-test" in sys.argv else "--frames" in sys.argv)
