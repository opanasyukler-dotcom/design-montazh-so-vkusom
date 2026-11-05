"""5 Instagram stories (1080x1920): a patient who dreamed of the operation almost all her life (abdominoplasty,
breast lift / reduction, lipofilling) -> «не могу на себя насмотреться» 1.5 months later.

Everything is static: the text, the arrows and the zone outlines stand on the story from the first to the last frame
(no appearing, no motion effects); only the background video plays and the subtitles change with the words.
Inter Bold caps headlines + Montserrat; subtitles in thin Montserrat (300) with a soft shadow.  Faces blurred
(face_blur_track.py on the interview clips, a hand blur on the photos), the patient's voice gently disguised
(clean_speech.py --disguise), subtitles from subs_retext.py.

Clips (CLIPS dir, made from the Drive folder 1XCqfpzWMBTqMFHwtDn9tVY4e0d-tEEXR):
  a1 «я уже очень давно хотела…» (v09 = 2024-03-18 23-01-39.mp4 3.6-9.1 s)
  a2 «что не любим, всё показываем… ну вот так её поднимем» (v06 = …(28).mp4)
  a3 «всё буду делать, кроме лица, в этот раз…» (v05 = …(2).mp4), f3 its video with the face blurred
  a4 «очень рада, что решилась… это того стоит» (v15 = 2024-03-19 16-15-15 (1).mp4), f4 its video (5-18 s) blurred
  a5a «я очень-очень рада, не могу на себя насмотреться» (v14 = 2024-03-19 16-14-58 (3).mp4)
  a5b «мне очень всё нравится…» (v17 = 2024-03-19 16-15-15 (3).mp4), f5 the mirror video (16-30 s) blurred
usage: python3 stories_patient_dream.py DRIVE_DIR CLIPS_DIR OUT_DIR [story numbers...]
"""
import json, os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

W, H, FPS = 1080, 1920, 30
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
WHITE, GRAPHITE = (255, 255, 255), (20, 23, 31)
DRIVE, CLIPS, OUT = sys.argv[1:4]
ONLY = {int(a) for a in sys.argv[4:]}
os.makedirs(OUT, exist_ok=True)
MARGIN = 72


def mont(size, wght=400):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([wght]); return f


def bold(size): return ImageFont.truetype(BOLD, size)


def wrap(text, f, width):
    out, line = [], ""
    for w in text.split(" "):
        t = (line + " " + w).strip()
        if f.getlength(t) <= width or not line: line = t
        else: out.append(line); line = w
    return out + [line]


class Block:
    """a static text block: [(text, font, line step, gap after)] wrapped to a width"""
    def __init__(self, width=W - 2 * MARGIN): self.items, self.width = [], width

    def add(self, text, f, lh=1.2, gap=0):
        for i, l in enumerate(wrap(text, f, self.width)): self.items.append((l, f, int(f.size * lh)))
        if gap: self.items[-1] = self.items[-1][:2] + (self.items[-1][2] + gap,)
        return self

    def head(self, text, size, gap=14): return self.add(text, bold(size), 1.14, gap)

    def body(self, text, size, wght=400, gap=10): return self.add(text, mont(size, wght), 1.3, gap)

    @property
    def h(self): return sum(s for _, _, s in self.items[:-1]) + int(self.items[-1][1].size * 1.15)

    def draw(self, lay, x, y, anchor="la"):
        d = ImageDraw.Draw(lay)
        for t, f, s in self.items:
            xx = x if anchor == "la" else x + self.width // 2
            d.text((xx, y), t, font=f, fill=WHITE + (255,), anchor=anchor); y += s
        return y


def with_shadow(lay, a=150, blur=8):
    al = lay.getchannel("A")
    sh = Image.new("RGBA", lay.size, GRAPHITE + (0,)); sh.putalpha(al.point(lambda v: v * a // 255).filter(ImageFilter.GaussianBlur(blur)))
    sh2 = Image.new("RGBA", lay.size, GRAPHITE + (0,)); sh2.putalpha(al.point(lambda v: v * 120 // 255).filter(ImageFilter.GaussianBlur(2)))
    return Image.alpha_composite(Image.alpha_composite(sh, sh2), lay)


def vfade(y0, y1, a):
    """black: 0 at y0 -> a at y1 (y1 < y0 fades upwards)"""
    ys = np.arange(H, dtype=np.float32)[:, None, None]
    return a * np.clip((ys - y0) / (y1 - y0), 0, 1) ** 1.3


# ---------------- subtitles ----------------
def cues(clip, t0, maxc=30):
    ws = json.load(open(os.path.join(CLIPS, clip + ".mp4.subs.json")))[0]["w"]
    out, cur = [], []
    for i, w in enumerate(ws):
        cur.append(w); txt = " ".join(x[2] for x in cur); nxt = ws[i + 1] if i + 1 < len(ws) else None
        if nxt is None or len(txt) + 1 + len(nxt[2]) > maxc or w[2][-1] in ".?!…" or (w[2][-1] == "," and len(txt) > 16):
            out.append(cur); cur = []
    res = []
    for i, c in enumerate(out):
        end = out[i + 1][0][0] if i + 1 < len(out) else c[-1][1] + 0.8
        res.append((t0 + c[0][0] - 0.05, t0 + min(end, c[-1][1] + 0.9), " ".join(x[2] for x in c)))
    return res


def sub_sprite(text, size=44):
    f = mont(size, 300); lines = wrap(text, f, W - 2 * MARGIN - 40)
    step = int(size * 1.3); lay = Image.new("RGBA", (W, step * len(lines) + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for i, l in enumerate(lines): d.text((W // 2, 20 + i * step), l, font=f, fill=WHITE + (255,), anchor="ma")
    return np.asarray(with_shadow(lay, 190, 6), np.float32) / 255


class Subs:
    def __init__(self, cl, y): self.c = [(a, b, sub_sprite(t)) for a, b, t in cl]; self.y = y

    def draw(self, fr, t):
        for a, b, sp in self.c:
            if a <= t < b:
                h = sp.shape[0]; y = self.y - h // 2; al = sp[..., 3:4]
                fr[y:y + h] = fr[y:y + h] * (1 - al) + sp[..., :3] * al


# ---------------- backgrounds ----------------
class Video:
    """a clip scaled to cover 1080x1920 at 30 fps (optionally slowed), from `ss`; the last frame holds"""
    def __init__(self, path, ss=0.0, speed=1.0, top=0):
        vf = f"setpts=PTS/{speed},fps={FPS},scale={W}:-2,crop={W}:{H}:0:{top}" if top else f"setpts=PTS/{speed},fps={FPS},scale={W}:{H}"
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(ss), "-i", path, "-an", "-vf", vf, "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.last = None

    def frame(self):
        b = self.p.stdout.read(W * H * 3)
        if len(b) == W * H * 3: self.last = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        return self.last.astype(np.float32) / 255


def photo(name):
    return ImageOps.exif_transpose(Image.open(os.path.join(DRIVE, name))).convert("RGB")


def blur_zone(im, box, r=40):
    """a hand blur for the photos (the chin / lips at the top edge): soft-edged ellipse"""
    m = Image.new("L", im.size, 0); ImageDraw.Draw(m).ellipse(box, fill=255); m = m.filter(ImageFilter.GaussianBlur(r // 2))
    return Image.composite(im.filter(ImageFilter.GaussianBlur(r)), im, m)


def render(name, dur, bg, overlay, shade=None, subs=None, audio=()):
    """bg(t) -> float frame; overlay: the static RGBA layer (text, arrows); audio: [(clip mp4, start)]"""
    out = os.path.join(OUT, name); tmp = out + ".v.mp4"
    ov = np.asarray(overlay, np.float32) / 255; oa = ov[..., 3:4]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for i in range(int(round(dur * FPS))):
        t = i / FPS; fr = bg(t)
        if shade is not None: fr = fr * (1 - shade)
        fr = fr * (1 - oa) + ov[..., :3] * oa
        if subs: subs.draw(fr, t)
        o = (np.clip(fr, 0, 1) * 255).astype(np.uint8); enc.stdin.write(o.tobytes())
        if i == int(dur * FPS * 0.45): Image.fromarray(o).save(os.path.join(OUT, "prev_" + name[:-4] + ".jpg"), quality=88)
    enc.stdin.close(); enc.wait()
    ins = []; flt = []
    for k, (clip, st) in enumerate(audio):
        ins += ["-i", os.path.join(CLIPS, clip + ".mp4")]
        flt.append(f"[{k + 1}:a]adelay={int(st * 1000)}:all=1[a{k}]")
    mix = "".join(f"[a{k}]" for k in range(len(audio)))
    flt.append(f"{mix}amix=inputs={len(audio)}:normalize=0,loudnorm=I=-16:TP=-1.5,apad,atrim=0:{dur}[aout]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp] + ins + ["-filter_complex", ";".join(flt), "-map", "0:v", "-map", "[aout]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out], check=True)
    os.remove(tmp); print("wrote", out, flush=True)


# ---------------- arrows, zone outlines ----------------
def arrow(d, p0, p1, sc, width=3, head=16):
    (x0, y0), (x1, y1) = p0, p1; ang = np.arctan2(y1 - y0, x1 - x0)
    d.line([(x0 * sc, y0 * sc), ((x1 - head * 0.6 * np.cos(ang)) * sc, (y1 - head * 0.6 * np.sin(ang)) * sc)], fill=WHITE + (255,), width=width * sc)
    for s in (-0.45, 0.45):
        d.line([(x1 * sc, y1 * sc), ((x1 - head * np.cos(ang + s)) * sc, (y1 - head * np.sin(ang + s)) * sc)], fill=WHITE + (255,), width=width * sc)


def dashed_ellipse(d, box, sc, width=2, n=34, fill=0.55):
    """thin dashed outline of a zone: the static «highlight» of a problem area"""
    x0, y0, x1, y1 = box; cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    for k in range(n):
        a0, a1 = 2 * np.pi * k / n, 2 * np.pi * (k + fill) / n
        ts = np.linspace(a0, a1, 8)
        d.line([((cx + rx * np.cos(t)) * sc, (cy + ry * np.sin(t)) * sc) for t in ts], fill=WHITE + (235,), width=width * sc)


def zones_layer(items):
    """items: ("label", label xy, anchor, arrow from, arrow to, ellipse box or None)"""
    sc = 3; big = Image.new("RGBA", (W * sc, H * sc), (0, 0, 0, 0)); d = ImageDraw.Draw(big)
    for _, _, _, p0, p1, box in items:
        if box: dashed_ellipse(d, box, sc)
        arrow(d, p0, p1, sc)
    lay = big.resize((W, H), Image.LANCZOS); dl = ImageDraw.Draw(lay)
    for lab, xy, anc, *_ in items: dl.text(xy, lab, font=mont(34, 500), fill=WHITE + (255,), anchor=anc)
    return lay


# ---------------- stories ----------------
def s1():
    """«я давно хочу что-то изменить»: the before photo from the side (full height), her own words, poll room"""
    im = blur_zone(photo("2024-03-18 23-01-39 (23).jpeg"), (470, -120, 1060, 330), 46)
    im = im.resize((W, int(im.height * W / im.width)), Image.LANCZOS)          # 1080 x 1440, the whole body, feet in
    cv = Image.new("RGB", (W, H), (0, 0, 0)); cv.paste(im, (0, H - im.height))
    a = np.asarray(cv, np.float32) / 255                                       # the photo's top is the black backdrop already
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    b = (Block().head("«Я ДАВНО ХОЧУ ЧТО-ТО ИЗМЕНИТЬ. НО ВСЁ ОТКЛАДЫВАЮ…»", 58, gap=18)
         .body("У этой пациентки мысль об операции была с ней много лет.", 32, wght=400, gap=0))
    b.draw(lay, MARGIN, 96)
    lay = with_shadow(lay)
    render("01_davno_hochu.mp4", 9.0, lambda t: a, lay,
           subs=Subs(cues("a1", 0.5), 446), audio=[("a1", 0.5)])
    # the poll sticker: lower right on the black backdrop, beside the hip (x 700-1050, y 1100-1650)


def s2():
    """the zones she wanted to change: the before photo from the front, neck to fingertips, thin arrows + dashed outlines"""
    src = blur_zone(photo("2024-03-18 23-01-39 (24).jpeg"), (560, -140, 1000, 120), 40)
    CX0, CY0 = 100, 284                                                         # neck to the fingertips, the wall edge out
    crop = src.crop((CX0, CY0, 1436, 1479)); k = W / crop.width; ph = crop.resize((W, int(crop.height * k)), Image.LANCZOS)
    Y0 = 430
    m = Image.new("L", ph.size, 255); dm = ImageDraw.Draw(m)
    for j in range(40):                                                         # feathered edges into the black
        v = int(255 * (j / 40) ** 1.5); dm.rectangle((j, j, ph.width - 1 - j, ph.height - 1 - j), outline=v)
    cv = Image.new("RGB", (W, H), (0, 0, 0)); cv.paste(ph, (0, Y0), m)
    # zone points: preview (540x720) coords -> source (x2.844) -> the story
    P = lambda x, y: ((x * 2.844 - CX0) * k, (y * 2.844 - CY0) * k + Y0)
    zl = zones_layer([
        ("Грудь", (W - 60, P(0, 196)[1]), "rs", (W - 150, P(0, 206)[1]), P(378, 238), (P(180, 205)[0], P(0, 205)[1], P(380, 0)[0], P(0, 290)[1])),
        ("Живот", (60, P(0, 395)[1]), "ls", (150, P(0, 405)[1]), P(222, 352), (P(205, 0)[0], P(0, 300)[1], P(355, 0)[0], P(0, 385)[1])),
        ("Бока", (W - 60, P(0, 365)[1]), "rs", (W - 140, P(0, 375)[1]), P(372, 330), None),
    ])
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    Block().body("После рождения ребёнка её стали беспокоить сразу несколько зон.", 36, wght=500).draw(lay, MARGIN, 90)
    b2 = (Block().body("И уже не хотелось «исправить что-то одно».", 32, gap=10)
          .body("Хотелось снова посмотреть на своё тело и почувствовать:", 32, gap=14)
          .head("«ВОТ ТЕПЕРЬ МНЕ НРАВИТСЯ»", 54, gap=0))
    b2.draw(lay, MARGIN, Y0 + ph.height + 60)
    lay = with_shadow(Image.alpha_composite(lay, zl))
    a = np.asarray(cv, np.float32) / 255
    render("02_zony.mp4", 10.0, lambda t: a, lay, subs=Subs(cues("a2", 0.6), 330), audio=[("a2", 0.6)])


def s3():
    """«всё буду делать, кроме лица»: her interview (face blurred), then the marking before the operation"""
    v1 = Video(os.path.join(CLIPS, "f3.mp4")); v2 = Video(os.path.join(DRIVE, "2024-03-18 23-01-39 (3).mp4"), speed=0.62)
    cut = 3.4; st = {}

    def bg(t):
        if t < cut: return v1.frame()
        return v2.frame()
    b = (Block().head("«ВСЁ БУДУ ДЕЛАТЬ, КРОМЕ ЛИЦА. В ЭТОТ РАЗ»", 66, gap=20)
         .body("После осмотра составили план комплексной коррекции.", 32, gap=0))
    top = H - 110 - b.h
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(lay, MARGIN, top); lay = with_shadow(lay)
    shade = np.maximum(vfade(top - 260, top + 120, 0.8), vfade(420, 0, 0.45))
    render("03_krome_lica.mp4", 11.2, bg, lay, shade=shade, subs=Subs(cues("a3", 0.3), 190), audio=[("a3", 0.3)])


def s4():
    """a month later: the restrictions were there, but «очень рада… не жалею… это того стоит»"""
    v = Video(os.path.join(CLIPS, "f4.mp4"), ss=2.9)
    b = (Block().head("ПРОШЁЛ МЕСЯЦ", 62, gap=18)
         .body("Да, были бытовые ограничения.", 34, wght=500, gap=4)
         .body("Было неудобно наклоняться, двигаться, завязывать шнурки.", 32, gap=12)
         .body("Но сама пациентка говорит:", 32, gap=150)
         .head("«ЭТО ТОГО СТОИТ»", 62, gap=16)
         .body("восстановление у каждой пациентки проходит индивидуально", 26, wght=300, gap=0))
    top = H - 90 - b.h
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(lay, MARGIN, top); lay = with_shadow(lay)
    # the quote (subtitles) in the room left between «говорит:» and «это того стоит»
    ys = [y for y in np.cumsum([top] + [s for _, _, s in b.items])]
    sub_y = int(ys[5]) - 80
    render("04_ne_zhaleyu.mp4", 10.5, lambda t: v.frame(), lay, shade=vfade(top - 300, top + 80, 0.82),
           subs=Subs(cues("a4", 0.5), sub_y), audio=[("a4", 0.5)])


def s5():
    """«не могу на себя насмотреться»: the mirror after (face blurred), the invitation to the consultation"""
    v = Video(os.path.join(CLIPS, "f5.mp4"))
    hb = Block().head("«НЕ МОГУ НА СЕБЯ НАСМОТРЕТЬСЯ»", 62, gap=0)
    b = (Block().body("Если вы тоже годами думаете об операции, но не понимаете, что именно нужно делать и можно ли "
                      "скорректировать несколько зон за один этап, начните с консультации.", 30, gap=14)
         .body("Для записи напишите «КОНСУЛЬТАЦИЯ» в Direct.", 30, wght=600, gap=0))
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); hb.draw(lay, MARGIN, 90)
    top = H - 90 - b.h; b.draw(lay, MARGIN, top); lay = with_shadow(lay)
    shade = np.maximum(vfade(top - 220, top + 80, 0.82), vfade(90 + hb.h + 160, 60, 0.6))
    render("05_ne_mogu_nasmotretsya.mp4", 13.0, lambda t: v.frame(), lay, shade=shade,
           subs=Subs(cues("a5a", 0.4) + cues("a5b", 5.0), top - 90), audio=[("a5a", 0.4), ("a5b", 5.0)])


for k, f in [(1, s1), (2, s2), (3, s3), (4, s4), (5, s5)]:
    if not ONLY or k in ONLY: f()
