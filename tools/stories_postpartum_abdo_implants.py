"""7 Instagram stories (1080x1920): after childbirth — abdominoplasty with diastasis repair + anatomical implants 375 ml.

Old story design: Inter Bold caps headlines + Montserrat, brand navy + white, black fade under the text, static text,
no emojis; the text never lies on the torso: on the videos it sits low on the legs, the photos and the clinic's
collages sit on top of their black background with the text below.  Short thin white arrows with navy labels mark
what we see on the before photos.  Photo stories are JPG, the two video stories MP4.  Stories 1 and 3 keep free space for
the Instagram poll sticker.

Sources (drive/): IMG_8392 before front, IMG_8390 before 3/4, IMG_8391 after front, IMG_8394 after 3/4,
IMG_0337.MOV before (belly), IMG_0331.MP4 first day with the surgeon (voice cut to day1.mp4 by clean_speech.py,
subtitles day1.mp4.subs.json), 5223F1B9-…png operating room.

usage: python3 stories_postpartum_abdo_implants.py SRC_DIR OUT_DIR [story numbers...]
"""
import json, os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

W, H, FPS = 1080, 1920, 30
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
WHITE, NAVY, GRAPHITE = (255, 255, 255), (2, 31, 71), (20, 23, 31)
SRC, OUT = sys.argv[1], sys.argv[2]
ONLY = {int(a) for a in sys.argv[3:]}
DRIVE = os.path.join(SRC, "drive")
os.makedirs(OUT, exist_ok=True)
MARGIN = 84; TW = W - 2 * MARGIN


def mont(size, wght=400):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([wght]); return f


def bold(size): return ImageFont.truetype(BOLD, size)


def wrap(text, f, width):
    out = []
    for par in text.split("\n"):
        line = ""
        for w in par.split(" "):
            t = (line + " " + w).strip()
            if f.getlength(t) <= width or not line: line = t
            else: out.append(line); line = w
        out.append(line)
    return out


def photo(name):
    return ImageOps.exif_transpose(Image.open(os.path.join(DRIVE, name))).convert("RGB")


# ---------------- text block: laid out once, drawn onto an RGBA layer ----------------
HIGHLIGHT = False        # navy word markers («[[...]]») — off: the client asked for the usual look of the earlier stories


def tokens(text):
    """«[[слова]]» -> highlighted in the surgeon's navy (when HIGHLIGHT); returns [(word, highlighted)]"""
    out, hl = [], False
    for part in text.replace("\n", " \n ").split(" "):
        if not part: continue
        w = part
        if w.startswith("[["): hl = True; w = w[2:]
        end = "]]" in w
        if end: k = w.index("]]"); w = w[:k] + w[k + 2:]           # punctuation after «]]» stays with the word
        out.append((w, hl and HIGHLIGHT))
        if end: hl = False
    return out


def wrap_tokens(toks, f, width):
    lines, cur = [], []
    for w, hl in toks:
        if w == "\n": lines.append(cur); cur = []; continue
        t = " ".join(x for x, _ in cur + [(w, hl)])
        if f.getlength(t) <= width or not cur: cur.append((w, hl))
        else: lines.append(cur); cur = [(w, hl)]
    return lines + [cur]


class Block:
    """left-aligned stack of text written straight onto the picture: white with a tight dark shadow, key words on a
    navy marker (the surgeon's colour). items: text lines | plate | gap"""
    def __init__(self, width=TW): self.items = []; self.h = 0; self.width = width

    def _text(self, text, f, step, gap):
        lines = wrap_tokens(tokens(text), f, self.width - 20)
        self.items.append(("text", lines, f, step)); self.h += step * len(lines) + gap; self.items.append(("gap", gap)); return self

    def head(self, text, size=52, gap=14): return self._text(text, bold(size), int(size * 1.22), gap)

    def body(self, text, size=34, wght=400, gap=16, lh=1.32): return self._text(text, mont(size, wght), int(size * lh), gap)

    def plate(self, text, size=34, wght=500, gap=14):
        """navy rounded plate with white text (a key item of a list)"""
        f = mont(size, wght); lines = wrap(text, f, self.width - 56); step = int(size * 1.3)
        h = step * len(lines) + 34; w = int(max(f.getlength(l) for l in lines)) + 56
        self.items.append(("plate", lines, f, step, w, h)); self.h += h + gap; self.items.append(("gap", gap)); return self

    def draw(self, layer, x, y):
        d = ImageDraw.Draw(layer); txt = Image.new("RGBA", layer.size, (0, 0, 0, 0)); dt = ImageDraw.Draw(txt)
        marks = Image.new("RGBA", layer.size, (0, 0, 0, 0)); dm = ImageDraw.Draw(marks)
        for it in self.items:
            if it[0] == "gap": y += it[1]; continue
            if it[0] == "text":
                _, lines, f, step = it; sp = f.getlength(" ")
                for i, ln in enumerate(lines):
                    cy = y + i * step + step / 2; xx = x; runs = []
                    for w, hl in ln:
                        lw = f.getlength(w)
                        if hl:
                            if runs and runs[-1][1] >= xx - sp - 1: runs[-1][1] = xx + lw
                            else: runs.append([xx, xx + lw])
                        dt.text((xx, cy), w, font=f, fill=WHITE + (255,), anchor="lm"); xx += lw + sp
                    hh = f.size * 0.62
                    for a, b in runs: dm.rounded_rectangle((a - 10, cy - hh, b + 10, cy + hh), 10, fill=NAVY + (245,))
                y += step * len(lines)
            else:
                _, lines, f, step, w, h = it
                dm.rounded_rectangle((x - 4, y, x - 4 + w, y + h), 22, fill=NAVY + (245,))
                for i, ln in enumerate(lines): dt.text((x + 24, y + 17 + i * step + step / 2), ln, font=f, fill=WHITE + (255,), anchor="lm")
                y += h
        a = txt.getchannel("A")                                 # tight dark shadow: readable on skin without any haze
        for r, k in [(9, 150)]:
            sh = Image.new("RGBA", layer.size, GRAPHITE + (0,)); sh.putalpha(a.point(lambda v, k=k: v * k // 255))
            layer.alpha_composite(sh.filter(ImageFilter.GaussianBlur(r)))
        layer.alpha_composite(marks); layer.alpha_composite(txt)
        return y


def pill(text, size=30, wght=500):
    f = mont(size, wght); h = int(size * 1.8); w = int(f.getlength(text)) + 52
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), h // 2, fill=NAVY + (235,))
    d.text((w / 2, h / 2 + 1), text, font=f, fill=WHITE + (255,), anchor="mm"); return im


def fade(top, length=380, a=0.8, up=False):
    """black darkening: from `top` down (or, up=True, from `top` up to the frame edge)"""
    ys = np.arange(H, dtype=np.float32)
    p = np.clip((ys - (top - length)) / (length + 60), 0, 1) if not up else np.clip(((top + length) - ys) / (length + 60), 0, 1)
    p = p * p * (3 - 2 * p); return (a * p)[:, None, None]


def save_jpg(canvas, name):
    out = os.path.join(OUT, name); canvas.convert("RGB").save(out, quality=94); print("wrote", out, flush=True)


def place(canvas, img, box):
    """fit img into box (x0, y0, x1, y1) by height, centred in it; whatever sticks out of the box is cut"""
    x0, y0, x1, y1 = box; s = (y1 - y0) / img.height
    im = img.resize((max(1, int(img.width * s)), y1 - y0), Image.LANCZOS)
    panel = Image.new("RGB", (x1 - x0, y1 - y0), (0, 0, 0)); panel.paste(im, ((x1 - x0 - im.width) // 2, 0))
    canvas.paste(panel, (x0, y0))


# ---------------- video with a static overlay ----------------
def render_video(name, src, a, dur, overlay, shade, shift=0, subs=None, audio=None, on_frame=None, fill_k=0.6, mute=False):
    out = os.path.join(OUT, name); tmp = out + ".v.mp4"
    ov = np.asarray(overlay, np.float32) / 255.0; oa = ov[..., 3:4]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(dur), "-i", os.path.join(DRIVE, src),
                          "-vf", f"scale={W}:{H}:flags=lanczos,fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    n = int(round(dur * FPS)); last = None
    for i in range(n):
        buf = p.stdout.read(W * H * 3)
        if len(buf) == W * H * 3:
            last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            if on_frame: last = on_frame(last.copy(), i)
        z = 1.0 + 0.03 * i / n
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z); M[1, 2] += shift
        fr = cv2.warpAffine(last, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32) / 255
        if shift > 0:                                      # blurred stretch of the frame's own top above it
            top = int(M[1, 2]) + 1
            strip = cv2.resize(last[:H // 6], (W // 10, max(2, top // 10 + 1)), interpolation=cv2.INTER_AREA)
            fill = cv2.resize(cv2.GaussianBlur(strip, (0, 0), 3), (W, top + 120)).astype(np.float32) / 255 * fill_k
            g = np.clip((np.arange(top + 120) - top) / 120.0, 0, 1)[:, None, None]
            fr[:top + 120] = fill * (1 - g) + fr[:top + 120] * g
        fr = fr * (1 - shade); fr = fr * (1 - oa) + ov[..., :3] * oa
        if subs: subs.draw(fr, i / FPS)
        o = (np.clip(fr, 0, 1) * 255).astype(np.uint8); enc.stdin.write(o.tobytes())
        if i == min(n - 1, int(2.5 * FPS)): Image.fromarray(o).save(out[:-4] + ".jpg", quality=90)
    p.kill(); enc.stdin.close(); enc.wait()
    au = audio or (src, a)
    if mute:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c", "copy", "-an", "-movflags", "+faststart", out], check=True)
        os.remove(tmp); print("wrote", out, flush=True); return
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", str(au[1]), "-t", str(dur), "-i", os.path.join(DRIVE, au[0]),
                    "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-af", f"loudnorm=I=-16:TP=-1.5,afade=t=out:st={dur - 0.3:.2f}:d=0.3",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out], check=True)
    os.remove(tmp); print("wrote", out, flush=True)


class PillSubs:
    """one word at a time, lower case, Montserrat Light on a navy pill, small pop-in"""
    def __init__(self, words, y, size=44, outline=False):
        f = mont(size, 300); self.items = []
        for i, (a, b, t) in enumerate(words):
            t = t.lower().strip(".,!?«»…"); h = int(size * 1.7); w = int(f.getlength(t)) + 44
            im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
            d.rounded_rectangle((0, 0, w - 1, h - 1), h // 2, fill=NAVY + (235,), outline=WHITE + (255,) if outline else None, width=3)
            d.text((w / 2, h / 2 + 1), t, font=f, fill=WHITE + (255,), anchor="mm")
            e = min(words[i + 1][0] if i + 1 < len(words) else b + 0.5, b + 0.45)
            self.items.append((a, e, np.asarray(im, np.float32) / 255.0))
        self.y = y

    def draw(self, fr, t):
        for a, e, arr in self.items:
            if a - 0.03 <= t < e:
                k = min(1.0, (t - a + 0.03) / 0.12); sc = 0.85 + 0.15 * k
                a2 = cv2.resize(arr, None, fx=sc, fy=sc) if sc < 0.999 else arr
                h, w = a2.shape[:2]; x, y = (W - w) // 2, self.y - h // 2; al = a2[..., 3:4] * k
                fr[y:y + h, x:x + w] = fr[y:y + h, x:x + w] * (1 - al) + a2[..., :3] * al
                return


# ---------------- clinic stickers on the nipples of a moving video ----------------
def sticker_rgba(d):
    """the clinic's round sticker (navy, white lettering) cut out of its black square"""
    im = Image.open(os.path.join(DRIVE, "sticker_navy.png")).convert("RGB").resize((d, d), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", (d * 4, d * 4), 0); ImageDraw.Draw(m).ellipse((6, 6, d * 4 - 7, d * 4 - 7), fill=255)
    im.putalpha(m.resize((d, d), Image.LANCZOS)); return np.asarray(im, np.float32) / 255.0


def frames_of(src, dur):
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(DRIVE, src), "-t", str(dur), "-vf", f"scale={W}:{H},fps={FPS}",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(b, np.uint8).reshape(-1, H, W)


def track(src, dur, inits):
    """positions of points through the clip: optical flow of the texture around each point (median shift),
    re-initialised from hand-placed `inits` (one list of points per shot) at every hard cut"""
    g = frames_of(src, dur); cuts = [0] + [i for i in range(1, len(g)) if np.abs(g[i].astype(np.int16) - g[i - 1]).mean() > 18]
    pos = np.zeros((len(g), len(inits[0]), 2), np.float32)
    for si, c0 in enumerate(cuts):
        c1 = cuts[si + 1] if si + 1 < len(cuts) else len(g); pts = np.array(inits[min(si, len(inits) - 1)], np.float32)
        pos[c0] = pts
        for i in range(c0 + 1, c1):
            for k, (x, y) in enumerate(pts):
                x0, y0 = int(max(x - 90, 0)), int(max(y - 90, 0)); roi = np.zeros_like(g[i - 1]); roi[y0:int(y + 90), x0:int(x + 90)] = 255
                f = cv2.goodFeaturesToTrack(g[i - 1], 60, 0.01, 6, mask=roi)
                if f is None: continue
                nf, st, _ = cv2.calcOpticalFlowPyrLK(g[i - 1], g[i], f, None, winSize=(31, 31), maxLevel=3)
                ok = st[:, 0] == 1
                if ok.sum() >= 5: pts[k] += np.median((nf - f)[ok, 0], axis=0)
            pos[i] = pts
    print("cuts at frames", cuts, flush=True)
    return pos


# ---------------- stories ----------------
NOSHADE = np.zeros((H, 1, 1), np.float32)


def s1():
    """before video (3/4) full frame; the text low on the legs under a black fade (torso open);
    the poll sticker goes on the wall at the top right"""
    b = (Block().head("«ПОСЛЕ РОДОВ Я ВРОДЕ ВЕРНУЛАСЬ В СВОЙ ВЕС… НО ТЕЛО УЖЕ СТАЛО ДРУГИМ»", 48, gap=16)
         .body("Живот не собирается так, как раньше.", 33, gap=0).body("Грудь потеряла форму.", 33, gap=0)
         .body("Появилось ощущение, что ты стараешься, а тело всё равно не возвращается назад.", 33, gap=0))
    top = H - 70 - b.h; ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top)
    render_video("01_posle_rodov.mp4", "IMG_0336.MOV", 0.0, 8.1, ly, fade(top + 40, 300, 0.85))


def s2():
    """before (belly) video full frame; the text low on the thighs under a black fade, the belly open"""
    b = (Block().body("У нашей пациентки после родов и кормления ослабла передняя брюшная стенка, был выраженный диастаз, "
                      "растяжки и старый рубец после полостной операции.", 31, gap=14)
         .body("И вот здесь очень важно понять:", 31, wght=500, gap=6)
         .head("ЭТО УЖЕ НЕ ИСТОРИЯ ПРО «ЕЩЁ НЕМНОГО ПОХУДЕТЬ»", 44, gap=12)
         .body("Некоторые изменения спортом просто не убрать.", 33, wght=600, gap=0))
    top = H - 60 - b.h; ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top)
    render_video("02_diastaz.mp4", "IMG_0337.MOV", 0.0, 9.9, ly, fade(top + 40, 300, 0.85))


def arrow(d, p0, p1, width=3, head=13):
    """short thin white arrow p0 -> p1 (with a faint dark edge so it reads on skin)"""
    (x0, y0), (x1, y1) = p0, p1; a = np.arctan2(y1 - y0, x1 - x0)
    pts = [(x1 - head * np.cos(a - 0.45), y1 - head * np.sin(a - 0.45)), (x1, y1), (x1 - head * np.cos(a + 0.45), y1 - head * np.sin(a + 0.45))]
    for col, w in [((20, 23, 31, 90), width + 2), ((255, 255, 255, 255), width)]:
        d.line([p0, p1], fill=col, width=w); d.line(pts, fill=col, width=w, joint="curve")


DIRS = {"L": (-1, 0), "R": (1, 0), "UL": (-0.8, -0.6), "UR": (0.8, -0.6), "DL": (-0.8, 0.6), "DR": (0.8, 0.6), "U": (0, -1)}


def finding(ly, text, target, side, size=26, length=62):
    """what we see on the before photo: a navy pill label on `side` of `target` and a short thin white arrow
    (`length` px) from just outside the label to the spot"""
    lab = pill(text, size); w, h = lab.size; dx, dy = DIRS[side]; n = np.hypot(dx, dy); dx, dy = dx / n, dy / n
    k = min((w / 2 + 8) / max(abs(dx), 1e-3), (h / 2 + 8) / max(abs(dy), 1e-3))   # centre -> edge along the direction
    sx, sy = target[0] + dx * length, target[1] + dy * length                      # arrow start
    cx, cy = sx + dx * k, sy + dy * k
    cx2 = min(max(cx, w / 2 + 12), W - w / 2 - 12)
    if cx2 != cx:                                   # pushed in from the frame edge: start the arrow at the label's edge
        cx = cx2; vx, vy = target[0] - cx, target[1] - cy; n = np.hypot(vx, vy) + 1e-6; vx, vy = vx / n, vy / n
        k2 = min((w / 2 + 8) / max(abs(vx), 1e-3), (h / 2 + 8) / max(abs(vy), 1e-3)); sx, sy = cx + vx * k2, cy + vy * k2
    ly.alpha_composite(lab, (int(cx - w / 2), int(cy - h / 2)))
    arrow(ImageDraw.Draw(ly), (sx, sy), target)


def photo_top(img, area_h, crop=None, fade_px=140):
    """the photo (on black) filling the top `area_h` px, its lower edge melting into black; returns canvas, map(x, y)"""
    if crop: img = img.crop(crop)
    sc = area_h / img.height; im = img.resize((int(img.width * sc), area_h), Image.LANCZOS)
    cv = Image.new("RGB", (W, H), (0, 0, 0)); xo = (W - im.width) // 2; cv.paste(im, (xo, 0))
    a = np.asarray(cv, np.float32); ys = np.arange(H, dtype=np.float32)
    g = np.clip((area_h - ys) / fade_px, 0, 1)[:, None, None]; a = a * g
    ox, oy = (crop[0], crop[1]) if crop else (0, 0)
    return Image.fromarray(a.astype(np.uint8)).convert("RGBA"), (lambda x, y: (xo + (x - ox) * sc, (y - oy) * sc))


def s3():
    """before 3/4 (chest to hips, full width — the far breast is not cut) on top, what we see marked with short
    arrows; text and the poll room below on black, no fade"""
    b = (Block().head("ГРУДЬ ТОЖЕ ИЗМЕНИЛАСЬ", 52, gap=14)
         .body("Появилось небольшое опущение и асимметрия.", 31, gap=8)
         .body("При этом пациентка не хотела дополнительных разрезов на груди.", 31, gap=8)
         .body("Поэтому подтяжку мы не делали.", 32, wght=600, gap=0))
    poll_h = 280; top = H - 50 - poll_h - b.h
    cv, m = photo_top(photo("IMG_8390.JPG"), top - 30, crop=(0, 300, 1080, 300 + top - 30), fade_px=1)   # 1:1, full width
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top)
    lab = pill("наше до", 28); ly.alpha_composite(lab, (MARGIN - 10, 60))
    finding(ly, "опущение груди", m(540, 820), "DL")                    # lower pole of the near breast (sticker 616, 700)
    finding(ly, "асимметрия", m(940, 610), "UL")                        # the far breast sits lower
    cv.alpha_composite(ly); save_jpg(cv, "03_grud.jpg")


def s4():
    """operating room photo; what was done — the two operations on navy plates"""
    cv = photo("5223F1B9-E4B1-4ABB-8DB1-EDB2B7955A04.png").resize((W, H), Image.LANCZOS)
    b = (Block().head("МЫ РЕШИЛИ РАБОТАТЬ СРАЗУ [[С ДВУМЯ ЗОНАМИ]]", 50, gap=18)
         .body("Сделали:", 32, wght=500, gap=10)
         .plate("абдоминопластику с ушиванием диастаза", 32, gap=12)
         .plate("увеличение груди анатомическими имплантами 375 мл", 32, gap=22)
         .body("На животе убрали [[избыток кожи,]] поработали со старой рубцовой тканью, сформировали [[новый пупок]] "
               "и более [[выраженную талию.]]", 31, gap=0))
    top = 1790 - b.h
    cv = Image.fromarray((np.asarray(cv, np.float32) * (1 - fade(top + 60, 420, 0.85))).astype(np.uint8)).convert("RGBA")
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top); cv.alpha_composite(ly)
    save_jpg(cv, "04_dve_zony.jpg")


def s5():
    """first day: the surgeon's voice; text on her dark scrubs; the clinic stickers follow the nipples"""
    words = json.load(open(os.path.join(DRIVE, "day1.mp4.subs.json")))[0]["w"]
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                os.path.join(DRIVE, "day1.mp4")], capture_output=True, text=True).stdout)
    # hand-placed nipples (near, far) at the start of each shot, then optical flow
    pos = track("day1.mp4", dur, [[(757, 1452), (730, 1290)], [(752, 1402), (733, 1276)]])
    st = [sticker_rgba(118), sticker_rgba(96)]

    def stick(fr, i):
        f = fr.astype(np.float32) / 255
        for k, (x, y) in enumerate(pos[min(i, len(pos) - 1)]):
            a = st[k]; h, w = a.shape[:2]; x0, y0 = int(x - w / 2), int(y - h / 2)
            if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H: continue
            al = a[..., 3:4]; f[y0:y0 + h, x0:x0 + w] = f[y0:y0 + h, x0:x0 + w] * (1 - al) + a[..., :3] * al
        return (f * 255).astype(np.uint8)

    b = (Block().head("[[ПЕРВЫЕ СУТКИ]]", 62, gap=14).body("Это ещё [[ранний результат.]]", 34, wght=600, gap=10)
         .body("Дальше ткани будут восстанавливаться, [[отёк уменьшаться,]] а форма постепенно меняться.", 32, gap=0))
    top = 690; ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top)
    render_video("05_pervye_sutki.mp4", "day1.mp4", 0.0, dur, ly, NOSHADE, on_frame=stick,
                 subs=PillSubs(words, y=top + b.h + 70, outline=True))   # white edge: navy pill on navy scrubs


# the clinic's before/after collages: as they are, no text, no arrows
def s6(): save_jpg(photo("IMG_8393.JPG").resize((W, H), Image.LANCZOS), "06_kollazh_pryamo.jpg")
def s7(): save_jpg(photo("IMG_8395.JPG").resize((W, H), Image.LANCZOS), "07_kollazh_3-4.jpg")
def s8(): save_jpg(photo("IMG_8396.JPG").resize((W, H), Image.LANCZOS), "08_kollazh_sboku.jpg")


def s9():
    """the message on its own slide: the surgeon operating (C0585, slow 4K), text low under the usual black fade"""
    b = (Block().head("И ВОТ ЧТО ХОЧЕТСЯ СКАЗАТЬ ЖЕНЩИНАМ, КОТОРЫЕ СМОТРЯТ НА СЕБЯ ПОСЛЕ РОДОВ И ДУМАЮТ:", 44, gap=18)
         .body("«Наверное, я просто недостаточно стараюсь»", 32, gap=2).body("«Надо ещё похудеть»", 32, gap=2)
         .body("«Может, спорт всё-таки поможет»", 32, gap=22).body("Иногда дело уже не в усилиях.", 35, wght=600, gap=8)
         .body("Просто есть изменения тканей, которые требуют совсем другого подхода.", 32, gap=0))
    top = H - 80 - b.h; ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top)
    render_video("09_ne_v_usiliyah.mp4", "C0585.MP4", 0.0, 8.1, ly, fade(top + 40, 420, 0.85), mute=True)


def s10():
    """the invitation on its own slide: photo of the surgeon at work (DSC00932), text low under the usual black fade"""
    cv = photo("DSC00932.JPG").resize((W, H), Image.LANCZOS)
    b = (Block().head("ЕСЛИ ВЫ СМОТРИТЕ НА НАШЕ ДО И ДУМАЕТЕ:", 44, gap=14)
         .body("«У меня очень похоже»", 31, gap=0).body("«Живот так и не восстановился после родов»", 31, gap=0)
         .body("«Грудь стала другой»", 31, gap=0).body("«Хочу снова чувствовать себя собой»", 31, gap=18)
         .head("ПРИХОДИТЕ НА КОНСУЛЬТАЦИЮ", 44, gap=10)
         .body("Посмотрим ваше ДО, состояние тканей и обсудим, какой объём коррекции имеет смысл именно в вашем случае.", 29, gap=14)
         .body("Для записи напишите в Direct «КОНСУЛЬТАЦИЯ» или свяжитесь с нами по телефону", 29, gap=8)
         .plate("+7 926 636 30 00", 32, wght=600, gap=8)
         .body("в WhatsApp, Telegram или MAX.", 29, gap=0))
    top = H - 70 - b.h
    cv = Image.fromarray((np.asarray(cv, np.float32) * (1 - fade(top + 60, 420, 0.88))).astype(np.uint8)).convert("RGBA")
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top); cv.alpha_composite(ly)
    save_jpg(cv, "10_konsultaciya.jpg")


STORIES = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6, 7: s7, 8: s8, 9: s9, 10: s10}
for k, fn in STORIES.items():
    if not ONLY or k in ONLY: fn()
