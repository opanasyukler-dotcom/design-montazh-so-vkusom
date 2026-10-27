"""8 Instagram stories (1080x1920), a light series with one accent: if the own volume is enough, the breast does not
have to be enlarged — breast lift without implants.

Old story design: Inter Bold caps headlines + Montserrat, brand navy + white, black fade under the text, static text,
no emojis; the text does not lie on the torso.  Sources (drive/ from the Drive folder 1HzA58eSGDYtd9aKp8ValVXn8pGnxPMPQ):
IMG_8714 before front, IMG_8725 the clinic's before/after collage (front), IMG_5489 marking with the surgeon's voice
(face already blurred in the source), IMG_5485 after; voice pieces v_obem.mp4 / v_forma.mp4 cut by clean_speech.py,
s4src.mp4 = v_forma + IMG_5485 (cropped under the chin).

usage: python3 stories_lift_no_volume.py SRC_DIR OUT_DIR [story numbers...]
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


NOSHADE = np.zeros((H, 1, 1), np.float32)


def dur(name):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 os.path.join(DRIVE, name)], capture_output=True, text=True).stdout)


def words(name): return json.load(open(os.path.join(DRIVE, name + ".subs.json")))[0]["w"]


def bazaart_crop(name, box):
    """the before/after photos sit on black with a «Bazaart» mark at the bottom right: crop (fractions) inside it"""
    im = photo(name); x0, y0, x1, y1 = box
    return im.crop((int(im.width * x0), int(im.height * y0), int(im.width * x1), int(im.height * y1)))


def s1():
    """before (front) on black, the question on top, the poll room under the body"""
    im = bazaart_crop("IMG_8714.JPG", (0.04, 0.03, 0.96, 0.03 + 0.92 * 2048 / 3640 * 16 / 9)).resize((W, H), Image.LANCZOS)
    cv = im.convert("RGBA")
    b = Block().head("А ЕСЛИ ОБЪЁМ ГРУДИ\nМЕНЯ УСТРАИВАЕТ,\nНО ФОРМА УЖЕ НЕТ?", 52, gap=0)
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, 110); cv.alpha_composite(ly)
    save_jpg(cv, "01_forma.jpg")                                   # poll «Что бы вы хотели изменить?» under the body


def s2():
    """the surgeon's voice over the marking video; text low on the legs under the usual fade, subtitles above it"""
    d = dur("v_obem.mp4")
    b = Block().head("ЗДЕСЬ ИМПЛАНТЫ\nНЕ НУЖНЫ", 58, gap=12).body("Собственного объёма достаточно.", 36, wght=500, gap=0)
    top = H - 90 - b.h; ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top)
    render_video("02_implanty_ne_nuzhny.mp4", "v_obem.mp4", 0.0, d, ly, fade(top + 40, 320, 0.85),
                 subs=PillSubs(words("v_obem.mp4"), y=top - 70))


def s3():
    """the before closer (chest to thighs); short arrows to what we see; the text low on the legs"""
    box = (0.15, 0.17, 0.85, 0.17 + 0.70 * 2048 / 3640 * 16 / 9)
    im = bazaart_crop("IMG_8714.JPG", box); sc = W / im.width; cv = im.resize((W, H), Image.LANCZOS).convert("RGBA")
    m = lambda x, y: ((x - 0.15 * 2048) * sc, (y - 0.17 * 3640) * sc)  # original pixels -> story
    b = (Block().body("У пациентки есть:", 33, wght=500, gap=6)
         .body("широкая ареола", 33, gap=0).body("опущение груди", 33, gap=0)
         .body("при этом хороший собственный объём", 33, wght=600, gap=16)
         .body("Поэтому задача здесь:", 33, wght=500, gap=6)
         .head("ПОДНЯТЬ ГРУДЬ И СДЕЛАТЬ\nФОРМУ БОЛЕЕ СОБРАННОЙ", 42, gap=8)
         .body("без увеличения.", 33, wght=600, gap=0))
    top = H - 70 - b.h
    cv = Image.fromarray((np.asarray(cv.convert("RGB"), np.float32) * (1 - fade(top + 50, 300, 0.88))).astype(np.uint8)).convert("RGBA")
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top)
    finding(ly, "широкая ареола", m(760, 1390), "UL")                 # edge of the left areola (sticker 853, 1450)
    finding(ly, "опущение груди", m(1250, 1585), "DR")                # lower pole of the right breast (sticker 1280, 1450)
    cv.alpha_composite(ly); save_jpg(cv, "03_u_pacientki.jpg")


def s4():
    """the surgeon's comment, then straight the after video; what changes, low under the usual fade"""
    d = dur("s4src.mp4")
    b = (Block().head("ВОТ ЧТО МЕНЯЕТСЯ\nБЕЗ ДОБАВЛЕНИЯ ОБЪЁМА", 48, gap=14)
         .body("форма", 34, wght=500, gap=0).body("положение груди", 34, wght=500, gap=0).body("контур", 34, wght=500, gap=0))
    top = H - 80 - b.h; ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top)
    render_video("04_bez_obema.mp4", "s4src.mp4", 0.0, d, ly, fade(top + 40, 320, 0.85),
                 subs=PillSubs(words("v_forma.mp4"), y=top - 70))


def s5():
    """the clinic's before/after collage (front) on top on its own black, the invitation below"""
    b = (Block().body("Если вы тоже думаете:", 31, wght=500, gap=4)
         .body("«Грудь у меня есть, просто хочется вернуть форму»", 31, gap=12)
         .head("НЕ НУЖНО АВТОМАТИЧЕСКИ ВЫБИРАТЬ ИМПЛАНТЫ", 40, gap=12)
         .body("На консультации посмотрим ваше ДО и обсудим, достаточно ли в вашем случае подтяжки.", 29, gap=12)
         .body("Для записи напишите в Direct «КОНСУЛЬТАЦИЯ» или свяжитесь с нами по телефону", 29, gap=8)
         .plate("+7 926 636 30 00", 32, wght=600, gap=8)
         .body("в WhatsApp, Telegram или MAX.", 29, gap=0))
    top = H - 60 - b.h
    col = photo("IMG_8725.PNG").resize((W, H), Image.LANCZOS)
    a = np.asarray(col).max(2) > 22; ys, xs = np.nonzero(a)
    col = col.crop((0, ys.min(), W, ys.max()))                     # the collage without its black top / bottom
    ah = top - 40; s = ah / col.height; col = col.resize((int(W * s), ah), Image.LANCZOS)
    cv = Image.new("RGB", (W, H), (0, 0, 0)); cv.paste(col, ((W - col.width) // 2, 20)); cv = cv.convert("RGBA")
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top)
    for lab, yy in [("после", 40), ("до", 20 + ah // 2 + 20)]:
        p = pill(lab, 26); ly.alpha_composite(p, ((W - col.width) // 2 + 20 if col.width < W - 200 else 40, yy))
    cv.alpha_composite(ly); save_jpg(cv, "08_konsultaciya.jpg")


# the clinic's before/after collages (after on top, before below), as they are — no text
def c1(): save_jpg(photo("IMG_8725.PNG").resize((W, H), Image.LANCZOS), "05_kollazh_pryamo.jpg")
def c2(): save_jpg(photo("IMG_8726.PNG").resize((W, H), Image.LANCZOS), "06_kollazh_3-4.jpg")
def c3(): save_jpg(photo("IMG_8727.PNG").resize((W, H), Image.LANCZOS), "07_kollazh_sboku.jpg")


STORIES = {1: s1, 2: s2, 3: s3, 4: s4, 5: c1, 6: c2, 7: c3, 8: s5}
for k, fn in STORIES.items():
    if not ONLY or k in ONLY: fn()
