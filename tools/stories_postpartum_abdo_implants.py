"""7 Instagram stories (1080x1920): after childbirth — abdominoplasty with diastasis repair + anatomical implants 375 ml.

Old story design: Inter Bold caps headlines + Montserrat, brand navy + white, black fade under the text, static text,
no emojis.  Text never lies on the body that the story is about (the before / after photos sit on black, the
text goes around them).  Photo stories are JPG, the two video stories MP4.  Stories 1 and 3 keep free space for
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
class Block:
    """left-aligned stack; items: ("head"|"body"|"plate"|"gap", ...)"""
    def __init__(self, width=TW): self.items = []; self.h = 0; self.width = width

    def head(self, text, size=52, gap=14):
        f = bold(size); lines = wrap(text, f, self.width); step = int(size * 1.14)
        self.items.append(("text", lines, f, step)); self.h += step * len(lines) + gap; self.items.append(("gap", gap)); return self

    def body(self, text, size=34, wght=400, gap=16, lh=1.3):
        f = mont(size, wght); lines = wrap(text, f, self.width); step = int(size * lh)
        self.items.append(("text", lines, f, step)); self.h += step * len(lines) + gap; self.items.append(("gap", gap)); return self

    def plate(self, text, size=34, wght=500, gap=14):
        """navy rounded plate with white text (a key item of a list)"""
        f = mont(size, wght); lines = wrap(text, f, self.width - 56); step = int(size * 1.3)
        h = step * len(lines) + 34; w = int(max(f.getlength(l) for l in lines)) + 56
        self.items.append(("plate", lines, f, step, w, h)); self.h += h + gap; self.items.append(("gap", gap)); return self

    def draw(self, layer, x, y):
        d = ImageDraw.Draw(layer); txt = Image.new("RGBA", layer.size, (0, 0, 0, 0)); dt = ImageDraw.Draw(txt)
        for it in self.items:
            if it[0] == "gap": y += it[1]; continue
            if it[0] == "text":
                _, lines, f, step = it
                for i, ln in enumerate(lines): dt.text((x, y + i * step + step / 2), ln, font=f, fill=WHITE + (255,), anchor="lm")
                y += step * len(lines)
            else:
                _, lines, f, step, w, h = it
                d.rounded_rectangle((x - 4, y, x - 4 + w, y + h), 22, fill=NAVY + (240,))
                for i, ln in enumerate(lines): dt.text((x + 24, y + 17 + i * step + step / 2), ln, font=f, fill=WHITE + (255,), anchor="lm")
                y += h
        sh = Image.new("RGBA", layer.size, GRAPHITE + (0,)); sh.putalpha(txt.getchannel("A").point(lambda v: v * 150 // 255))
        layer.alpha_composite(sh.filter(ImageFilter.GaussianBlur(9))); layer.alpha_composite(txt)
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
def render_video(name, src, a, dur, overlay, shade, shift=0, subs=None, audio=None):
    out = os.path.join(OUT, name); tmp = out + ".v.mp4"
    ov = np.asarray(overlay, np.float32) / 255.0; oa = ov[..., 3:4]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(dur), "-i", os.path.join(DRIVE, src),
                          "-vf", f"scale={W}:{H}:flags=lanczos,fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    n = int(round(dur * FPS)); last = None
    for i in range(n):
        buf = p.stdout.read(W * H * 3)
        if len(buf) == W * H * 3: last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        z = 1.0 + 0.03 * i / n
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z); M[1, 2] += shift
        fr = cv2.warpAffine(last, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32) / 255
        if shift > 0:                                      # blurred stretch of the frame's own top above it
            top = int(M[1, 2]) + 1
            strip = cv2.resize(last[:H // 6], (W // 10, max(2, top // 10 + 1)), interpolation=cv2.INTER_AREA)
            fill = cv2.resize(cv2.GaussianBlur(strip, (0, 0), 3), (W, top + 120)).astype(np.float32) / 255 * 0.6
            g = np.clip((np.arange(top + 120) - top) / 120.0, 0, 1)[:, None, None]
            fr[:top + 120] = fill * (1 - g) + fr[:top + 120] * g
        fr = fr * (1 - shade); fr = fr * (1 - oa) + ov[..., :3] * oa
        if subs: subs.draw(fr, i / FPS)
        o = (np.clip(fr, 0, 1) * 255).astype(np.uint8); enc.stdin.write(o.tobytes())
        if i == min(n - 1, int(2.5 * FPS)): Image.fromarray(o).save(out[:-4] + ".jpg", quality=90)
    p.kill(); enc.stdin.close(); enc.wait()
    au = audio or (src, a)
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


# ---------------- stories ----------------
def s1():
    """before front on black; text on top, room for the poll between the text and the body"""
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    b = (Block().head("«ПОСЛЕ РОДОВ Я ВРОДЕ ВЕРНУЛАСЬ В СВОЙ ВЕС… НО ТЕЛО УЖЕ СТАЛО ДРУГИМ»", 48, gap=20)
         .body("Живот не собирается так, как раньше.", 32, gap=2).body("Грудь потеряла форму.", 32, gap=2)
         .body("Появилось ощущение, что ты стараешься, а тело всё равно не возвращается назад.", 32, gap=0))
    top = 150; poll = (top + b.h + 20, top + b.h + 290)            # free band for «Узнаёте себя?»
    im = photo("IMG_8392.JPG"); im = im.crop((0, 8, im.width, int(im.height * 0.9)))   # shoulders .. the briefs (no light top edge)
    place(cv, im, (0, poll[1], W, H))
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top)
    cv.alpha_composite(ly); save_jpg(cv, "01_posle_rodov.jpg")


def s2():
    """before (belly) video: the body moved down, the text on top over a dark fade"""
    b = (Block().body("У нашей пациентки после родов и кормления ослабла передняя брюшная стенка, был выраженный диастаз, "
                      "растяжки и старый рубец после полостной операции.", 31, gap=22)
         .body("И вот здесь очень важно понять:", 31, wght=500, gap=8)
         .head("ЭТО УЖЕ НЕ ИСТОРИЯ ПРО «ЕЩЁ НЕМНОГО ПОХУДЕТЬ»", 44, gap=18)
         .body("Некоторые изменения спортом просто не убрать.", 34, wght=600, gap=0))
    top = 150; ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top)
    render_video("02_diastaz.mp4", "IMG_0337.MOV", 0.0, 9.9, ly, fade(top + b.h + 40, 330, 0.82, up=True), shift=330)


def s3():
    """before 3/4: the breast stays clear, text below it, poll room at the bottom"""
    im = photo("IMG_8390.JPG").resize((W, H), Image.LANCZOS)                            # whole frame: both stickers in
    cv = im.convert("RGBA")
    b = (Block().head("ГРУДЬ ТОЖЕ ИЗМЕНИЛАСЬ", 54, gap=16)
         .body("Появилось небольшое опущение и асимметрия.", 33, gap=12)
         .body("При этом пациентка не хотела дополнительных разрезов на груди.", 33, gap=12)
         .body("Поэтому подтяжку мы не делали.", 34, wght=600, gap=0))
    poll_h = 300; top = H - 60 - poll_h - b.h
    sh = fade(top + 40, 360, 0.85)
    cv = Image.fromarray((np.asarray(cv.convert("RGB"), np.float32) * (1 - sh)).astype(np.uint8)).convert("RGBA")
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top); cv.alpha_composite(ly)
    save_jpg(cv, "03_grud.jpg")


def s4():
    """operating room photo; what was done — the two operations on navy plates"""
    cv = photo("5223F1B9-E4B1-4ABB-8DB1-EDB2B7955A04.png").resize((W, H), Image.LANCZOS)
    b = (Block().head("МЫ РЕШИЛИ РАБОТАТЬ СРАЗУ С ДВУМЯ ЗОНАМИ", 50, gap=18)
         .body("Сделали:", 32, wght=500, gap=10)
         .plate("абдоминопластику с ушиванием диастаза", 32, gap=8)
         .body("и", 32, gap=8)
         .plate("увеличение груди анатомическими имплантами 375 мл", 32, gap=22)
         .body("На животе убрали избыток кожи, поработали со старой рубцовой тканью, сформировали новый пупок "
               "и более выраженную талию.", 31, gap=0))
    top = 1790 - b.h
    sh = fade(top + 60, 420, 0.85)
    cv = Image.fromarray((np.asarray(cv, np.float32) * (1 - sh)).astype(np.uint8)).convert("RGBA")
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top); cv.alpha_composite(ly)
    save_jpg(cv, "04_dve_zony.jpg")


def s5():
    """first day: the surgeon's voice; text on her dark scrubs between the masked face and the patient"""
    words = json.load(open(os.path.join(DRIVE, "day1.mp4.subs.json")))[0]["w"]
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                os.path.join(DRIVE, "day1.mp4")], capture_output=True, text=True).stdout)
    b = (Block().head("ПЕРВЫЕ СУТКИ", 60, gap=14).body("Это ещё ранний результат.", 34, wght=600, gap=10)
         .body("Дальше ткани будут восстанавливаться, отёк уменьшаться, а форма постепенно меняться.", 32, gap=0))
    top = 690; ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); b.draw(ly, MARGIN, top)
    band = np.zeros((H, 1, 1), np.float32); ys = np.arange(H)
    band[:, 0, 0] = 0.55 * np.clip(1 - np.abs(ys - (top + b.h / 2)) / (b.h / 2 + 140), 0, 1) ** 0.7    # soft dark band
    render_video("05_pervye_sutki.mp4", "day1.mp4", 0.0, dur, ly, band, subs=PillSubs(words, y=top + b.h + 70, outline=True))   # white edge: navy pill on navy scrubs


def s6():
    """before | after side by side on black; the question above, the women's thoughts below"""
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    t = Block().head("И ВОТ ЧТО ХОЧЕТСЯ СКАЗАТЬ ЖЕНЩИНАМ, КОТОРЫЕ СМОТРЯТ НА СЕБЯ ПОСЛЕ РОДОВ И ДУМАЮТ:", 40, gap=0)
    bt = (Block().body("«Наверное, я просто недостаточно стараюсь»", 30, gap=2).body("«Надо ещё похудеть»", 30, gap=2)
          .body("«Может, спорт всё-таки поможет»", 30, gap=18)
          .body("Иногда дело уже не в усилиях.", 34, wght=600, gap=8)
          .body("Просто есть изменения тканей, которые требуют совсем другого подхода.", 31, gap=0))
    top = 140; y0 = top + t.h + 40; yb = H - 70 - bt.h; y1 = yb - 30
    for k, (name, lab) in enumerate([("IMG_8392.JPG", "до"), ("IMG_8391.JPG", "после")]):
        im = photo(name); im = im.crop((0, 0, im.width, int(im.height * 0.86)))            # shoulders .. under the briefs
        place(cv, im, (k * W // 2, y0, (k + 1) * W // 2, y1)); cx = W // 4 + k * W // 2
        p = pill(lab, 30); cv.alpha_composite(p, (cx - p.width // 2, y1 - p.height - 18))
    d = ImageDraw.Draw(cv); d.line((W // 2, y0, W // 2, y1), fill=(0, 0, 0), width=6)
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); t.draw(ly, MARGIN, top); bt.draw(ly, MARGIN, yb); cv.alpha_composite(ly)
    save_jpg(cv, "06_ne_v_usiliyah.jpg")


def s7():
    """the result (after, 3/4) on black between the thoughts and the invitation"""
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    t = (Block().head("ЕСЛИ ВЫ СМОТРИТЕ НА НАШЕ ДО И ДУМАЕТЕ:", 44, gap=14)
         .body("«У меня очень похоже»", 30, gap=0).body("«Живот так и не восстановился после родов»", 30, gap=0)
         .body("«Грудь стала другой»", 30, gap=0).body("«Хочу снова чувствовать себя собой»", 30, gap=0))
    bt = (Block().head("ПРИХОДИТЕ НА КОНСУЛЬТАЦИЮ", 44, gap=12)
          .body("Посмотрим ваше ДО, состояние тканей и обсудим, какой объём коррекции имеет смысл именно в вашем случае.", 29, gap=16)
          .body("Для записи напишите в Direct «КОНСУЛЬТАЦИЯ» или свяжитесь с нами по телефону", 29, gap=6)
          .plate("+7 926 636 30 00", 34, wght=600, gap=6)
          .body("в WhatsApp, Telegram или MAX.", 29, gap=0))
    top = 130; y0 = top + t.h + 20; yb = H - 60 - bt.h; y1 = yb - 20
    im = photo("IMG_8394.JPG"); a = np.asarray(im).max(2) > 25                          # the photo sits on black: crop to it
    ys, xs = np.nonzero(a); im = im.crop((xs.min(), ys.min(), xs.max(), int(ys.min() + (ys.max() - ys.min()) * 0.97)))
    place(cv, im, (0, y0, W, y1))
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); t.draw(ly, MARGIN, top); bt.draw(ly, MARGIN, yb); cv.alpha_composite(ly)
    save_jpg(cv, "07_konsultaciya.jpg")


STORIES = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6, 7: s7}
for k, fn in STORIES.items():
    if not ONLY or k in ONLY: fn()
