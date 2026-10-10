"""Instagram carousel (1080x1350) in Anna Tvauri post style:
Bebas headline, Montserrat body, Montserrat Light italic accent, lilac rule,
powder-pink logo on top, dark graphite gradient under text.

usage: python3 carousel_tvauri.py SRC_DIR OUT_DIR [slide_numbers...]
"""
import os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

W, H = 1080, 1350
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEBAS = os.path.join(ROOT, "skills/montage-svg-typography/Шрифты/Коллекция шрифтов/BebasNeueCyrillic.ttf")
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
MONT_I = os.path.join(ROOT, "fonts/Montserrat-Italic-VariableFont_wght.ttf")
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

# palette taken from the surgeon's posts
WHITE = (255, 255, 255)
LILAC = (252, 211, 255)        # rule under headline
POWDER = (233, 207, 199)       # logo
GRAPHITE = (20, 23, 31)        # studio backdrop / gradients
NAVY = (31, 52, 92)            # brand stickers
INK = (43, 48, 64)             # dark text on white panel
MIST = (122, 130, 150)         # contacts on white panel

SRC, OUT = sys.argv[1], sys.argv[2]
ONLY = {int(a) for a in sys.argv[3:]}
MARGIN = 104


def mont(size, wght=500, italic=False):
    f = ImageFont.truetype(MONT_I if italic else MONT, size)
    f.set_variation_by_axes([wght])
    return f


def bebas(size): return ImageFont.truetype(BEBAS, size)


def wrap(text, f, width):
    out = []
    for para in text.split("\n"):
        line = ""
        for w in para.split(" "):
            t = (line + " " + w).strip()
            if f.getlength(t) <= width or not line: line = t
            else: out.append(line); line = w
        out.append(line)
    return out


# ---------------- logo ----------------
def logo(alpha=0.9, scale=1.0):
    s = 4
    r = int(25 * scale)
    f = mont(int(40 * scale) * s, 300)
    txt = "ANNA TVAURI"
    tw = int(f.getlength(txt) + 0.06 * f.size * (len(txt) - 1))
    gap = int(22 * scale) * s
    w, h = 2 * r * s + gap + tw + 8 * s, 2 * r * s + 8 * s
    im = Image.new("L", (w, h), 0); d = ImageDraw.Draw(im)
    lw = max(2, int(2.4 * scale)) * s
    cx, cy, R = r * s + 4 * s, h // 2, r * s
    d.ellipse((cx - R, cy - R, cx + R, cy + R), outline=255, width=lw)
    yt = cy - int(R * 0.62); xt = int((R * R - (R * 0.62) ** 2) ** 0.5)
    d.line((cx - xt, yt, cx + xt, yt), fill=255, width=lw)
    yb = cy + int(R * 0.80); xb = int((R * R - (R * 0.80) ** 2) ** 0.5)
    d.line((cx, yt, cx - xb, yb), fill=255, width=lw)
    d.line((cx, yt, cx + xb, yb), fill=255, width=lw)
    yc = cy + int(R * 0.28); k = (yc - yt) / (yb - yt)
    d.line((cx - xb * k, yc, cx + xb * k, yc), fill=255, width=lw)
    x = cx + R + gap; asc, desc = f.getmetrics()
    for ch in txt:
        d.text((x, cy), ch, font=f, fill=255, anchor="lm")
        x += f.getlength(ch) + 0.06 * f.size
    im = im.resize((w // s, h // s), Image.LANCZOS)
    out = Image.new("RGBA", im.size, POWDER + (0,))
    out.putalpha(im.point(lambda v: int(v * alpha)))
    return out


def put_logo(canvas, y=142, alpha=0.9):
    lg = logo(alpha)
    canvas.alpha_composite(lg, ((W - lg.width) // 2, y - lg.height // 2))


# ---------------- gradients ----------------
def vgrad(y0, y1, a0, a1, col=GRAPHITE):
    ys = np.arange(H, dtype=np.float32)
    p = np.clip((ys - y0) / max(1, y1 - y0), 0, 1)
    p = p * p * (3 - 2 * p)
    a = (a0 + (a1 - a0) * p) * 255
    arr = np.zeros((H, W, 4), np.uint8); arr[..., :3] = col
    arr[..., 3] = a[:, None].astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


# ---------------- text blocks ----------------
class Block:
    """vertical stack of text items; measured, then drawn at (x, y)"""
    def __init__(self, width=W - 2 * MARGIN, align="left"):
        self.items = []; self.width = width; self.align = align

    def head(self, text, size=76, gap_after=18, align=None, color=WHITE):
        f = bebas(size); lines = wrap(text.upper(), f, self.width)
        self.items.append(("text", f, lines, int(size * 0.98), color, gap_after, align)); return self

    def body(self, text, size=31, gap_after=22, wght=500, align=None, color=WHITE, lh=1.32):
        f = mont(size, wght); lines = wrap(text, f, self.width)
        self.items.append(("text", f, lines, int(size * lh), color, gap_after, align)); return self

    def ital(self, text, size=40, gap_after=20, align=None, color=WHITE):
        f = mont(size, 300, italic=True); lines = wrap(text, f, self.width)
        self.items.append(("text", f, lines, int(size * 1.25), color, gap_after, align)); return self

    def rule(self, length=0.30, gap_after=26):
        self.items.append(("rule", int(W * length), gap_after)); return self

    def arrows(self, gap_after=0):
        self.items.append(("arrows", gap_after)); return self

    def plaque(self, text, size=30, gap_after=20):
        self.items.append(("plaque", mont(size, 500), text, gap_after)); return self

    def bullets(self, items, size=31, gap_after=22, wght=500, color=WHITE):
        f = mont(size, wght)
        self.items.append(("bullets", f, [wrap(t, f, self.width - 34) for t in items], int(size * 1.32), color, gap_after)); return self

    def space(self, h): self.items.append(("space", h)); return self

    def _h(self, it):
        k = it[0]
        if k == "text": return len(it[2]) * it[3] + it[5]
        if k == "rule": return 4 + it[2]
        if k == "arrows": return 30 + it[1]
        if k == "plaque": asc, desc = it[1].getmetrics(); return asc + desc + 22 + it[3]
        if k == "bullets": return sum(len(l) for l in it[2]) * it[3] + (len(it[2]) - 1) * 10 + it[5]
        if k == "space": return it[1]

    def height(self): return sum(self._h(i) for i in self.items)

    def draw(self, canvas, x, y):
        d = ImageDraw.Draw(canvas)
        for it in self.items:
            k = it[0]
            if k == "text":
                _, f, lines, lh, col, ga, al = it; al = al or self.align
                asc, desc = f.getmetrics()
                for i, ln in enumerate(lines):
                    lx = x if al == "left" else (x + self.width - f.getlength(ln) if al == "right" else x + (self.width - f.getlength(ln)) / 2)
                    shadow_text(canvas, (lx, y + i * lh), ln, f, col)
                y += len(lines) * lh + ga
            elif k == "rule":
                lx = x if self.align == "left" else x + (self.width - it[1]) / 2
                d.rectangle((lx + 4, y, lx + 4 + it[1], y + 3), fill=LILAC + (255,)); y += 4 + it[2]
            elif k == "arrows":
                shadow_text(canvas, (x + 2, y - 4), ">>>", mont(30, 700), WHITE); y += 30 + it[1]
            elif k == "plaque":
                f, text = it[1], it[2]; asc, desc = f.getmetrics(); ph = asc + desc + 22
                pw = f.getlength(text) + 36
                d.rounded_rectangle((x, y, x + pw, y + ph), radius=10, fill=WHITE + (245,))
                d.text((x + 18, y + 11), text, font=f, fill=INK + (255,))
                y += ph + it[3]
            elif k == "bullets":
                _, f, groups, lh, col, ga = it
                for g in groups:
                    d.ellipse((x + 4, y + lh * 0.38, x + 15, y + lh * 0.38 + 11), fill=LILAC + (255,))
                    for i, ln in enumerate(g): shadow_text(canvas, (x + 34, y + i * lh), ln, f, col)
                    y += len(g) * lh + 10
                y += ga - 10
            elif k == "space": y += it[1]
        return y


def shadow_text(canvas, pos, text, f, col, blur=6, a=110):
    """soft dark halo for legibility over footage (like the original posts)"""
    if col == WHITE:
        sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pos[0], pos[1] + 2), text, font=f, fill=GRAPHITE + (a,))
        canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(blur)))
    ImageDraw.Draw(canvas).text(pos, text, font=f, fill=col + (255,))


def heart(canvas, x, y, size=34):
    em = ImageFont.truetype(EMOJI, 109)
    im = Image.new("RGBA", (140, 140), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((4, 4), "❤️", font=em, embedded_color=True)
    im = im.crop(im.getbbox()).resize((size, int(size * 0.9)), Image.LANCZOS)
    canvas.alpha_composite(im, (int(x), int(y)))


# ---------------- overlay for a video slide ----------------
def text_overlay(block, bottom=1240, top=None, grad_top=0.30, grad_bottom=0.86, top_shade=0.35):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    h = block.height()
    if top is not None:  # text under the logo, footage stays clear at the bottom
        y = top
        ov.alpha_composite(vgrad(y + h - 120, y + h + 260, grad_bottom, 0))
    else:
        y = bottom - h
        ov.alpha_composite(vgrad(y - 330, y + 140, 0, grad_bottom))
        if top_shade: ov.alpha_composite(vgrad(0, 300, top_shade, 0))
    put_logo(ov)
    block.draw(ov, MARGIN, y)
    return ov


# ---------------- media helpers ----------------
def run(cmd): subprocess.run(cmd, check=True)


def render_video(name, srcs, ov, out, crop_y=0.5, zoom=1.0, grade="", trim=None):
    """srcs: list of (file, start, end|None); fit-cover to 1080x1350, overlay PNG"""
    ovp = os.path.join(OUT, f"_{name}_overlay.png"); ov.save(ovp)
    ins, fl, dur = [], [], 0.0
    for i, (f, a, b) in enumerate(srcs):
        dur += (b or probe_dur(os.path.join(SRC, f))) - a
        ins += ["-ss", str(a)] + (["-to", str(b)] if b else []) + ["-i", os.path.join(SRC, f)]
        sw = int(W * zoom)
        fl.append(f"[{i}:v]scale={sw}:-2:flags=lanczos,crop={W}:{H}:(iw-{W})/2:(ih-{H})*{crop_y},setsar=1,fps=30,format=yuv420p[v{i}]")
    n = len(srcs)
    cat = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[cat]" if n > 1 else "[v0]null[cat]"
    g = f"[cat]{grade}[g]" if grade else "[cat]null[g]"
    fc = ";".join(fl + [cat, g, f"[g][{n}:v]overlay=0:0:format=auto:shortest=1,format=yuv420p[out]"])
    run(["ffmpeg", "-v", "error", "-y"] + ins + ["-loop", "1", "-i", ovp, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
         "-filter_complex", fc, "-map", "[out]", "-map", f"{n + 1}:a", "-t", f"{dur:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out])
    os.remove(ovp)
    # poster frame for preview
    run(["ffmpeg", "-v", "error", "-y", "-ss", "1.2", "-i", out, "-frames:v", "1", out.replace(".mp4", "_preview.jpg")])


def probe_dur(f):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f],
                                capture_output=True, text=True, check=True).stdout)


def load_photo(f):
    return ImageOps.exif_transpose(Image.open(os.path.join(SRC, f))).convert("RGB")


def body_crop(im, aspect, pad=0.03, wm_cut=0.90):
    """crop a before/after photo on black around the body, dropping the app watermark at the bottom"""
    a = np.asarray(im.convert("L"))
    a = a[: int(a.shape[0] * wm_cut)]
    m = a > 40
    e = 12; m[:e] = m[-e:] = False; m[:, :e] = m[:, -e:] = False  # photos carry a thin light frame line
    # robust to thin frame lines / specks: need a real share of bright pixels per row/col
    xs = np.where(m.mean(0) > 0.03)[0]; ys = np.where(m.mean(1) > 0.03)[0]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    ch = bh * (1 + 2 * pad); cw = ch * aspect
    if cw < bw * (1 + 2 * pad): cw = bw * (1 + 2 * pad); ch = cw / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2))
    bg = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (0, 0, 0))
    bg.paste(im.crop(box))
    bg = Image.eval(bg, lambda v: 0 if v < 14 else v)  # crush JPEG noise in the black backdrop
    # blank anything beyond the watermark cut
    cut = int(im.height * wm_cut) - box[1]
    if cut < bg.height: ImageDraw.Draw(bg).rectangle((0, cut, bg.width, bg.height), fill=(0, 0, 0))
    return bg


def fit_cover(im, w, h, cy=0.5, cx=0.5):
    s = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    x = int((im.width - w) * cx); y = int((im.height - h) * cy)
    return im.crop((x, y, x + w, y + h))


def save_jpg(canvas, out):
    canvas.convert("RGB").save(out, quality=95, subsampling=0)


# ================= slides =================
GRADE_WALL = "eq=brightness=-0.06:contrast=1.04:saturation=0.95,vignette=PI/4.2"
GRADE_DARK = "eq=contrast=1.03:saturation=1.02"


def s01():
    b = (Block().head("Подтяжка груди и абдоминопластика за одну операцию", 84, 22)
         .rule().ital("Разбираем случай: что было до и что получили после", 38, 26).arrows())
    render_video("01", [("IMG_3375.MOV", 0, None), ("IMG_3376.MOV", 0, None), ("IMG_3377.MOV", 0, None)],
                 text_overlay(b, bottom=1200), f"{OUT}/01_cover.mp4", crop_y=0.28, grade=GRADE_DARK)


def s02():
    b = (Block().head("С чем пациентка пришла на консультацию:", 70, 26)
         .bullets(["Выраженное опущение груди: ареолы опустились ниже складки под грудью, ткани растянуты",
                   "Избыток кожи на животе: она собирается в складку и нависает над бельём"], 30, 24)
         .body("После беременностей и колебаний веса кожа теряет упругость и уже не сокращается сама.", 30, 0))
    render_video("02", [("IMG_3357.MOV", 0, None)], text_overlay(b), f"{OUT}/02_before.mp4", crop_y=0.12, grade=GRADE_WALL)


def s03():
    b = (Block().head("Почему тут не помогут спорт и диета?", 76, 22).rule()
         .body("Тренировки укрепляют мышцы, правильное питание уменьшает жировую ткань.", 30, 18)
         .body("Но растянутую кожу они не уберут — её избыток так и останется складкой.", 30, 26)
         .ital("Лишнюю кожу можно убрать только хирургически.", 38, 0))
    render_video("03", [("IMG_3374.MOV", 0, None)], text_overlay(b, top=240), f"{OUT}/03_why.mp4", crop_y=1.0, grade=GRADE_WALL)


def s04():
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    b = (Block().body("В данном случае основная задача:", 30, 20)
         .bullets(["поднять грудь и вернуть ей форму", "убрать избыток кожи и жира на животе",
                   "сделать контур талии ровнее"], 30, 26)
         .ital("Поэтому планируем подтяжку груди и абдоминопластику одним этапом.", 38, 0))
    y = 1240 - b.height()
    ov.alpha_composite(vgrad(y - 420, y + 60, 0, 0.88)); ov.alpha_composite(vgrad(0, 300, 0.35, 0))
    put_logo(ov)
    hb = Block(align="right").head("Наше до", 104, 0)
    hb.draw(ov, MARGIN - 40, y - 150)
    b.draw(ov, MARGIN, y)
    render_video("04", [("IMG_3358.MOV", 0, None)], ov, f"{OUT}/04_nashe_do.mp4", crop_y=0.05, grade=GRADE_WALL)


def s05():
    im = load_photo("DSC00831.JPG")
    c = fit_cover(im, W, H, cy=0.18, cx=0.45).convert("RGBA")
    b = (Block().head("Одна операция — две зоны", 84, 20).rule()
         .body("Если есть показания сразу к двум вмешательствам, их можно объединить: один наркоз и один восстановительный период вместо двух.", 30, 26)
         .ital("Но объединять операции можно не всем. Решение принимаем только после обследования.", 36, 0))
    y = 1250 - b.height()
    c.alpha_composite(vgrad(y - 380, y + 120, 0, 0.84)); c.alpha_composite(vgrad(0, 300, 0.25, 0))
    put_logo(c); b.draw(c, MARGIN, y)
    save_jpg(c, f"{OUT}/05_one_operation.jpg")


def s06():
    b = (Block().head("Что делаем во время операции", 76, 24)
         .plaque("Грудь", 28, 14)
         .body("поднимаем ткани железы, убираем лишнюю кожу и формируем новую, более высокую форму", 29, 24)
         .plaque("Живот", 28, 14)
         .body("удаляем избыток кожи и жира ниже пупка, при необходимости укрепляем мышцы передней брюшной стенки", 29, 0))
    render_video("06", [("IMG_8058.mov", 0, None)], text_overlay(b, bottom=1250), f"{OUT}/06_surgery.mp4",
                 crop_y=0.0, zoom=1.0, grade="eq=brightness=-0.03:contrast=1.04")


def before_after(name, before, after, caption, out, cut=0.90):
    c = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    c.alpha_composite(vgrad(1110, H, 0, 1, GRAPHITE))
    put_logo(c)
    cw, ch, gap, top = 452, 860, 24, 240
    x0 = (W - 2 * cw - gap) // 2
    for i, (f, lab) in enumerate([(before, "До"), (after, "После")]):
        ph = body_crop(load_photo(f), cw / ch, wm_cut=cut).resize((cw, ch), Image.LANCZOS)
        x = x0 + i * (cw + gap)
        c.paste(ph, (x, top))
        fb = bebas(64); tw = fb.getlength(lab.upper())
        ImageDraw.Draw(c).text((x + (cw - tw) / 2, top + ch + 26), lab.upper(), font=fb, fill=WHITE)
        ImageDraw.Draw(c).rectangle((x + cw / 2 - 40, top + ch + 102, x + cw / 2 + 40, top + ch + 105), fill=LILAC)
    f = mont(30, 300, italic=True); tw = f.getlength(caption)
    ImageDraw.Draw(c).text(((W - tw) / 2, top + ch + 140), caption, font=f, fill=WHITE)
    save_jpg(c, out)


def s07(): before_after("07", "IMG_7465.JPG", "IMG_7475.JPG", "Вид спереди", f"{OUT}/07_before_after_front.jpg")
def s08(): before_after("08", "IMG_7466.JPG", "IMG_7469.JPG", "Вид сбоку", f"{OUT}/08_before_after_side.jpg")


def s09():
    b = (Block().head("Что получили в итоге", 80, 20).rule(0.30, 22)
         .ital("Грудь приподнята, живот ровный, талия выразительнее. Швы со временем светлеют.", 36, 0))
    render_video("09", [("IMG_3378.MOV", 0, None)], text_overlay(b, bottom=1260), f"{OUT}/09_result.mp4", crop_y=0.50, grade=GRADE_DARK)


def s10():
    c = Image.new("RGBA", (W, H), WHITE + (255,))
    ph = fit_cover(load_photo("DSC00888.JPG"), W, 640, cy=0.20, cx=0.5).convert("RGBA")
    c.alpha_composite(ph, (0, 0))
    put_logo(c, y=80, alpha=0.95)
    # soft white fade from photo into the panel
    arr = np.zeros((H, W, 4), np.uint8); arr[..., :3] = 255
    ys = np.arange(H)[:, None]; xs = np.arange(W)[None, :]
    p = np.clip((ys - 520 + (xs / W) * 60) / 160, 0, 1); arr[..., 3] = (p * p * (3 - 2 * p) * 255).astype(np.uint8)
    c.alpha_composite(Image.fromarray(arr, "RGBA"))
    bw = 820; x = (W - bw) // 2
    b = (Block(bw, "center").head("Как понять, подходит ли это вам?", 64, 16, color=INK).rule(0.16, 30)
         .body("Только на очной консультации: оцениваем положение груди, состояние кожи и мышц живота, ваше здоровье и пожелания.", 32, 34, 500, color=INK, lh=1.36))
    y = b.draw(c, x, 720)
    f = mont(33, 500); t = "Поэтому жду вас"; tw = f.getlength(t)
    ImageDraw.Draw(c).text(((W - tw - 44) / 2, y), t, font=f, fill=INK)
    heart(c, (W - tw - 44) / 2 + tw + 10, y + 4, 32)
    y += 96
    b2 = Block(bw, "center").ital("Для записи напишите в Direct «КОНСУЛЬТАЦИЯ» или свяжитесь с нами по телефону +7 926 636 30 00 в WhatsApp, Telegram или MAX.", 31, 0, color=MIST)
    b2.draw(c, x, y)
    save_jpg(c, f"{OUT}/10_cta.jpg")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i, fn in enumerate([s01, s02, s03, s04, s05, s06, s07, s08, s09, s10], 1):
        if not ONLY or i in ONLY:
            fn(); print("done", i, flush=True)
