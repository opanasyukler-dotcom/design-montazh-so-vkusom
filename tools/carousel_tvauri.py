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
        if zoom >= 1:
            sw = int(W * zoom)
            fl.append(f"[{i}:v]scale={sw}:-2:flags=lanczos,crop={W}:{H}:(iw-{W})/2:(ih-{H})*{crop_y},setsar=1,fps=30,format=yuv420p[v{i}]")
        else:  # whole body in frame: shrunk footage over a blurred, dimmed copy of itself
            sw = int(W * zoom) // 2 * 2
            fl.append(f"[{i}:v]split[a{i}][b{i}];"
                      f"[a{i}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=40:2,eq=brightness=-0.10:saturation=0.8[bg{i}];"
                      f"[b{i}]scale={sw}:-2:flags=lanczos,crop={sw}:'min(ih,{H})':0:'(ih-min(ih,{H}))*{crop_y}'[fg{i}];"
                      f"[bg{i}][fg{i}]overlay=({W}-w)/2:0,setsar=1,fps=30,format=yuv420p[v{i}]")
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
GRADE_WALL = "eq=brightness=-0.04:contrast=1.04:saturation=0.95"
GRADE_DARK = "eq=contrast=1.03:saturation=1.02"


def caption_overlay(block, bottom=1300, strength=0.72, top=None):
    """short caption at the very bottom (or right under the logo); shade only behind it so the body stays visible"""
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if top is not None:
        y = top
        ov.alpha_composite(vgrad(y + block.height() - 40, y + block.height() + 140, strength, 0))
    else:
        y = bottom - block.height()
        ov.alpha_composite(vgrad(y - 170, y + 70, 0, strength))
        ov.alpha_composite(vgrad(0, 230, 0.22, 0))
    put_logo(ov)
    block.draw(ov, MARGIN, y)
    return ov


def body_box(im, aspect, pad=0.03, wm_cut=0.90):
    a = np.asarray(im.convert("L"))[: int(im.height * wm_cut)]
    m = a > 40
    e = 12; m[:e] = m[-e:] = False; m[:, :e] = m[:, -e:] = False
    xs = np.where(m.mean(0) > 0.03)[0]; ys = np.where(m.mean(1) > 0.03)[0]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    ch = bh * (1 + 2 * pad); cw = ch * aspect
    if cw < bw * (1 + 2 * pad): cw = bw * (1 + 2 * pad); ch = cw / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2))


def body_img(f, w, h, wm_cut=0.90):
    im = load_photo(f); box = body_box(im, w / h, wm_cut=wm_cut)
    return body_crop(im, w / h, wm_cut=wm_cut).resize((w, h), Image.LANCZOS), box


# ---- 1. cover: after footage, whole torso visible
def s01():
    b = (Block().head("Я вроде уже в форме.\nПочему тело всё равно\nне нравится?", 74, 18).rule(0.30, 20)
         .ital("История комплексной коррекции груди и живота", 30, 16).arrows())
    render_video("01", [("IMG_3375.MOV", 0, None), ("IMG_3376.MOV", 0, None), ("IMG_3377.MOV", 0, None)],
                 caption_overlay(b, 1300), f"{OUT}/01_cover.mp4", crop_y=0.45, grade=GRADE_DARK)


# ---- 2. before footage, minimal caption
def s02():
    b = Block().head("Наше до", 84, 8).ital("С этим пациентка пришла на консультацию", 32, 0)
    render_video("02", [("IMG_3357.MOV", 0, None)], caption_overlay(b, top=210), f"{OUT}/02_before.mp4",
                 crop_y=0.17, grade=GRADE_WALL)


# ---- 3. before photo with arrows
def arrow(d, pts, col=WHITE + (255,), w=4, head=22):
    pts = np.asarray(pts, float)
    d.line([tuple(p) for p in pts], fill=col, width=w, joint="curve")
    e = pts[-1]; v = pts[-1] - pts[-3]; v /= np.linalg.norm(v); n = np.array([-v[1], v[0]])
    d.polygon([tuple(e + v * 4), tuple(e - v * head + n * head * 0.5), tuple(e - v * head - n * head * 0.5)], fill=col)


def curve(a, b, bend=0.25, n=40):
    a, b = np.asarray(a, float), np.asarray(b, float); m = (a + b) / 2; dd = b - a
    c = m + np.array([-dd[1], dd[0]]) * bend; t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b


def s03():
    S = 3  # supersample for smooth arrows
    c = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    put_logo(c)
    t = bebas(84); tw = t.getlength("ЧТО ИМЕЕМ ДО")
    ImageDraw.Draw(c).text(((W - tw) / 2, 196), "ЧТО ИМЕЕМ ДО", font=t, fill=WHITE)
    bw, bh = 500, 768; bx, by = (W - bw) // 2, 330
    ph, box = body_img("IMG_7465.JPG", bw, bh)
    c.paste(ph, (bx, by))
    k = bw / (box[2] - box[0])
    def P(x, y):  # point on the 512x910 preview of IMG_7465 -> canvas
        return (bx + (x * 4 - box[0]) * k, by + (y * 4 - box[1]) * k)
    hi = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0)); d = ImageDraw.Draw(hi)
    f1, f2 = mont(26, 600), mont(23, 300, italic=True)
    labels = [  # (text, note, label x, label y, align, target)
        ("Опущение груди", "ареолы ниже\nскладки под грудью", 40, 440, "left", P(190, 440)),
        ("Растянутая кожа", "и растяжки\nна животе", 830, 520, "right", P(325, 515)),
        ("Кожно-жировой «фартук»", "нависает\nнад бельём", 40, 880, "left", P(215, 640)),
        ("Нет талии", "избыток тканей\nпо бокам", 830, 820, "right", P(380, 585)),
    ]
    for title, note, lx, ly, al, tgt in labels:
        tl = wrap(title, f1, 215); nl = note.split("\n")
        yy = ly
        for ln in tl:
            x = lx if al == "left" else W - 40 - f1.getlength(ln)
            shadow_text(c, (x, yy), ln, f1, WHITE); yy += 33
        for ln in nl:
            x = lx if al == "left" else W - 40 - f2.getlength(ln)
            ImageDraw.Draw(c).text((x, yy), ln, font=f2, fill=POWDER); yy += 29
        # arrow from under the label towards the target
        sx = (lx + 60) if al == "left" else (W - 100)
        sy = yy + 14
        pts = curve((sx, sy), (tgt[0] + (-14 if al == "left" else 14), tgt[1]), 0.28 if al == "left" else -0.28)
        arrow(d, pts * S, w=4 * S, head=20 * S)
    hi = hi.resize((W, H), Image.LANCZOS)
    c.alpha_composite(hi)
    f = mont(30, 300, italic=True); cap = "Эти изменения уже не уйдут сами по себе"
    ImageDraw.Draw(c).text(((W - f.getlength(cap)) / 2, 1232), cap, font=f, fill=WHITE)
    save_jpg(c, f"{OUT}/03_before_arrows.jpg")


# ---- 4. pinch video
def s04():
    b = Block().head("Почему не помогут спорт и диета?", 68, 8).ital("Растянутую кожу убирает только хирургия", 32, 0)
    render_video("04", [("IMG_3374.MOV", 0, None)], caption_overlay(b, top=210), f"{OUT}/04_why.mp4",
                 crop_y=0.17, grade=GRADE_WALL)


# ---- 5. bending video
def s05():
    b = Block().head("В наклоне", 84, 8).ital("хорошо видно, насколько растянуты ткани груди", 32, 0)
    render_video("05", [("IMG_3358.MOV", 0, None)], caption_overlay(b, top=210), f"{OUT}/05_bend.mp4",
                 crop_y=0.17, grade=GRADE_WALL)


# ---- 6. surgeon photo: plan + one operation
def s06():
    im = load_photo("DSC00831.JPG")
    c = fit_cover(im, W, H, cy=0.18, cx=0.45).convert("RGBA")
    b = (Block().head("План операции", 84, 18).rule()
         .bullets(["поднять грудь и вернуть ей форму", "убрать избыток кожи и жира на животе",
                   "сделать контур талии ровнее"], 30, 22)
         .body("Обе задачи решаем за одну операцию: один наркоз и один восстановительный период.", 30, 22)
         .ital("Объединять операции можно не всем — решение только после обследования.", 34, 0))
    y = 1260 - b.height()
    c.alpha_composite(vgrad(y - 380, y + 120, 0, 0.86)); c.alpha_composite(vgrad(0, 300, 0.25, 0))
    put_logo(c); b.draw(c, MARGIN, y)
    save_jpg(c, f"{OUT}/06_plan.jpg")


# ---- 7. surgery footage
def s07():
    b = (Block().head("Что делаем во время операции", 76, 24)
         .plaque("Грудь", 28, 14)
         .body("поднимаем ткани железы, убираем лишнюю кожу и формируем новую, более высокую форму", 29, 24)
         .plaque("Живот", 28, 14)
         .body("удаляем избыток кожи и жира ниже пупка, при необходимости укрепляем мышцы передней брюшной стенки", 29, 0))
    render_video("07", [("IMG_8058.mov", 0, None)], text_overlay(b, bottom=1250), f"{OUT}/07_surgery.mp4",
                 crop_y=0.0, zoom=1.0, grade="eq=brightness=-0.03:contrast=1.04")


# ---- 8. before -> after wipe
def s08():
    FPS = 30; cw, ch, cx, cy = 720, 960, (W - 720) // 2, 236
    views = [("IMG_7465.JPG", "IMG_7475.JPG", "Вид спереди"), ("IMG_7466.JPG", "IMG_7469.JPG", "Вид сбоку")]
    base = Image.new("RGBA", (W, H), (0, 0, 0, 255)); base.alpha_composite(vgrad(1150, H, 0, 1)); put_logo(base)
    base = np.asarray(base.convert("RGB")).astype(np.float32)
    def label_layer(lab, cap):
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        fb = bebas(72); tw = fb.getlength(lab); d.text(((W - tw) / 2, cy + ch + 22), lab, font=fb, fill=WHITE)
        f = mont(28, 300, italic=True); tw = f.getlength(cap); d.text(((W - tw) / 2, cy + ch + 104), cap, font=f, fill=POWDER)
        a = np.asarray(im).astype(np.float32) / 255; return a[..., :3] * a[..., 3:], a[..., 3:]
    imgs = []
    for bf, af, cap in views:
        bi = np.asarray(body_crop(load_photo(bf), cw / ch).resize((cw, ch), Image.LANCZOS)).astype(np.float32)
        ai = np.asarray(body_crop(load_photo(af), cw / ch).resize((cw, ch), Image.LANCZOS)).astype(np.float32)
        imgs.append((bi, ai, label_layer("ДО", cap), label_layer("ПОСЛЕ", cap)))
    HOLD_B, WIPE, HOLD_A, X = 1.2, 1.5, 2.0, 0.5
    seg = HOLD_B + WIPE + HOLD_A
    total = seg * 2 + X
    xs = np.arange(cw)[None, :, None].astype(np.float32)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                             "-i", "-", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{total:.3f}",
                             "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", f"{OUT}/08_transition.mp4"],
                            stdin=subprocess.PIPE)
    def frame_of(v, t):
        bi, ai, lb, la = imgs[v]
        p = min(max((t - HOLD_B) / WIPE, 0), 1); p = p * p * (3 - 2 * p)
        lx = cw * (1 - p)  # line travels right -> left, "after" is revealed behind it
        m = np.clip((xs - lx) / 6 + 0.5, 0, 1)
        cell = bi * (1 - m) + ai * m
        if 0 < p < 1:
            glow = np.exp(-((xs - lx) / 5) ** 2)
            cell = cell * (1 - glow) + np.array(LILAC, np.float32) * glow
        fr = base.copy(); fr[cy:cy + ch, cx:cx + cw] = cell
        q = min(max((p - 0.35) / 0.3, 0), 1)
        for (col, al), w_ in ((lb, 1 - q), (la, q)):
            if w_ > 0: fr = fr * (1 - al[..., 0:1] * w_) + col * 255 * w_
        return fr
    n = int(total * FPS)
    for i in range(n):
        t = i / FPS
        if t < seg: fr = frame_of(0, t)
        elif t < seg + X:  # cross-fade from front "after" to side "before"
            k = (t - seg) / X; fr = frame_of(0, seg) * (1 - k) + frame_of(1, 0) * k
        else: fr = frame_of(1, t - seg - X)
        proc.stdin.write(np.clip(fr, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    run(["ffmpeg", "-v", "error", "-y", "-ss", "2.0", "-i", f"{OUT}/08_transition.mp4", "-frames:v", "1", f"{OUT}/08_transition_preview.jpg"])


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


def s09(): before_after("09", "IMG_7465.JPG", "IMG_7475.JPG", "Вид спереди", f"{OUT}/09_before_after_front.jpg")
def s10(): before_after("10", "IMG_7466.JPG", "IMG_7469.JPG", "Вид сбоку", f"{OUT}/10_before_after_side.jpg")


# ---- 11. after footage, minimal caption
def s11():
    b = Block().head("Что получили в итоге", 80, 8).ital("Грудь приподнята, живот ровный, талия выразительнее", 30, 0)
    render_video("11", [("IMG_3378.MOV", 0, None)], caption_overlay(b, 1310), f"{OUT}/11_result.mp4",
                 crop_y=0.60, grade=GRADE_DARK)


# ---- 12. CTA: surgeon photo kept whole (cap to shoulders), white panel below
def s12():
    PH = 930
    c = Image.new("RGBA", (W, H), WHITE + (255,))
    im = load_photo("DSC00888.JPG")
    s = W / im.width; sh = im.height * s
    off = 0.15 * sh  # start just above the surgical cap
    ph = im.resize((W, round(sh)), Image.LANCZOS).crop((0, int(off), W, int(off) + PH)).convert("RGBA")
    c.alpha_composite(ph, (0, 0))
    put_logo(c, y=70, alpha=0.95)
    arr = np.zeros((H, W, 4), np.uint8); arr[..., :3] = 255
    ys = np.arange(H)[:, None]; p = np.clip((ys - (PH - 150)) / 150, 0, 1); p = np.broadcast_to(p, (H, W))
    arr[..., 3] = (p * p * (3 - 2 * p) * 255).astype(np.uint8)
    c.alpha_composite(Image.fromarray(arr, "RGBA"))
    bw = 860; x = (W - bw) // 2
    b = (Block(bw, "center").head("Как понять, подходит ли это вам?", 58, 10, color=INK).rule(0.14, 22)
         .body("Только на очной консультации — оцениваем именно ваш случай.", 30, 18, 500, color=INK))
    y = b.draw(c, x, PH - 30)
    f = mont(31, 500); t = "Жду вас"; tw = f.getlength(t)
    ImageDraw.Draw(c).text(((W - tw - 42) / 2, y), t, font=f, fill=INK)
    heart(c, (W - tw - 42) / 2 + tw + 10, y + 4, 30)
    y += 62
    Block(bw, "center").ital("Для записи напишите в Direct «КОНСУЛЬТАЦИЯ» или свяжитесь с нами по телефону +7 926 636 30 00 в WhatsApp, Telegram или MAX.", 27, 0, color=MIST).draw(c, x, y)
    save_jpg(c, f"{OUT}/12_cta.jpg")


SLIDES = [s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i, fn in enumerate(SLIDES, 1):
        if not ONLY or i in ONLY:
            fn(); print("done", i, flush=True)
