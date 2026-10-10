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
TS = 29  # running text size


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


def arrow(d, pts, col=WHITE + (255,), w=4, head=22):
    pts = np.asarray(pts, float)
    d.line([tuple(p) for p in pts], fill=col, width=w, joint="curve")
    e = pts[-1]; v = pts[-1] - pts[-3]; v /= np.linalg.norm(v); n = np.array([-v[1], v[0]])
    d.polygon([tuple(e + v * 4), tuple(e - v * head + n * head * 0.5), tuple(e - v * head - n * head * 0.5)], fill=col)


def curve(a, b, bend=0.25, n=40):
    a, b = np.asarray(a, float), np.asarray(b, float); m = (a + b) / 2; dd = b - a
    c = m + np.array([-dd[1], dd[0]]) * bend; t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b


def panel_video(name, srcs, block, out, src_top=0.4, speed=1.0, grade="", text_top=212):
    """text on graphite at the top, footage full-width in the window below it (nothing drawn over the body)"""
    th = block.height()
    y0 = text_top + th + 36
    wh = H - y0
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov.alpha_composite(vgrad(y0 - 2, y0 + 110, 1, 0))   # soft seam between panel and footage
    ImageDraw.Draw(ov).rectangle((0, 0, W, y0 - 2), fill=GRAPHITE + (255,))
    put_logo(ov); block.draw(ov, MARGIN, text_top)
    ovp = os.path.join(OUT, f"_{name}_overlay.png"); ov.save(ovp)
    ins, fl, dur = [], [], 0.0
    for i, (f, a, b) in enumerate(srcs):
        dur += ((b or probe_dur(os.path.join(SRC, f))) - a) / speed
        ins += ["-ss", str(a)] + (["-to", str(b)] if b else []) + ["-i", os.path.join(SRC, f)]
        fl.append(f"[{i}:v]scale={W}:-2:flags=lanczos,crop={W}:{wh}:0:'min(ih-{wh},ih*{src_top})',"
                  f"setpts=PTS/{speed},setsar=1,fps=30,format=yuv420p[v{i}]")
    n = len(srcs)
    cat = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[cat]" if n > 1 else "[v0]null[cat]"
    g = f"[cat]{grade}[g]" if grade else "[cat]null[g]"
    fc = ";".join(fl + [cat, g, f"color=c=0x{GRAPHITE[0]:02x}{GRAPHITE[1]:02x}{GRAPHITE[2]:02x}:s={W}x{H}:r=30[bg]",
                        f"[bg][g]overlay=0:{y0}:shortest=1[c]", f"[c][{n}:v]overlay=0:0:format=auto:shortest=1,format=yuv420p[out]"])
    run(["ffmpeg", "-v", "error", "-y"] + ins + ["-loop", "1", "-i", ovp, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
         "-filter_complex", fc, "-map", "[out]", "-map", f"{n + 1}:a", "-t", f"{dur:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out])
    os.remove(ovp)


# ---- 1. cover on the "before" footage
def s01():
    b = (Block().head("Я вроде уже скинула вес...\nПочему тело всё равно\nне нравится?", 74, 18).rule(0.30, 20)
         .ital("История комплексной коррекции груди и живота", 30, 16).arrows())
    render_video("01", [("IMG_3357.MOV", 0, None)], caption_overlay(b, 1300), f"{OUT}/01_cover.mp4",
                 crop_y=0.55, grade=GRADE_WALL)


# ---- 2. not about weight: text panel + pinch footage
def s02():
    b = (Block().head("Иногда проблема уже не в весе", 70, 20)
         .body("Можно похудеть.", TS, 4).body("Можно заниматься спортом.", TS, 4).body("Можно следить за питанием.", TS, 18)
         .body("Но если после беременности, похудения или колебаний веса остались избытки кожи и изменилось положение груди, тренировки не всегда способны это исправить.", TS, 18)
         .body("И вот здесь женщина часто думает:", TS, 10)
         .ital("«Что ещё мне сделать, чтобы наконец нравиться себе?»", 33, 0))
    panel_video("02", [("IMG_3374.MOV", 1.9, None)], b, f"{OUT}/02_not_weight.mp4", src_top=0.50, speed=0.6, grade=GRADE_WALL)


# ---- 3. "наше до": before photo, findings with arrows
def s03():
    S = 3
    c = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    c.alpha_composite(vgrad(1120, H, 0, 1)); put_logo(c)
    t = bebas(88); tw = t.getlength("НАШЕ ДО")
    ImageDraw.Draw(c).text(((W - tw) / 2, 192), "НАШЕ ДО", font=t, fill=WHITE)
    bw, bh = 514, 790; bx, by = (W - bw) // 2, 322
    ph, box = body_img("IMG_7465.JPG", bw, bh)
    c.paste(ph, (bx, by))
    k = bw / (box[2] - box[0])
    def P(x, y): return (bx + (x * 4 - box[0]) * k, by + (y * 4 - box[1]) * k)
    hi = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0)); d = ImageDraw.Draw(hi)
    fh, fb = bebas(46), mont(23, 500)
    groups = [("По груди:", ["выраженное опущение тканей", "потеря более собранной формы", "снижение наполненности верхней части груди"],
               "left", 360, P(178, 425)),
              ("По животу:", ["выраженный избыток кожи", "снижение тонуса тканей передней брюшной стенки", "изменения в нижней части живота"],
               "right", 720, P(318, 610))]
    colw = bx - 56
    for head, items, al, y, tgt in groups:
        x0 = 36 if al == "left" else bx + bw + 20
        ImageDraw.Draw(c).text((x0, y), head.upper(), font=fh, fill=WHITE); yy = y + 58
        for it in items:
            lines = wrap(it, fb, colw - 22)
            ImageDraw.Draw(c).ellipse((x0 + 2, yy + 10, x0 + 11, yy + 19), fill=LILAC)
            for ln in lines: shadow_text(c, (x0 + 22, yy), ln, fb, WHITE); yy += 30
            yy += 10
        sx, sy = (x0 + 70, yy + 8) if al == "left" else (x0 + colw - 90, y - 16)
        tx = tgt[0] - 12 if al == "left" else tgt[0] + 12
        pts = curve((sx, sy), (tx, tgt[1]), 0.30 if al == "left" else 0.30)
        arrow(d, pts * S, w=4 * S, head=20 * S)
    c.alpha_composite(hi.resize((W, H), Image.LANCZOS))
    save_jpg(c, f"{OUT}/03_nashe_do.jpg")


# ---- 4. two tasks: surgeon photo
def s04():
    c = fit_cover(load_photo("DSC00831.JPG"), W, H, cy=0.10, cx=0.45).convert("RGBA")
    b = (Block().head("Поэтому здесь мы решали сразу две задачи", 70, 18).rule(0.30, 22)
         .body("Грудь нужно было поднять и сформировать более собранную форму.", TS, 14)
         .body("Живот нужно было скорректировать за счёт удаления избытков кожи и работы с передней брюшной стенкой.", TS, 14)
         .body("То есть задача была не просто «сделать живот» или «поднять грудь».", TS, 18)
         .ital("Мы работали с силуэтом целиком.", 36, 0))
    y = 1280 - b.height()
    c.alpha_composite(vgrad(y - 330, y + 110, 0, 0.88)); c.alpha_composite(vgrad(0, 300, 0.25, 0))
    put_logo(c); b.draw(c, MARGIN, y)
    save_jpg(c, f"{OUT}/04_two_tasks.jpg")


# ---- 5. breast: surgery footage
def s05():
    b = (Block().head("Что сделали с грудью", 76, 14)
         .plaque("Выполнили подтяжку груди", 27, 18)
         .body("Задача:", TS, 8)
         .bullets(["поднять ткани", "изменить положение груди", "сделать форму более собранной",
                   "скорректировать выраженное опущение"], TS, 16)
         .body("Без попытки сделать грудь «другой».", TS, 10)
         .ital("Хотелось сохранить её собственный объём, но изменить форму и положение.", 33, 0))
    render_video("05", [("IMG_8058.mov", 0, None)], text_overlay(b, bottom=1280), f"{OUT}/05_breast.mp4",
                 crop_y=0.0, grade="eq=brightness=-0.03:contrast=1.04")


# ---- 6. abdomen: text panel + "after" belly footage
def s06():
    b = (Block().head("Что сделали с животом", 72, 14)
         .plaque("Выполнили абдоминопластику", 27, 18)
         .body("При такой коррекции мы работаем не просто с объёмом.", TS, 12)
         .body("Основная задача — убрать избыток кожи и сформировать более аккуратный контур живота.", TS, 12)
         .ital("Поэтому абдоминопластику нельзя заменить только спортом или липосакцией, если проблема уже в избытке тканей.", 31, 0))
    panel_video("06", [("IMG_3378.MOV", 0, None)], b, f"{OUT}/06_abdomen.mp4", src_top=0.42, grade=GRADE_DARK)


# ---- 7. whole figure: before -> after wipe with the slide text
def s07():
    FPS = 30; cw, ch = 540, 720; cx, cy = (W - cw) // 2, 372
    views = [("IMG_7465.JPG", "IMG_7475.JPG"), ("IMG_7466.JPG", "IMG_7469.JPG")]
    base = Image.new("RGBA", (W, H), (0, 0, 0, 255)); put_logo(base)
    hb = Block(W - 2 * MARGIN, "center").head("Посмотрите не на отдельные зоны, а на фигуру целиком", 62, 0)
    hb.draw(base, MARGIN, 206)
    tb = (Block(W - 2 * MARGIN, "center")
          .body("После коррекции изменились: положение груди, контур живота, линия талии и общие пропорции силуэта.", 27, 12)
          .ital("И именно это даёт ощущение совсем другого тела.", 31, 0))
    tb.draw(base, MARGIN, cy + ch + 34)
    base = np.asarray(base.convert("RGB")).astype(np.float32)
    def tag(lab):
        f = bebas(44); tw = int(f.getlength(lab)) + 32
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.rounded_rectangle((cx + 16, cy + 16, cx + 16 + tw, cy + 70), radius=10, fill=GRAPHITE + (215,))
        d.text((cx + 32, cy + 20), lab, font=f, fill=WHITE)
        a = np.asarray(im).astype(np.float32) / 255; return a[..., :3] * a[..., 3:], a[..., 3:]
    tb_, ta_ = tag("ДО"), tag("ПОСЛЕ")
    imgs = [tuple(np.asarray(body_crop(load_photo(f), cw / ch).resize((cw, ch), Image.LANCZOS)).astype(np.float32) for f in v) for v in views]
    HOLD_B, WIPE, HOLD_A, X = 1.2, 1.5, 2.0, 0.5
    seg = HOLD_B + WIPE + HOLD_A; total = seg * 2 + X
    xs = np.arange(cw)[None, :, None].astype(np.float32)
    out = f"{OUT}/07_whole_figure.mp4"
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                             "-i", "-", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{total:.3f}",
                             "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    def frame_of(v, t):
        bi, ai = imgs[v]
        p = min(max((t - HOLD_B) / WIPE, 0), 1); p = p * p * (3 - 2 * p)
        lx = cw * (1 - p)
        m = np.clip((xs - lx) / 6 + 0.5, 0, 1)
        cell = bi * (1 - m) + ai * m
        if 0 < p < 1:
            glow = np.exp(-((xs - lx) / 5) ** 2)
            cell = cell * (1 - glow) + np.array(LILAC, np.float32) * glow
        fr = base.copy(); fr[cy:cy + ch, cx:cx + cw] = cell
        q = min(max((p - 0.35) / 0.3, 0), 1)
        for (col, al), w_ in ((tb_, 1 - q), (ta_, q)):
            if w_ > 0: fr = fr * (1 - al[..., 0:1] * w_) + col * 255 * w_
        return fr
    for i in range(int(total * FPS)):
        t = i / FPS
        if t < seg: fr = frame_of(0, t)
        elif t < seg + X: k = (t - seg) / X; fr = frame_of(0, seg) * (1 - k) + frame_of(1, 0) * k
        else: fr = frame_of(1, t - seg - X)
        proc.stdin.write(np.clip(fr, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close(); proc.wait()


# ---- 8. why several zones: text panel + "after" footage on the dark backdrop
def s08():
    b = (Block().head("Почему иногда лучше корректировать не одну зону", 66, 18)
         .body("Потому что тело мы воспринимаем целиком.", TS, 12)
         .body("Можно идеально скорректировать живот, но при выраженном опущении груди всё равно чувствовать, что образ не завершён. И наоборот.", TS, 12)
         .ital("Поэтому на консультации я всегда смотрю не только на одну жалобу, а на всю фигуру и пропорции в целом.", 31, 0))
    panel_video("08", [("IMG_3376.MOV", 0, None), ("IMG_3377.MOV", 0, None)], b, f"{OUT}/08_whole_body.mp4",
                src_top=0.24, grade=GRADE_DARK)


# ---- 9. "if you look at our before": quotes over the before footage (reference layout)
def s09():
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov.alpha_composite(vgrad(0, 420, 0.78, 0))
    hb = Block().head("Если вы смотрите на наше до и думаете:", 66, 0)
    put_logo(ov); hb.draw(ov, MARGIN, 206)
    quotes = [("«У меня очень похоже»", "right", 470), ("«Живот не уходит, даже когда вес нормальный»", "left", 600),
              ("«Грудь сильно изменилась и опустилась»", "right", 730),
              ("«Хочется не одну операцию, а привести фигуру в порядок целиком»", "left", 860)]
    f = mont(28, 400, italic=True); d = ImageDraw.Draw(ov)
    for q, al, y in quotes:
        lines = wrap(q, f, 470); lh = 36
        w_ = max(f.getlength(l) for l in lines) + 40; h_ = lh * len(lines) + 22
        x = MARGIN - 30 if al == "left" else W - MARGIN + 30 - w_
        d.rounded_rectangle((x, y, x + w_, y + h_), radius=16, fill=GRAPHITE + (200,))
        for i, l in enumerate(lines): d.text((x + 20, y + 10 + i * lh), l, font=f, fill=WHITE)
    b = (Block().body("Не нужно самостоятельно решать, какие именно операции вам нужны.", TS, 10)
         .ital("Сначала важно оценить ваше ДО.", 36, 0))
    y = 1290 - b.height()
    ov.alpha_composite(vgrad(y - 150, y + 60, 0, 0.8)); b.draw(ov, MARGIN, y)
    render_video("09", [("IMG_3358.MOV", 0, None)], ov, f"{OUT}/09_if_you.mp4", crop_y=0.17, grade=GRADE_WALL)


# ---- 10. consultation + contacts: surgeon photo, white panel
def s10():
    PH = 790
    c = Image.new("RGBA", (W, H), WHITE + (255,))
    im = load_photo("DSC00888.JPG")
    s = W / im.width; sh = im.height * s; off = 0.17 * sh
    ph = im.resize((W, round(sh)), Image.LANCZOS).crop((0, int(off), W, int(off) + PH)).convert("RGBA")
    c.alpha_composite(ph, (0, 0))
    put_logo(c, y=70, alpha=0.95)
    arr = np.zeros((H, W, 4), np.uint8); arr[..., :3] = 255
    ys = np.arange(H)[:, None]; p = np.clip((ys - (PH - 140)) / 140, 0, 1); p = np.broadcast_to(p, (H, W))
    arr[..., 3] = (p * p * (3 - 2 * p) * 255).astype(np.uint8)
    c.alpha_composite(Image.fromarray(arr, "RGBA"))
    bw = 820; x = (W - bw) // 2
    b = Block(bw, "center").head("На консультации мы разберём", 60, 10, color=INK).rule(0.14, 22)
    y = b.draw(c, x, PH - 40)
    items = ["что именно вас беспокоит", "что можно скорректировать", "какие операции имеет смысл сочетать",
             "какой результат можно получить именно в вашем случае"]
    f = mont(28, 500); d = ImageDraw.Draw(c)
    for it in items:
        tw = f.getlength(it); x0 = (W - tw) / 2
        d.ellipse((x0 - 24, y + 12, x0 - 14, y + 22), fill=(214, 160, 222)); d.text((x0, y), it, font=f, fill=INK); y += 42
    y += 22
    Block(bw, "center").ital("Для записи напишите в Direct «КОНСУЛЬТАЦИЯ» или свяжитесь с нами по телефону +7 926 636 30 00 в WhatsApp, Telegram или MAX.", 27, 0, color=MIST).draw(c, x, y)
    save_jpg(c, f"{OUT}/10_cta.jpg")


SLIDES = [s01, s02, s03, s04, s05, s06, s07, s08, s09, s10]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i, fn in enumerate(SLIDES, 1):
        if not ONLY or i in ONLY:
            fn(); print("done", i, flush=True)
