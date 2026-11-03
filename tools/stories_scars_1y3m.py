"""2 stories (1080x1920): breast augmentation with implants + lift, 1 year 3 months.

1. before -> «light sweep» transition -> after; before again with minimal arrows (what we see on the before), a slider
   to the after with its arrows, then the 1-year-3-months collages (the client's layout).  Silent video.
2. the scars after 1 year 3 months (IMG_2065) with the text, and a big circle with the scar one month after (for
   comparison) and its caption.

Bodies: IMG_9285 before / IMG_2052 after 1 y 3 m, brought to one framing (neck 150, fingertips 1650, the body axis in
the middle); the dark wall is pulled to black.  Faces are out of frame (cut from the neck).

usage: python3 stories_scars_1y3m.py SRC_DIR OUT_DIR [1|2]     (SRC_DIR/drive: <name>.png photos, the collages dir)
"""
import os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC, OUT = sys.argv[1], sys.argv[2]; ONLY = {int(a) for a in sys.argv[3:]}
DRIVE = os.path.join(SRC, "drive"); os.makedirs(OUT, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf"); BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
COLL = os.path.join(ROOT, "collages_ba_3m_1y3m")
W, H, FPS = 1080, 1920, 30
WHITE, NAVY, GRAPHITE = (255, 255, 255), (2, 31, 71), (20, 23, 31)
MARGIN = 72


def mont(size, wght=400):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([wght]); return f


def to_black(f):
    k = int(f.shape[1] * 0.06)
    bg = np.minimum(np.minimum(np.median(f[:, :k], 1), np.median(f[:, -k:], 1)), 0.35)[:, None, :]
    return np.clip((f - bg) / (1 - bg), 0, 1)


def frame(name, neck, tips, top=150, bot=1650):
    """the photo in the story framing: neck at `top`, fingertips at `bot`, body axis in the middle; float RGB"""
    f = to_black(np.asarray(Image.open(os.path.join(DRIVE, name + ".png")).convert("RGB"), np.float32) / 255)
    ax = np.median(np.nonzero(f[neck:tips].mean(2) > 0.25)[1]); s = (bot - top) / (tips - neck)
    im = Image.fromarray((f * 255).astype(np.uint8)); im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    cv = Image.new("RGB", (W, H)); cv.paste(im, (int(W / 2 - ax * s), int(top - neck * s)))
    a = np.asarray(cv, np.float32) / 255; a[:max(0, int(top - 2))] *= np.linspace(0, 1, max(1, int(top - 2)))[:, None, None]  # nothing above the neck
    return a


def text_layer(lines, x, y, size=34, wght=400, lh=1.3, anchor="la"):
    """white text with a soft dark shadow on a transparent layer (float RGBA)"""
    f = mont(size, wght); im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        d.text((x, y + i * size * lh), t, font=f, fill=WHITE + (255,), anchor=anchor)
    sh = Image.new("RGBA", (W, H), GRAPHITE + (0,)); sh.putalpha(im.getchannel("A").point(lambda v: v * 170 // 255))
    out = Image.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6)), im)
    return np.asarray(out, np.float32) / 255


def arrow_layer(p0, p1, prog=1.0, width=3, head=14):
    """a thin white arrow p0 -> p1, drawn up to `prog`"""
    im = np.zeros((H, W, 4), np.float32)
    if prog <= 0: return im
    (x0, y0), (x1, y1) = p0, p1; xe, ye = x0 + (x1 - x0) * prog, y0 + (y1 - y0) * prog
    for col, w in [((0.08, 0.09, 0.12, 0.55), width + 3), ((1, 1, 1, 1), width)]:
        cv2.line(im, (int(x0), int(y0)), (int(xe), int(ye)), col, w, cv2.LINE_AA)
        if prog > 0.85:
            a = np.arctan2(y1 - y0, x1 - x0)
            for s in (-0.5, 0.5):
                cv2.line(im, (int(xe), int(ye)), (int(xe - head * np.cos(a + s)), int(ye - head * np.sin(a + s))), col, w, cv2.LINE_AA)
    return im


def over(fr, lay, k=1.0):
    al = lay[..., 3:4] * k; return fr * (1 - al) + lay[..., :3] * al


def ease(x): x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)


# what we see: (label lines, label anchor point, arrow start, arrow end, appear time in its phase)
BEFORE_NOTES = [(["снижение наполненности", "верхнего полюса"], (620, 300), (700, 385), (690, 455), 0.3),
                (["опущение груди"], (MARGIN, 880), (230, 865), (290, 760), 1.2)]
AFTER_NOTES = [(["наполненный", "верхний полюс"], (MARGIN, 300), (230, 385), (410, 460), 0.6),
               (["грудь стала выше"], (650, 330), (790, 360), (840, 545), 1.3),
               (["собранная форма"], (MARGIN, 880), (240, 865), (420, 745), 2.0)]


def s1():
    out = os.path.join(OUT, "01_do_posle.mp4"); tmp = out + ".v.mp4"
    before = frame("IMG_9285.JPG", 300, 3100); after = frame("IMG_2052.JPG", 160, 1730)
    hdr_do = text_layer(["ДО:"], MARGIN, 70, 44, 700)
    hdr_po = text_layer(["ПОСЛЕ:", "увеличение груди имплантами + подтяжка"], MARGIN, 70, 34, 500, lh=1.35)
    notes_b = [(text_layer(l, x, y, 32, 400), p0, p1, t) for l, (x, y), p0, p1, t in BEFORE_NOTES]
    notes_a = [(text_layer(l, x, y, 32, 400), p0, p1, t) for l, (x, y), p0, p1, t in AFTER_NOTES]
    colls = [np.asarray(Image.open(os.path.join(COLL, f"kollazh_1_god_3_mesyaca_{v}_kak_u_vas.jpg")).convert("RGB"), np.float32) / 255
             for v in ["pryamo", "3-4", "sboku"]]
    seam = Image.new("RGBA", (W, H), (0, 0, 0, 0)); f = mont(30, 500); t = "спустя 1 год 3 месяца после операции"
    w = int(f.getlength(t)) + 52; d = ImageDraw.Draw(seam)
    d.rounded_rectangle(((W - w) // 2, 960 - 30, (W + w) // 2, 960 + 30), 30, fill=NAVY + (235,))
    d.text((W / 2, 961), t, font=f, fill=WHITE + (255,), anchor="mm"); seam = np.asarray(seam, np.float32) / 255
    # timeline (s)
    T1, T2, T3 = 1.4, 1.1, 1.4                 # before, light-sweep, after
    T4, T5 = 0.4, 2.8                          # back to before, its notes
    T6, T7 = 1.3, 3.4                          # slider, the after notes
    TC, X = 2.0, 0.35                          # each collage, crossfade
    b4 = T1 + T2 + T3; b5 = b4 + T4; b6 = b5 + T5; b7 = b6 + T6; b8 = b7 + T7
    total = b8 + 3 * TC
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); diag = (xx * 0.55 + yy) / (W * 0.55 + H)   # 0..1 along the diagonal
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for i in range(int(round(total * FPS))):
        t = i / FPS
        if t < T1:
            z = 1 + 0.03 * t / T1; fr = cv2.warpAffine(before, cv2.getRotationMatrix2D((W / 2, H * 0.4), 0, z), (W, H))
        elif t < T1 + T2:                      # light sweep: a soft glowing edge runs down the diagonal and turns before into after
            p = ease((t - T1) / T2); e = p * 1.3 - 0.15
            m = np.clip((e - diag) / 0.12 + 0.5, 0, 1)[..., None]
            glow = np.exp(-((diag - e) / 0.05) ** 2)[..., None] * 0.55
            fr = np.clip(before * (1 - m) + after * m + glow * np.array([1.0, 0.97, 0.92]), 0, 1)
        elif t < b4:
            z = 1 + 0.03 * (t - T1 - T2) / T3; fr = cv2.warpAffine(after, cv2.getRotationMatrix2D((W / 2, H * 0.4), 0, z), (W, H))
        elif t < b6:                           # back to the before; what we see appears step by step
            fr = after * (1 - ease((t - b4) / T4)) + before * ease((t - b4) / T4)
            lt = t - b5
            fr = over(fr, hdr_do, ease(lt / 0.3))
            for lay, p0, p1, ta in notes_b:
                fr = over(fr, arrow_layer(p0, p1, ease((lt - ta) / 0.35))); fr = over(fr, lay, ease((lt - ta - 0.2) / 0.3))
        elif t < b7:                           # slider left -> right, the after takes over
            p = ease((t - b6) / T6); xs = int(W * p); fr = before.copy(); fr[:, :xs] = after[:, :xs]
            fr[:, max(xs - 2, 0):xs + 2] = 1.0
            ring = np.zeros((H, W, 4), np.float32); cy = 960
            cv2.circle(ring, (xs, cy), 40, (0, 0, 0, 0.35), -1, cv2.LINE_AA); cv2.circle(ring, (xs, cy), 40, (1, 1, 1, 1), 3, cv2.LINE_AA)
            for s in (-1, 1):
                pts = np.array([[xs + s * 8, cy - 10], [xs + s * 20, cy], [xs + s * 8, cy + 10]], np.int32); cv2.fillPoly(ring, [pts], (1, 1, 1, 1), cv2.LINE_AA)
            fr = over(fr, ring)
            k = 1 - ease((t - b6) / 0.3)
            if k > 0:
                fr = over(fr, hdr_do, k)
                for lay, p0, p1, _ in notes_b: fr = over(fr, arrow_layer(p0, p1), k); fr = over(fr, lay, k)
        elif t < b8:                           # the after with its notes
            fr = after.copy(); lt = t - b7
            fr = over(fr, hdr_po, ease(lt / 0.35))
            for lay, p0, p1, ta in notes_a:
                fr = over(fr, arrow_layer(p0, p1, ease((lt - ta) / 0.35))); fr = over(fr, lay, ease((lt - ta - 0.2) / 0.3))
        else:                                  # the collages
            tc = t - b8; k = min(int(tc // TC), 2); lt = tc - k * TC
            fr = colls[k].copy()
            if lt < X: q = ease(lt / X); fr = (colls[k - 1] if k else after) * (1 - q) + fr * q
            fr = over(fr, seam, ease(lt / 0.3) if k == 0 else 1.0)
        o = (np.clip(fr, 0, 1) * 255).astype(np.uint8); enc.stdin.write(o.tobytes())
        for nm, tt in [("a_do", b6 - 0.1), ("b_posle", b8 - 0.1)]:
            if i == int(tt * FPS): Image.fromarray(o).save(os.path.join(OUT, f"01_preview_{nm}.jpg"), quality=90)
    enc.stdin.close(); enc.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c", "copy", "-movflags", "+faststart", out], check=True)
    os.remove(tmp); print("wrote", out, f"{total:.1f}s", flush=True)


def warm(img, da=4.0, db=10.0, dl=3.0):
    """a little warmth on the skin (a/b towards peach on warm pixels only)"""
    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    m = cv2.GaussianBlur(np.clip((lab[..., 2] - 128) / 8, 0, 1) * np.clip((lab[..., 0] - 50) / 40, 0, 1), (0, 0), 2)
    lab[..., 0] += dl * m; lab[..., 1] += da * m; lab[..., 2] += db * m
    return Image.fromarray(cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))


def s2():
    """the scars now (IMG_2065 as shot: the whole photo, 1536x2730 = 9:16, only scaled to 1080x1920); the text low on
    the left, the circle with the scar after one month (whole chest as shot, warmed up) low on the right with its
    caption; «Подробнее…» straight in italics at the bottom; no dark haze"""
    bg = np.asarray(Image.open(os.path.join(DRIVE, "IMG_2065.JPG.png")).convert("RGB").resize((W, H), Image.LANCZOS), np.float32) / 255
    fr = bg                                                  # no dark haze: the text holds by its own shadow
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    hb = ImageFont.truetype(BOLD, 40); f = mont(29, 400); fb = mont(30, 600); y = 990; x = 56
    for t in ["ПРОШЁЛ 1 ГОД И 3 МЕСЯЦА", "ПОСЛЕ ОПЕРАЦИИ"]: d.text((x, y), t, font=hb, fill=WHITE); y += 48
    y += 16; d.text((x, y), "Вот так сейчас", font=fb, fill=WHITE); y += 38; d.text((x, y), "выглядят рубцы.", font=fb, fill=WHITE); y += 52
    for ln in ["В первые месяцы они могут", "быть яркими и заметными,", "и этого многие пациентки", "боятся. Но со временем",
               "рубцы светлеют, становятся", "мягче и практически", "сливаются с кожей."]:
        d.text((x, y), ln, font=f, fill=WHITE); y += 38
    y += 12
    for ln in ["Поэтому не стоит оценивать", "их слишком рано."]: d.text((x, y), ln, font=fb, fill=WHITE); y += 40
    # the circle on the right: the chest one month after, as shot (both breasts, the scars visible), warmed up
    D = 430; cx, cy = W - 40 - (D + 14), 1000
    ph = warm(Image.open(os.path.join(DRIVE, "scar_1m.bin.png")).convert("RGB")).crop((1500, 950, 2500, 1950)).resize((D, D), Image.LANCZOS)   # close-up of the right breast: areola and vertical scar
    circ = Image.new("RGBA", (D + 14, D + 14), (0, 0, 0, 0)); ImageDraw.Draw(circ).ellipse((0, 0, D + 13, D + 13), fill=WHITE + (255,))
    m = Image.new("L", (D, D), 0); ImageDraw.Draw(m).ellipse((0, 0, D - 1, D - 1), fill=255); ph = ph.convert("RGBA"); ph.putalpha(m)
    circ.alpha_composite(ph, (7, 7)); lay.alpha_composite(circ, (cx, cy))
    fc = mont(26, 400); yc = cy + D + 26
    for ln in ["здесь вы можете видеть рубец", "спустя всего месяц после операции"]:
        d.text((cx + (D + 14) / 2, yc), ln, font=fc, fill=WHITE, anchor="ma"); yc += 34
    # «Подробнее…» straight, in the italic Montserrat
    fi = ImageFont.truetype(os.path.join(ROOT, "fonts/Montserrat-Italic-VariableFont_wght.ttf"), 34); fi.set_variation_by_axes([600])
    d.text((W // 2, H - 90), "Подробнее про рубцы в актуальном «Рубцы»", font=fi, fill=WHITE, anchor="mm")
    a = lay.getchannel("A")                                  # soft shadow, a bit denser now that there is no haze
    sh = Image.new("RGBA", (W, H), GRAPHITE + (0,)); sh.putalpha(a.point(lambda v: v * 200 // 255).filter(ImageFilter.GaussianBlur(9)))
    sh2 = Image.new("RGBA", (W, H), GRAPHITE + (0,)); sh2.putalpha(a.point(lambda v: v * 170 // 255).filter(ImageFilter.GaussianBlur(3)))
    lay2 = Image.alpha_composite(Image.alpha_composite(sh, sh2), lay)
    fr = over(fr, np.asarray(lay2, np.float32) / 255)
    out = os.path.join(OUT, "02_rubcy.jpg"); Image.fromarray((np.clip(fr, 0, 1) * 255).astype(np.uint8)).save(out, quality=94); print("wrote", out)


if not ONLY or 1 in ONLY: s1()
if not ONLY or 2 in ONLY: s2()
