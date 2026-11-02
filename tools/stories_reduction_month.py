"""3 simple stories (1080x1920): breast reduction (autoimplant), result after one month.

1. the patient's own words over the one-month video (from the frontal view at 0:17.5, face blurred, voice gently
   changed) + «наше до» circle low on the left whose before body turns with her (front -> 3/4 -> side photos).
2. the thesis: the way to the dream started with this before; the reduction by the autoimplant (Ribeiro) method.
3. one video: the before body (neck to fingertips) -> a slider -> the after body (same framing, same scale), then the
   clinic's one-week collages and the same collages made with the one-month photos; the bottom stays free for the
   «Записаться на консультацию» button.

Old story design: Inter Bold caps headlines + Montserrat, navy + white, black fade under the text, static text.
Sources (drive/ from the Drive folder 1EyuB8yLY1E7px4mXB37EBpnS-B6fB_zJ): IMG_2047.MP4 one-month video
(month_blur.mp4 = 21-33 s through face_blur_track.py), IMG_9150-9152 the clinic's one-week collages (after on top,
before below), IMG_2029 / 2045 / 2042 one-month photos (front / 3/4 / side; <name>.mask.png = rembg cut-out).

usage: python3 stories_reduction_month.py SRC_DIR OUT_DIR [1|2|3] [notext]   (notext: *_bez_teksta.mp4)
"""
import os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

W, H, FPS = 1080, 1920, 30
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
WHITE, NAVY, GRAPHITE = (255, 255, 255), (2, 31, 71), (20, 23, 31)
SRC, OUT = sys.argv[1], sys.argv[2]
ONLY = {int(a) for a in sys.argv[3:] if a.isdigit()}
NOTEXT = "notext" in sys.argv[3:]          # clean versions: no captions, labels or fade (the circle and collages stay)
SUF = "_bez_teksta" if NOTEXT else ""
DRIVE = os.path.join(SRC, "drive")
os.makedirs(OUT, exist_ok=True)
MARGIN = 84
VOICE = ("rubberband=pitch=1.16:formant=shifted:transients=smooth,"      # the same gentle disguise as in the earlier series
         "rubberband=pitch=0.957:formant=preserved:transients=smooth")


def mont(size, wght=400):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([wght]); return f


def photo(name): return ImageOps.exif_transpose(Image.open(os.path.join(DRIVE, name))).convert("RGB")


def shadowed_text(lines, x, y, layer):
    """lines: [(text, font, step)]; white text with the usual soft shadow"""
    txt = Image.new("RGBA", layer.size, (0, 0, 0, 0)); d = ImageDraw.Draw(txt)
    for t, f, step in lines:
        d.text((x, y + step / 2), t, font=f, fill=WHITE + (255,), anchor="lm"); y += step
    sh = Image.new("RGBA", layer.size, GRAPHITE + (0,)); sh.putalpha(txt.getchannel("A").point(lambda v: v * 150 // 255))
    layer.alpha_composite(sh.filter(ImageFilter.GaussianBlur(9))); layer.alpha_composite(txt)
    return y


def pill(text, size=28, wght=400):
    f = mont(size, wght); h = int(size * 1.75); w = int(f.getlength(text)) + 48
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), h // 2, fill=NAVY + (235,))
    d.text((w / 2, h / 2 + 1), text, font=f, fill=WHITE + (255,), anchor="mm"); return im


def fade(top, length=360, a=0.85):
    ys = np.arange(H, dtype=np.float32); p = np.clip((ys - (top - length)) / (length + 60), 0, 1); p = p * p * (3 - 2 * p)
    return (a * p)[:, None, None]


def warm(img, da=4.0, db=9.0, dl=2.0):
    """a little warmth on the skin: a/b towards peach on warm pixels only (the background and briefs keep their colour)"""
    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    m = np.clip((lab[..., 2] - 128) / 8, 0, 1) * np.clip((lab[..., 0] - 50) / 40, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 2)
    lab[..., 0] += dl * m; lab[..., 1] += da * m; lab[..., 2] += db * m
    return Image.fromarray(cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))


def on_black(name):
    """the one-month photo cut out onto black (rembg mask), like the clinic's own photos"""
    im = np.asarray(photo(name), np.float32); m = np.asarray(Image.open(os.path.join(DRIVE, name + ".mask.png")).convert("L"), np.float32) / 255
    return Image.fromarray((im * m[..., None]).astype(np.uint8))


# ---------------- story 1 ----------------
def s1():
    src = os.path.join(DRIVE, "month_blur17.mp4"); out = os.path.join(OUT, f"01_odin_v_odin{SUF}.mp4"); tmp = out + ".v.mp4"
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                               capture_output=True, text=True).stdout)
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    # the quote and what it is, low under the usual fade
    hb, bb = ImageFont.truetype(BOLD, 64), mont(38, 500); sm = mont(34, 400)
    lines = [("«КАК Я И ХОТЕЛА", hb, 76), ("ОДИН В ОДИН»", hb, 92), ("результат спустя МЕСЯЦ", bb, 52),
             ("была выполнена операция по уменьшению", sm, 44), ("груди методом аутоимпланта", sm, 46)]
    top = H - 90 - sum(s for _, _, s in lines)
    if not NOTEXT: shadowed_text(lines, MARGIN, top, ly)
    shade = fade(top + 40, 380, 0.85) if not NOTEXT else np.zeros((H, 1, 1), np.float32)
    # «наше до»: low on the left, above the text: the before video (IMG_0713) in the circle, turning with her.
    # Her angles: here front 0-4 s, side at 9 s, back at 13.5 s (abs 17.5 / 26.5 / 31 s of IMG_2047);
    # in IMG_0713 front 9.6-10.4 s, side 16.7 s, back 21.5 s -> piecewise-linear time map
    D = 380; R = D // 2 + 6
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(DRIVE, "IMG_0713.MOV"),
                          "-vf", f"crop=2400:2400:-120:300,scale={D}:{D},fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    if not raw:                                             # crop larger than the frame: pad first
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(DRIVE, "IMG_0713.MOV"),
                              "-vf", f"pad=2400:ih:120:0,crop=2400:2400:0:300,scale={D}:{D},fps={FPS}", "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    bvid = np.frombuffer(raw, np.uint8).reshape(-1, D, D, 3)
    MAP = [(0.0, 9.6), (4.0, 10.4), (9.0, 16.7), (13.5, 21.5), (15.6, 23.6)]
    ring = Image.new("RGBA", (D + 12, D + 12), (0, 0, 0, 0)); ImageDraw.Draw(ring).ellipse((0, 0, D + 11, D + 11), fill=WHITE + (255,))
    ring = np.asarray(ring, np.float32) / 255
    m = Image.new("L", (D, D), 0); ImageDraw.Draw(m).ellipse((0, 0, D - 1, D - 1), fill=255); m = np.asarray(m, np.float32)[..., None] / 255
    ccx, ccy = 40, (top - 64 - (D + 12)) if not NOTEXT else H - 120 - (D + 12)
    if not NOTEXT:
        f32 = mont(32, 300); shadowed_text([("наше до", f32, 44)], ccx + R - int(f32.getlength("наше до")) // 2, ccy + D + 18, ly)

    def ring_at(t):
        tb = np.interp(t, [a for a, _ in MAP], [b for _, b in MAP])
        fr = bvid[min(int(tb * FPS), len(bvid) - 1)].astype(np.float32) / 255
        out = ring.copy(); out[6:6 + D, 6:6 + D, :3] = out[6:6 + D, 6:6 + D, :3] * (1 - m) + fr * m
        return out
    ov = np.asarray(ly, np.float32) / 255; oa = ov[..., 3:4]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf", f"scale={W}:{H}:flags=lanczos,fps={FPS}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    n = int(round(dur * FPS)); last = None
    for i in range(n):
        b = p.stdout.read(W * H * 3)
        if len(b) == W * H * 3: last = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        z = 1.0 + 0.03 * i / n; M = cv2.getRotationMatrix2D((W / 2, H * 0.55), 0, z)
        fr = cv2.warpAffine(last, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255
        fr = fr * (1 - shade); fr = fr * (1 - oa) + ov[..., :3] * oa
        rg = ring_at(i / FPS); al = rg[..., 3:4]; h, w = rg.shape[:2]
        fr[ccy:ccy + h, ccx:ccx + w] = fr[ccy:ccy + h, ccx:ccx + w] * (1 - al) + rg[..., :3] * al
        o = (np.clip(fr, 0, 1) * 255).astype(np.uint8); enc.stdin.write(o.tobytes())
        if i == int(3 * FPS): Image.fromarray(o).save(out[:-4] + ".jpg", quality=90)
    p.kill(); enc.stdin.close(); enc.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", src, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", VOICE + f",loudnorm=I=-16:TP=-1.5,afade=t=out:st={dur - 0.4:.2f}:d=0.4", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    os.remove(tmp); print("wrote", out, flush=True)


# ---------------- story 2: the thesis ----------------
def s_thesis():
    """the surgeon at the marking (her own words: gigantomastia, 32 cm, the Ribeiro / autoimplant method) with the thesis
    low under the usual fade; the clinic's word captions of that video stay"""
    src = os.path.join(DRIVE, "surgeon_crop.mp4"); out = os.path.join(OUT, "02_put_k_mechte.mp4"); tmp = out + ".v.mp4"
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                               capture_output=True, text=True).stdout)
    hb = ImageFont.truetype(BOLD, 52); b = mont(31, 400); bm = mont(31, 600)
    lines = [("ПУТЬ К МЕЧТЕ", hb, 62), ("НАЧАЛСЯ С ТАКОГО ДО", hb, 78),
             ("Уменьшение груди методом аутоимпланта", bm, 42), ("(метод Рибейро):", bm, 52),
             ("из собственных тканей нижней части груди", b, 41), ("формируем «аутоимплант» и поднимаем его вверх.", b, 41),
             ("Он наполняет верхний полюс и держит форму,", b, 41), ("красивый объём без силиконового импланта.", b, 41)]
    top = H - 70 - sum(s for _, _, s in lines)
    ly = Image.new("RGBA", (W, H), (0, 0, 0, 0)); shadowed_text(lines, MARGIN, top, ly)
    ov = np.asarray(ly, np.float32) / 255; oa = ov[..., 3:4]; shade = fade(top + 40, 420, 0.88)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf", f"scale={W}:{H},fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    last = None
    for i in range(int(round(dur * FPS))):
        bb = p.stdout.read(W * H * 3)
        if len(bb) == W * H * 3: last = np.frombuffer(bb, np.uint8).reshape(H, W, 3)
        fr = last.astype(np.float32) / 255 * (1 - shade); fr = fr * (1 - oa) + ov[..., :3] * oa
        o = (fr * 255).astype(np.uint8); enc.stdin.write(o.tobytes())
        if i == int(6 * FPS): Image.fromarray(o).save(out[:-4] + ".jpg", quality=90)
    p.kill(); enc.stdin.close(); enc.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", src, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", f"loudnorm=I=-16:TP=-1.5,afade=t=out:st={dur - 0.4:.2f}:d=0.4", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    os.remove(tmp); print("wrote", out, flush=True)


# ---------------- story 3 ----------------
AREA = (150, 1650)            # picture band; under it (1650..1920) the free room for the consultation button


def fit_band(img, area=AREA):
    """an image on black scaled into the picture band, centred"""
    h = area[1] - area[0]; s = h / img.height; im = img.resize((int(img.width * s), h), Image.LANCZOS)
    cv = Image.new("RGB", (W, H), (0, 0, 0)); cv.paste(im, ((W - im.width) // 2, area[0])); return np.asarray(cv, np.float32) / 255


BODY = (0, H)                 # the before / after bodies on the whole screen: neck down to the knees


def month_collage(after_name, after_crop, week_name):
    """the clinic's one-week collage layout, after = the one-month photo: after (on black) on top, caption, the same
    before half below (copied from the one-week collage)"""
    wk = photo(week_name); cw, ch = wk.size
    cv = Image.new("RGB", (cw, ch), (0, 0, 0))
    TOP = 1095                                    # the after photo ends above the caption, as on the clinic's collages
    a = warm(on_black(after_name)).crop(after_crop); s = TOP / a.height; a = a.resize((int(a.width * s), TOP), Image.LANCZOS)
    if a.width > cw: a = a.crop(((a.width - cw) // 2, 0, (a.width - cw) // 2 + cw, TOP))
    cv.paste(a, ((cw - a.width) // 2, 0))
    cv.paste(wk.crop((0, 1255, cw, ch)), (0, 1255))
    # caption in the clinic's own type (Inter SemiBold): size and line centres measured on that one-week collage
    a = np.asarray(wk).astype(int); spans = []
    for r0, r1 in [(1105, 1172), (1176, 1250)]:
        ys, xs = np.nonzero(a[r0:r1].min(2) > 225); spans.append(((xs.min() + xs.max()) / 2, r0 + (ys.min() + ys.max()) / 2, xs.max() - xs.min()))
    f = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-SemiBold.otf", round(60 * spans[0][2] / 517))
    d = ImageDraw.Draw(cv)
    for (cx, cy, _), t in zip(spans, ["Результат спустя", "месяц после операции"]):
        d.text((cx, cy), t, font=f, fill=WHITE, anchor="mm")
    return cv


def s2():
    out = os.path.join(OUT, f"03_do_posle{SUF}.mp4"); tmp = out + ".v.mp4"
    # same framing for both: neck to fingertips (before: neck 1270 .. fingertips 2190 in IMG_9150;
    # after: neck 70 .. fingertips 1540 in IMG_2029 -> scale 0.626), body centred
    # before: the frontal frame of the before video (IMG_0713 at 10.0 s: neck 384, fingertips 2750, axis x 1080), from
    # under the chin to the bottom of the frame (the knees); after: IMG_2029 (neck 70, fingertips 1540, axis 790) at /1.61
    fb = subprocess.run(["ffmpeg", "-v", "error", "-ss", "10.0", "-i", os.path.join(DRIVE, "IMG_0713.MOV"), "-frames:v", "1",
                         "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    import io
    before = fit_band(Image.open(io.BytesIO(fb)).convert("RGB").crop((93, 330, 2067, 3840)), BODY)
    after = fit_band(warm(photo("IMG_2029.JPG")).crop((177, 37, 1403, 2217)), BODY)
    labs = {k: np.asarray(pill(k, 30), np.float32) / 255 for k in ["до", "после"]}
    month = [month_collage("IMG_2029.JPG", (78, 60, 1502, 1500), "IMG_9150.PNG"),
             month_collage("IMG_2045.JPG", (40, 190, 1480, 1630), "IMG_9151.PNG"),
             month_collage("IMG_2042.JPG", (0, 130, 1440, 1570), "IMG_9152.PNG")]
    for i, im in enumerate(month): im.save(os.path.join(OUT, f"kollazh_mesyac_{i + 1}.jpg"), quality=94)
    slides = [fit_band(im, (0, H)) for im in month]                                 # only the one-month collages, whole screen                          # the collages on the whole screen

    def blit(fr, a, x, y, k=1.0):
        h, w = a.shape[:2]; al = a[..., 3:4] * k; fr[y:y + h, x:x + w] = fr[y:y + h, x:x + w] * (1 - al) + a[..., :3] * al

    T_B, T_S, T_A, T_C, X = 1.6, 1.4, 1.6, 1.9, 0.35          # before, slide, after, each collage, crossfade
    total = T_B + T_S + T_A + len(slides) * T_C
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for i in range(int(round(total * FPS))):
        t = i / FPS
        if t < T_B + T_S + T_A:
            p = min(max((t - T_B) / T_S, 0), 1); p = p * p * (3 - 2 * p)
            xs = int(W * p)                                       # the slider moves left -> right, the after takes over
            fr = before.copy(); fr[:, :xs] = after[:, :xs]
            if 0 < p < 1:                                         # the handle: thin white line + a ring with arrows
                fr[BODY[0]:BODY[1], max(xs - 2, 0):xs + 2] = 1.0              # the slider line, top to bottom
                ring = Image.new("RGBA", (84, 84), (0, 0, 0, 0)); d = ImageDraw.Draw(ring)
                d.ellipse((2, 2, 81, 81), outline=WHITE + (255,), width=4, fill=(0, 0, 0, 90))
                d.polygon([(30, 42), (40, 32), (40, 52)], fill=WHITE + (255,)); d.polygon([(54, 42), (44, 32), (44, 52)], fill=WHITE + (255,))
                blit(fr, np.asarray(ring, np.float32) / 255, min(max(xs - 42, 0), W - 84), (BODY[0] + BODY[1]) // 2 - 42)
            if p < 1 and not NOTEXT: blit(fr, labs["до"], W - 40 - labs["до"].shape[1], BODY[0] + 60)
            if p > 0 and not NOTEXT: blit(fr, labs["после"], 40, BODY[0] + 60)
        else:
            tc = t - (T_B + T_S + T_A); k = min(int(tc // T_C), len(slides) - 1); lt = tc - k * T_C
            fr = slides[k].copy()
            if k > 0 and lt < X: q = lt / X; fr = slides[k - 1] * (1 - q) + fr * q
            elif k == 0 and lt < X: q = lt / X; fr = after * (1 - q) + fr * q
        o = (np.clip(fr, 0, 1) * 255).astype(np.uint8); enc.stdin.write(o.tobytes())
        if i == int((T_B + T_S * 0.5) * FPS): Image.fromarray(o).save(out[:-4] + ".jpg", quality=90)
    enc.stdin.close(); enc.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c", "copy", "-movflags", "+faststart", out], check=True)
    os.remove(tmp); print("wrote", out, f"{total:.1f}s", flush=True)


if not ONLY or 1 in ONLY: s1()
if (not ONLY or 2 in ONLY) and not NOTEXT: s_thesis()
if not ONLY or 3 in ONLY: s2()
