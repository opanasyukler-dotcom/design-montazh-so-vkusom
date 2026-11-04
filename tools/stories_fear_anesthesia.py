"""Instagram stories (1080x1920): the poll about the biggest fear before the operation -> anesthesia leads ->
the patients' emotions after the operation -> the highlight «Наркоз».

Old story design: Inter Bold caps headlines + Montserrat, white text with a soft shadow, no emojis, no dashes.
1. 01_opros.jpg: the operating photo, the question on top, the middle left free for the Instagram poll sticker.
2. 02_rezultat_oprosa.jpg: free room for the screenshot of the poll results, the comment below, «>>>».
5. 05_aktualnoe_narkoz.jpg: the surgeon's portrait, the call to the highlight in navy on the white jacket.
(3 and 4 are the patients' videos after the operation.)

Sources (SRC_DIR): operaciya.png (hands with the needle holder), portret.jpg (the surgeon).
usage: python3 stories_fear_anesthesia.py SRC_DIR OUT_DIR [1|2|5 ...]
"""
import os, subprocess, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

W, H = 1080, 1920
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
WHITE, NAVY, GRAPHITE = (255, 255, 255), (2, 31, 71), (20, 23, 31)
SRC, OUT = sys.argv[1], sys.argv[2]
ONLY = {int(a) for a in sys.argv[3:]}
os.makedirs(OUT, exist_ok=True)
MARGIN = 84


def mont(size, wght=400):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([wght]); return f


def cover(name):
    im = ImageOps.exif_transpose(Image.open(os.path.join(SRC, name))).convert("RGB")
    s = max(W / im.width, H / im.height); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im.crop(((im.width - W) // 2, (im.height - H) // 2, (im.width - W) // 2 + W, (im.height - H) // 2 + H))


def lines_draw(lay, lines, x, y, fill=WHITE, anchor="la"):
    d = ImageDraw.Draw(lay)
    for t, f, step in lines:
        d.text((x, y), t, font=f, fill=fill, anchor=anchor); y += step
    return y


def shadowed(bg, lay, a=170, blur=9):
    """white text with the usual soft graphite shadow"""
    al = lay.getchannel("A")
    sh = Image.new("RGBA", (W, H), GRAPHITE + (0,)); sh.putalpha(al.point(lambda v: v * a // 255).filter(ImageFilter.GaussianBlur(blur)))
    sh2 = Image.new("RGBA", (W, H), GRAPHITE + (0,)); sh2.putalpha(al.point(lambda v: v * 140 // 255).filter(ImageFilter.GaussianBlur(3)))
    return Image.alpha_composite(Image.alpha_composite(Image.alpha_composite(bg.convert("RGBA"), sh), sh2), lay)


def top_fade(im, h, k):
    a = np.asarray(im, np.float32); ys = np.arange(H, dtype=np.float32)[:, None, None]
    a *= 1 - k * np.clip(1 - ys / h, 0, 1) ** 1.6
    return Image.fromarray(a.astype(np.uint8))


def save(im, name):
    out = os.path.join(OUT, name); im.convert("RGB").save(out, quality=94); print("wrote", out)


def text_overlay(lay):
    """the text layer with its soft shadow, as float RGBA"""
    return np.asarray(shadowed(Image.new("RGBA", (W, H), (0, 0, 0, 0)), lay), np.float32) / 255


def render_video(src, ov, shade, name, fix=None):
    """the clip scaled to 1080x1920, darkened by shade (H,1,1), the static text layer on top; the clip's own sound"""
    out = os.path.join(OUT, name); tmp = out + ".v.mp4"
    fps = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
                          os.path.join(SRC, src)], capture_output=True, text=True).stdout.strip()
    sw, sh_ = 720, 1280
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", os.path.join(SRC, src), "-vf", f"scale={sw}:{sh_}", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", fps, "-i", "-",
                            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    oa = ov[..., 3:4]; n = 0; state = {}
    while True:
        b = p.stdout.read(sw * sh_ * 3)
        if len(b) < sw * sh_ * 3: break
        f = np.frombuffer(b, np.uint8).reshape(sh_, sw, 3).copy()
        if fix: f = fix(f, state)
        f = cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        f = f * (1 - shade); f = f * (1 - oa) + ov[..., :3] * oa
        o = (np.clip(f, 0, 1) * 255).astype(np.uint8); enc.stdin.write(o.tobytes())
        if n == 15: Image.fromarray(o).save(out[:-4] + "_kadr.jpg", quality=90)
        n += 1
    p.wait(); enc.stdin.close(); enc.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", os.path.join(SRC, src), "-map", "0:v", "-map", "1:a?", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out], check=True)
    os.remove(tmp); print("wrote", out, flush=True)


def hide_patient(f, state):
    """IMG_0032: the patient on the table at the bottom of the frame (and any blood) under a heavy blur. The mask: skin
    tones in the lower part of the frame plus red anywhere below the doctor's face, closed, grown and smoothed in time"""
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV); h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ys = np.arange(f.shape[0])[:, None]
    skin = (h_ >= 3) & (h_ <= 24) & (s_ >= 35) & (v_ >= 60) & (ys >= 960)
    red = ((h_ < 8) | (h_ > 170)) & (s_ > 110) & (v_ > 40) & (ys >= 760)
    m = (skin | red).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    m = cv2.dilate(m, np.ones((45, 45), np.uint8)).astype(np.float32) / 255
    prev = state.get("m"); m = m if prev is None else np.maximum(m, prev * 0.85)        # no flicker at the edges
    state["m"] = m
    a = cv2.GaussianBlur(m, (0, 0), 14)[..., None]
    bl = cv2.GaussianBlur(cv2.resize(cv2.resize(f, (36, 64), interpolation=cv2.INTER_AREA), (f.shape[1], f.shape[0])), (0, 0), 12)
    return (f * (1 - a) + bl * a).astype(np.uint8)


def s1():
    """the poll over the anesthesiologist at work (IMG_0032); the patient and blood blurred"""
    ys = np.arange(H, dtype=np.float32)[:, None, None]; shade = 0.55 * np.clip(1 - ys / 760, 0, 1) ** 1.6
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hb = ImageFont.truetype(BOLD, 64)
    y = lines_draw(lay, [("ОПРОС", mont(30, 600), 70)], MARGIN, 150)
    ImageDraw.Draw(lay).line((MARGIN, y - 22, MARGIN + 70, y - 22), fill=WHITE, width=3)
    lines_draw(lay, [("КАКОЙ СТРАХ", hb, 76), ("ОСТАНАВЛИВАЕТ ВАС", hb, 76), ("ОТ ОПЕРАЦИИ?", hb, 104)], MARGIN, y + 6)
    render_video("IMG_0032.mp4", text_overlay(lay), shade, "01_opros.mp4", fix=hide_patient)   # middle free for the poll


def s2():
    """the surgeon in loupes (IMG_9379); the comment low on the gown, under the chin, so the face stays free"""
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hb = ImageFont.truetype(BOLD, 58); b = mont(38, 400)
    lines = [("ВИЖУ, ЧТО СТРАХ", hb, 70), ("АНЕСТЕЗИИ ЛИДИРУЕТ", hb, 96),
             ("Очень хочу показать вам", b, 50), ("эмоции моих пациенток", b, 50), ("после операции", b, 50)]
    top = H - 110 - sum(s for _, _, s in lines)
    lines_draw(lay, lines, MARGIN, top)
    ImageDraw.Draw(lay).text((W - MARGIN, H - 80), ">>>", font=mont(44, 500), fill=WHITE, anchor="rs")
    ys = np.arange(H, dtype=np.float32)[:, None, None]; shade = 0.6 * np.clip((ys - (top - 80)) / 300, 0, 1)
    render_video("IMG_9379.mp4", text_overlay(lay), shade, "02_rezultat_oprosa.mp4")


def s5():
    bg = cover("portret.jpg")
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    hb = ImageFont.truetype(BOLD, 64); b = mont(37, 500)
    lines = [("ВСЁ ПРО НАРКОЗ", hb, 84), ("Как проходит анестезия,", b, 48),
             ("подготовка и пробуждение,", b, 48), ("смотрите в актуальном", b, 48), ("«Наркоз»", mont(37, 700), 0)]
    top = H - 260 - 84 - 48 * 3 - 34
    for (t, f, st), yy in zip(lines, np.cumsum([0] + [s for _, _, s in lines[:-1]])):
        d.text((W - MARGIN, top + yy), t, font=f, fill=NAVY + (255,), anchor="ra")
    save(Image.alpha_composite(bg.convert("RGBA"), lay), "05_aktualnoe_narkoz.jpg")


for k, f in [(1, s1), (2, s2), (5, s5)]:
    if not ONLY or k in ONLY: f()
