"""Carousel post (1080x1350) cut from the reel 'Увеличение груди без имплантов' (lipofilling).

Post style of the brand: Bebas Neue headlines in powder, thin Montserrat Light text, navy tone,
ANNA TVAURI mark on top. Frames are taken from the reel and cropped below its baked overlays
(logo, word bubble, Instagram icons); the small bottom-left Instagram tag is inpainted.

usage: python3 post_lipofilling_reel.py REEL.mp4 OUT_DIR
"""
import os, sys, subprocess, tempfile
import numpy as np, cv2
from PIL import Image, ImageFilter, ImageDraw

REEL, OUT = sys.argv[1:3]
TMP = tempfile.mkdtemp()
first = os.path.join(TMP, "f.png")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "3.1", "-i", REEL, "-frames:v", "1", first], check=True)
sys.argv = [sys.argv[0], first, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surgeon_style as ss  # noqa: E402  (brand helpers: Slide, fonts, logo, icons, palette)
from surgeon_style import Slide, bebas, mont, logo, vshade, icon_circle, POWDER, NAVY, DEEP, WHITE, INK, S  # noqa: E402

PW, PH = 1080, 1350


def frame(t, inpaint=()):
    p = os.path.join(TMP, f"{t}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t}", "-i", REEL, "-frames:v", "1", p], check=True)
    a = cv2.imread(p)
    if inpaint:
        m = np.zeros(a.shape[:2], np.uint8)
        for x0, y0, x1, y1 in inpaint: m[y0:y1, x0:x1] = 255
        a = cv2.inpaint(a, m, 9, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))


def donor_frame(t=2.0, axis=374, bubble=(156, 350, 118)):
    """the donor-zone illustration is symmetric: the word bubble sitting on its left flank is
    replaced by the mirrored right half (axis found by flip-correlation); the crop starts below the IG tag"""
    im = np.asarray(frame(t)).copy()
    cx, cy, r = bubble; yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    src_x = np.clip(2 * axis - xx, 0, im.shape[1] - 1)
    mir = im[yy, src_x]
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 4)[..., None]
    im = (im * (1 - soft) + mir * soft).astype(np.uint8)
    return Image.fromarray(im)


def crop45(im, x0, y0, w):
    return im.crop((x0, y0, x0 + w, y0 + int(w * 1.25)))


def grade(im, w, h, tint=NAVY, k=0.18, dark=0.0):
    """upscale with lanczos + gentle sharpening, a light navy cast to sit in the brand palette"""
    im = im.resize((w * S, h * S), Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=2 * S, percent=60, threshold=2))
    a = np.asarray(im).astype(np.float32) / 255
    a = a * (1 - k) + np.float32(tint) / 255 * k
    a *= 1 - dark
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))


def photo_slide(img, name, head, text, n, top_shade=0.55, y_head=None, size=104):
    sl = Slide(PW, PH, grade(img, PW, PH))
    vshade(sl, 0, 230, top_shade, 0.0, DEEP)
    vshade(sl, 640, 1180, 0.0, 0.9, DEEP)
    logo(sl, 92)
    lh = size * 0.9
    y = y_head if y_head is not None else 1200 - len(head) * lh - len(text) * 44 - 40
    y = sl.lines((80, y), head, bebas(size), POWDER, lh=0.9)
    sl.lines((84, y + 26), text, mont(36, 300), WHITE, lh=1.22)
    counter(sl, n)
    return sl.save(name)


def counter(sl, n, total=7, col=WHITE):
    sl.text((PW - 80, 1286), f"{n:02d} / {total:02d}", mont(24, 400), col, "rs", a=170, spacing=2)


def arch_photo(sl, img, box):
    x0, y0, x1, y1 = box; w, h = int((x1 - x0) * S), int((y1 - y0) * S)
    ph = grade(img, x1 - x0, y1 - y0, k=0.1)
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1 + w // 2), radius=w // 2, fill=255)
    sl.shadow(box, 40, 22, 120)
    sl.im.paste(ph, (int(x0 * S), int(y0 * S)), m)


def main():
    os.makedirs(OUT, exist_ok=True)
    ig = ((14, 740, 210, 860),)  # bottom-left Instagram tag on the OR shots
    out = []

    # 1 cover: the surgeon on camera
    out.append(photo_slide(crop45(frame(3.1), 32, 460, 656), "01_cover.jpg",
                           ["УВЕЛИЧЕНИЕ ГРУДИ", "БЕЗ ИМПЛАНТОВ"],
                           ["и без видимых рубцов.", "Многие даже не догадываются,", "что так можно"], 1))
    # 2 donor zones
    out.append(photo_slide(crop45(donor_frame(), 0, 380, 720), "02_donor.jpg",
                           ["ЖИР ИЗ ДОНОРСКИХ ЗОН"], ["Забираем жир из ваших", "донорских зон"], 2))
    # 3 into the breast
    out.append(photo_slide(crop45(frame(3.6), 32, 460, 656), "03_transfer.jpg",
                           ["ПЕРЕСАЖИВАЕМ", "В ОБЛАСТЬ ГРУДИ"], ["Собственная ткань", "вместо импланта"], 3))
    # 4 syringes
    out.append(photo_slide(crop45(frame(9.4, ig), 32, 460, 656), "04_syringes.jpg",
                           ["ГОТОВИМ ЖИР", "К ВВЕДЕНИЮ"], ["Собранный жир перекачиваем", "в специальные большие шприцы"], 4))
    # 5 equal volume
    out.append(photo_slide(crop45(frame(10.4, ig), 32, 460, 656), "05_volume.jpg",
                           ["ОДИНАКОВЫЙ ОБЪЁМ"], ["В каждую железу вводим точное", "количество миллилитров, чтобы", "объём был одинаковым в обеих"], 5))

    # 6 asymmetry: surgeon in an arch on navy (her face sits under the reel's word bubble,
    #   so only a narrow clean crop exists; the arch shows it at its native sharpness)
    sl = Slide(PW, PH); sl.rrect((0, 0, PW, PH), 0, NAVY)
    vshade(sl, 0, PH, 0.0, 0.45, DEEP)
    logo(sl, 92)
    y = sl.lines((80, 200), ["А ЕСЛИ ЕСТЬ", "АСИММЕТРИЯ?"], bebas(104), POWDER, lh=0.9)
    sl.lines((84, y + 26), ["Тогда в железы вводим", "разный объём и выравниваем", "форму груди"], mont(36, 300), WHITE, lh=1.22)
    arch_photo(sl, frame(12.2).crop((290, 250, 720, 790)), (540, 640, 1000, 1218))
    sl.dashed(ss.bez((300, 760), (300, 960), (420, 1040), (520, 1000)), POWDER, 2, 10, 9, 200)
    sl.head(ss.bez((300, 760), (300, 960), (420, 1040), (520, 1000)), POWDER, 2, 14)
    counter(sl, 6)
    out.append(sl.save("06_asymmetry.jpg"))

    # 7 CTA
    bg = grade(frame(13.0).crop((0, 460, 720, 1280)).resize((656, 820)).filter(ImageFilter.GaussianBlur(10)), PW, PH, k=0.55, dark=0.5)
    sl = Slide(PW, PH, bg)
    logo(sl, 110)
    y = sl.lines((PW / 2, 300), ["ПОДРОБНЕЕ", "О ЛИПОФИЛИНГЕ"], bebas(120), POWDER, lh=0.9, anchor="ma")
    sl.poly([(PW / 2 - 70, y + 34), (PW / 2 + 70, y + 34)], POWDER, 2)
    sl.lines((PW / 2, y + 80), ["в актуальном «Липофилинг»", "у меня в профиле"], mont(40, 300), WHITE, lh=1.25, anchor="ma")
    box = (150, y + 270, PW - 150, y + 470); sl.shadow(box, 36, 20, 110); sl.rrect(box, 36, WHITE, 22, outline=POWDER, w=1.2)
    icon_circle(sl, (box[0] + 90, (box[1] + box[3]) / 2), "chat", 34)
    sl.lines((box[0] + 150, box[1] + 50), ["Запись на консультацию —", "в Direct"], mont(36, 300), WHITE, lh=1.25)
    sl.text((PW / 2, box[3] + 110), "@dr_tvaurianna", mont(34, 400), POWDER, "ma")
    counter(sl, 7)
    out.append(sl.save("07_cta.jpg"))

    ss.sheet(out, 4, 540, "00_overview.jpg")
    print("ok")


if __name__ == "__main__":
    main()
