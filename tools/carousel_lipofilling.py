"""Instagram carousel (1080x1350) cut from the finished lipofilling reel (IMG_8380.MP4).

Each video slide is one block of the reel with its own headline and voice,
cropped to 4:5 so the headline, subtitles and body all stay in frame.
The last slide is a consultation card in the Tvauri style.

usage: python3 carousel_lipofilling.py REEL.MP4 SURGEON_PHOTO_DIR OUT_DIR
"""
import os, sys, subprocess

REEL, PHOTOS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], PHOTOS, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carousel_tvauri as ct  # noqa: E402  (shared fonts, palette, text blocks)

W, H = 1080, 1350
CROP_Y = 60  # keeps the reel's headline band and the body below it
# block boundaries = hard cuts in the reel (headline changes)
SLIDES = [
    ("01_implants_no", 0.00, 4.93),
    ("02_lift_no", 4.93, 10.67),
    ("03_chose_lipofilling", 10.67, 19.20),
    ("04_donor_zones", 19.20, 23.07),
    ("05_volume", 23.07, 29.20, "fit"),
    ("06_not_a_lift", 29.20, 36.60, "fit"),
    ("07_this_task", 36.60, None, "fit"),
]


G = "0x%02x%02x%02x" % ct.GRAPHITE


def cut(name, a, b, mode="crop"):
    dur = (b or float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", REEL],
                                     capture_output=True, text=True).stdout)) - a
    fo = max(dur - 0.12, 0)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", REEL,
                    "-vf", (f"crop={W}:{H}:0:{CROP_Y},setsar=1" if mode == "crop" else
                            f"scale=-2:{H}:flags=lanczos,pad={W}:{H}:(ow-iw)/2:0:color={G},setsar=1"),  # whole frame: content sits low
                    "-af", f"afade=t=in:d=0.06,afade=t=out:st={fo:.3f}:d=0.12",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", f"{OUT}/{name}.mp4"], check=True)


def cta():
    """surgeon from the reel on top, consultation card below (same layout as the lift carousel)"""
    from PIL import Image, ImageDraw
    import numpy as np
    PH = 800
    im = ct.load_photo("DSC00888.JPG"); sc = W / im.width; sh = im.height * sc; off = 0.17 * sh
    ph = im.resize((W, round(sh)), Image.LANCZOS).crop((0, int(off), W, int(off) + PH)).convert("RGBA")
    c = Image.new("RGBA", (W, H), ct.WHITE + (255,)); c.alpha_composite(ph, (0, 0))
    ct.put_logo(c, y=64, alpha=0.95)
    arr = np.zeros((H, W, 4), np.uint8); arr[..., :3] = 255
    p = np.clip((np.arange(H)[:, None] - (PH - 150)) / 150, 0, 1); p = np.broadcast_to(p, (H, W))
    arr[..., 3] = (p * p * (3 - 2 * p) * 255).astype(np.uint8)
    c.alpha_composite(Image.fromarray(arr, "RGBA"))
    bw = 840; x = (W - bw) // 2
    b = ct.Block(bw, "center").head("На консультации разберём", 60, 10, color=ct.INK).rule(0.14, 22)
    y = b.draw(c, x, PH - 50)
    items = ["подойдёт ли вам липофилинг", "есть ли донорские зоны", "какой объём можно добавить",
             "нужна ли подтяжка или импланты"]
    f = ct.mont(28, 500); d = ImageDraw.Draw(c)
    for it in items:
        tw = f.getlength(it); x0 = (W - tw) / 2
        d.ellipse((x0 - 24, y + 12, x0 - 14, y + 22), fill=(214, 160, 222)); d.text((x0, y), it, font=f, fill=ct.INK); y += 42
    y += 22
    ct.Block(bw, "center").ital("Для записи напишите в Direct «КОНСУЛЬТАЦИЯ» или свяжитесь с нами по телефону +7 926 636 30 00 в WhatsApp, Telegram или MAX.", 27, 0, color=ct.MIST).draw(c, x, y)
    ct.save_jpg(c, f"{OUT}/08_cta.jpg")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for n, a, b, *m in SLIDES:
        cut(n, a, b, *m); print("done", n, flush=True)
    cta(); print("done cta")
