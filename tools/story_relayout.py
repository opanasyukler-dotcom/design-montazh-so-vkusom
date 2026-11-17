"""Re-layout finished stories into the 'quote on black' story design:
three powder dots + big bold caps headline on black at the top, footage fading in below it,
bold lead line + regular line at the bottom on a dark gradient (Montserrat Bold / Regular).

The source stories have text baked into their lower part, so only the clean upper band of each
clip is kept (crop height per story) and placed under the new headline; the voice track is kept.

usage: python3 story_relayout.py SRC_DIR OUT_DIR
"""
import os, sys, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC, OUT = sys.argv[1:3]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONT = os.path.join(ROOT, "fonts/Montserrat-VariableFont_wght.ttf")
W, H = 1080, 1920
X = 78                       # left text margin, as in the reference
POWDER = (226, 210, 211)     # brand powder for the dots


def mont(size, w):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([w]); return f


def wrap(text, f, width):
    if "\n" in text: return text.split("\n")  # manual breaks: no orphan words
    out, line = [], ""
    for wd in text.split():
        t = (line + " " + wd).strip()
        if f.getlength(t) <= width or not line: line = t
        else: out.append(line); line = wd
    return out + [line]


# src rows y0..y1 are clean (above the baked subtitles); they are scaled by k, cropped to 1080 at x
# and placed at canvas y; fade = length of the dark gradient under the captions
STORIES = [
    dict(fn="02fdde07-02_drugoy_variant.mp4", y0=0, y1=1300, k=1.0, x=0, y=520, fade=560,
         head=["В ЕЁ СЛУЧАЕ", "РАССМАТРИВАЕМ", "ДРУГОЙ ВАРИАНТ"],
         lead="Анатомический имплант\nвысокой проекции", body="Задача: заполнить грудь и добавить объём"),
    dict(fn="e0c1055d-03_pochemu_bez_podtyazhki.mp4", y0=0, y1=1340, k=1.0, x=0, y=440, fade=560,
         head=["ПОЧЕМУ", "БЕЗ ПОДТЯЖКИ?"],
         lead="Потому что степень опущения небольшая,\nа собственного объёма груди мало",
         body="В такой ситуации имплант может помочь\nулучшить форму без дополнительных рубцов"),
    dict(fn="e7c850f9-05_zhdu_vas.mp4", y0=220, y1=1230, k=1.2, x=0, y=420, fade=260,
         head=["ЖДУ ВАС", "НА КОНСУЛЬТАЦИИ"],
         lead="Чтобы найти для вас лучшее решение", body="Напишите мне в Direct «КОНСУЛЬТАЦИЯ»"),
]


def shade(y, hc, fade):
    """black above the footage, soft fade into it, dark gradient at the bottom for the captions"""
    ys = np.arange(H, dtype=np.float32)
    top = 1 - np.clip((ys - y) / 300, 0, 1); top = top * top * (3 - 2 * top)
    b0, b1 = y + hc - fade, y + hc - 10
    bot = np.clip((ys - b0) / (b1 - b0), 0, 1); bot = bot * bot * (3 - 2 * bot)  # reaches full black: no seam
    a = np.maximum(top, bot)
    arr = np.zeros((H, W, 4), np.uint8); arr[..., 3] = (a * 255)[:, None]
    return Image.fromarray(arr, "RGBA")


def captions(head, lead, body):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for i in range(3):
        cx = X + 6 + i * 24; d.ellipse((cx - 6, 144, cx + 6, 156), fill=POWDER + (255,))
    fh = mont(76, 700); lh = 84
    for i, ln in enumerate(head): d.text((X, 210 + i * lh), ln, font=fh, fill=(255, 255, 255, 255))
    fb, fr = mont(40, 600), mont(40, 400)
    lead_l, body_l = wrap(lead, fb, W - 2 * X), wrap(body, fr, W - 2 * X)
    total = len(lead_l) * 50 + 22 + len(body_l) * 50
    y = 1790 - total                      # block sits above the IG reply bar
    for ln in lead_l: d.text((X, y), ln, font=fb, fill=(255, 255, 255, 255)); y += 50
    y += 22
    for ln in body_l: d.text((X, y), ln, font=fr, fill=(255, 255, 255, 235)); y += 50
    return im


def render(fn, y0, y1, k, x, y, fade, head, lead, body):
    hc = int((y1 - y0) * k) // 2 * 2; sw = int(W * k) // 2 * 2
    base = os.path.splitext(fn)[0].split("-", 1)[1]
    sh, tx = f"{OUT}/{base}_shade.png", f"{OUT}/{base}_text.png"
    shade(y, hc, fade).save(sh); captions(head, lead, body).save(tx)
    out = f"{OUT}/{base}.mp4"
    fc = (f"[0:v]crop={W}:{y1 - y0}:0:{y0},scale={sw}:{hc},crop={W}:{hc}:{x}:0,pad={W}:{H}:0:{y}:black[v];[v][1:v]overlay=shortest=1[s];"
          f"[2:v]format=rgba,fade=in:st=0.15:d=0.5:alpha=1[t];[s][t]overlay=shortest=1,format=yuv420p[o]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(SRC, fn), "-loop", "1", "-i", sh, "-loop", "1", "-i", tx,
                    "-filter_complex", fc, "-map", "[o]", "-map", "0:a?", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                    "-c:a", "copy", "-r", "30", "-movflags", "+faststart", out], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "2", "-i", out, "-frames:v", "1", f"{OUT}/{base}.jpg"], check=True)
    os.remove(sh); os.remove(tx)
    print("done", out, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s in STORIES: render(**s)
