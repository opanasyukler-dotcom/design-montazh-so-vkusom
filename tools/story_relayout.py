"""Stories 02 / 03 / 05 in the 'quote on black' story design, rebuilt from the clean iPhone sources:
three powder dots + big bold caps headline on black at the top, footage fading in below it and
running to the bottom edge, bold lead line + regular line over a dark gradient at the bottom
(Montserrat Bold / Regular).

Each story's segment was located in its source by cross-correlating the audio of the finished
story with the source (offsets below). The video comes from the clean source; the audio is the
finished story's track, so timing and sound stay exactly as approved.

usage: python3 story_relayout.py STORIES_DIR SOURCES_DIR OUT_DIR
"""
import os, sys, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

STORIES_DIR, SRC, OUT = sys.argv[1:4]
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


# story = finished story (audio + duration); src/ss = clean source and the segment start in it.
# The source (720x1280) is scaled to width 1080*k, rows from y0 (in 1080-wide units) are placed at
# canvas y, horizontally cropped at x. Everything below runs to the bottom edge of the story.
STORIES = [
    dict(story="02fdde07-02_drugoy_variant.mp4", src="IMG_2742.MP4", ss=8.495, k=1.0, y0=0, x=0, y=500,
         head=["В ЕЁ СЛУЧАЕ", "РАССМАТРИВАЕМ", "ДРУГОЙ ВАРИАНТ"],
         lead="Анатомический имплант\nвысокой проекции", body="Задача: заполнить грудь и добавить объём"),
    dict(story="e0c1055d-03_pochemu_bez_podtyazhki.mp4", src="IMG_2742.MP4", ss=27.655, k=1.0, y0=0, x=0, y=420,
         head=["ПОЧЕМУ", "БЕЗ ПОДТЯЖКИ?"],
         lead="Потому что степень опущения небольшая,\nа собственного объёма груди мало",
         body="В такой ситуации имплант может помочь\nулучшить форму без дополнительных рубцов"),
    dict(story="e7c850f9-05_zhdu_vas.mp4", src="IMG_6171.MP4", ss=13.095, k=1.15, y0=250, x=0, y=420,
         head=["ЖДУ ВАС", "НА КОНСУЛЬТАЦИИ"],
         lead="Чтобы найти для вас лучшее решение", body="Напишите мне в Direct «КОНСУЛЬТАЦИЯ»"),
]


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
    return im, 1790 - total


def shade(y, text_top):
    """black above the footage with a soft fade into it; dark gradient under the captions"""
    ys = np.arange(H, dtype=np.float32)
    top = 1 - np.clip((ys - y) / 300, 0, 1); top = top * top * (3 - 2 * top)
    b0, b1 = text_top - 380, text_top + 60
    bot = np.clip((ys - b0) / (b1 - b0), 0, 1); bot = bot * bot * (3 - 2 * bot) * 0.82
    a = np.maximum(top, bot)
    arr = np.zeros((H, W, 4), np.uint8); arr[..., 3] = (a * 255)[:, None]
    return Image.fromarray(arr, "RGBA")


def dur(f):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f]))


def render(story, src, ss, k, y0, x, y, head, lead, body):
    base = os.path.splitext(story)[0].split("-", 1)[1]
    sh, tx = f"{OUT}/{base}_shade.png", f"{OUT}/{base}_text.png"
    cap, text_top = captions(head, lead, body); cap.save(tx); shade(y, text_top).save(sh)
    st = os.path.join(STORIES_DIR, story); t = dur(st)
    sw = int(W * k) // 2 * 2; shh = int(H * k) // 2 * 2; vis = min(shh - int(y0 * k), H - y)
    out = f"{OUT}/{base}.mp4"
    fc = (f"[0:v]scale={sw}:{shh}:flags=lanczos,crop={W}:{vis}:{x}:{int(y0 * k)},pad={W}:{H}:0:{y}:black,setsar=1[v];"
          f"[v][2:v]overlay=shortest=1[s];"
          f"[3:v]format=rgba,fade=in:st=0.15:d=0.5:alpha=1[t];[s][t]overlay=shortest=1,format=yuv420p[o]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{ss}", "-t", f"{t}", "-i", os.path.join(SRC, src), "-i", st,
                    "-loop", "1", "-i", sh, "-loop", "1", "-i", tx,
                    "-filter_complex", fc, "-map", "[o]", "-map", "1:a", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
                    "-c:a", "copy", "-r", "30", "-t", f"{t}", "-movflags", "+faststart", out], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "2", "-i", out, "-frames:v", "1", f"{OUT}/{base}.jpg"], check=True)
    os.remove(sh); os.remove(tx)
    print("done", out, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s in STORIES: render(**s)
