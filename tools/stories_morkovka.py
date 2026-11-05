"""Daily-update stories: blurred copy of the shot as background, sharp shot on top as a card.

Photo slides render to JPG, video slides to MP4 (static blurred frame + text, live video card).
Text: bold lines in Inter Display Bold (Instagram "Classic" look), body in Manrope Light,
emoji from Noto Color Emoji.

Usage: python3 tools/stories_morkovka.py
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_editorial import BODY, BROWN, H, HEADER, PAPER, ROOT, W, cover_crop, manrope  # noqa: E402

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # only needed for .HEIC sources
    pass

SRC = os.environ.get("STORIES_SRC", "/tmp/claude-0/-home-user-design-montazh-so-vkusom/"
                     "407108f0-3ce4-585b-bf12-4b6e65a8be4f/scratchpad/drive/Мои сторис")
OUT = os.path.join(ROOT, "stories_morkovka")
CLASSIC = os.path.join(ROOT, "fonts", "commercial-safe", "Inter", "InterDisplay-Bold.otf")
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

NB = " "
SLIDES = [
    {
        "video": "IMG_9361.mov", "frame": 5,
        "blocks": [
            ("bold", f"Ну что, мы наконец-то ввели морковку!{NB}🥕"),
            ("light", f"Сначала у{NB}дочки была немного странная реакция на{NB}новый вкус, "
                      f"но в{NB}целом морковка нам понравилась{NB}😂"),
        ],
    },
    {
        "video": "IMG_9362.mov", "frame": 9,
        "blocks": [
            ("light", f"Завтра хочу попробовать дать индейку, а{NB}потом уже наконец-то начать "
                      f"миксовать разные вкусы: индейка, морковка, кабачок. Мне кажется, так "
                      f"будет уже намного интереснее!"),
            ("bold", f"А вот цветную капусту я{NB}пока решила вообще не давать."),
            ("light", f"После нашей весёлой ночки что-то не{NB}хочется к{NB}ней возвращаться{NB}😂"),
        ],
    },
    {
        "photo": "IMG_9485.HEIC",
        "blocks": [
            ("bold", f"Если спросите, как у{NB}нас дела со{NB}сном, то через одно место!"),
            ("light", f"От укачиваний мы так и{NB}не смогли уйти. И{NB}я, если честно, вообще "
                      f"не понимаю, КАК от{NB}них уходят?!"),
            ("mark", f"Мамы, поделитесь опытом, потому что я{NB}уже не знаю, что делать."),
        ],
    },
    {
        "photo": "E7CA53F1-372D-41B8-AFE2-7AFFC580A2CE.png",
        "blocks": [
            ("bold", f"А сегодня маму наконец-то отпускают на{NB}массаж!"),
            ("light", f"Я начала ходить на{NB}расслабляющий массаж раз в{NB}неделю. Сегодня иду "
                      f"всего второй раз, но уже могу сказать, что это лучший час в{NB}моей "
                      f"жизни{NB}😂"),
        ],
    },
    {
        "photo": "0E168C5F-0AAE-4E41-96DE-CF882206C536.png",
        "blocks": [
            ("light", f"После родов у{NB}меня до{NB}сих пор безумно напряжено всё тело, поэтому "
                      f"этот час, когда можно просто лежать, расслабиться и{NB}ни о{NB}чём "
                      f"не{NB}думать,"),
            ("bold", f"для меня теперь отдельный вид удовольствия."),
        ],
    },
    {
        "photo": "0C3CDFFA-4B81-495C-96E2-31C2645E016B.png",
        "blocks": [
            ("light", f"С домашними тренировками пока всё нестабильно. То занимаюсь, то пропускаю. "
                      f"Иногда просто лень, а{NB}иногда настолько нет сил, что даже думать "
                      f"о{NB}тренировке не{NB}хочется."),
            ("bold", f"В общем, пытаюсь потихоньку возвращать в{NB}свою жизнь что-то для себя. "
                     f"Пока получается через раз, но хотя бы так!"),
        ],
    },
]

MARGIN, TOP, CARD_BOTTOM = 100, 250, 1640
STYLES = {  # font, colour, line step
    "bold": (lambda: ImageFont.truetype(CLASSIC, 56), BROWN, 70),
    "mark": (lambda: ImageFont.truetype(CLASSIC, 48), BROWN, 64),
    "light": (lambda: manrope(40, "Light"), BODY, 58),
}


def is_emoji(ch):
    return ord(ch) >= 0x2600 and ch != NB


class Rich:
    """Text with inline colour emoji: measuring, wrapping and drawing."""

    def __init__(self, font):
        self.font = font
        self.size = font.size
        self.emoji_font = ImageFont.truetype(EMOJI, 109)

    def length(self, d, text):
        n = sum(is_emoji(c) for c in text)
        plain = "".join(c for c in text if not is_emoji(c))
        return d.textlength(plain, font=self.font) + n * self.size * 1.15

    def wrap(self, d, text, width):
        lines, line = [], ""
        for word in text.split(" "):
            trial = f"{line} {word}".strip()
            if self.length(d, trial) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
        return lines

    def draw(self, img, d, xy, text, fill):
        x, y = xy
        run = ""
        for ch in text + "\0":
            if ch == "\0" or is_emoji(ch):
                if run:
                    d.text((x, y), run, font=self.font, fill=fill)
                    x += d.textlength(run, font=self.font)
                    run = ""
                if ch != "\0":
                    x += self.paste_emoji(img, ch, x, y)
            else:
                run += ch

    def paste_emoji(self, img, ch, x, y):
        glyph = Image.new("RGBA", (136, 128), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((0, 0), ch, font=self.emoji_font, embedded_color=True)
        glyph = glyph.crop(glyph.getbbox() or (0, 0, 1, 1))
        s = round(self.size * 1.05)
        glyph = glyph.resize((s, round(s * glyph.height / glyph.width)), Image.LANCZOS)
        asc = self.font.getbbox("Н")[1]
        img.paste(glyph, (round(x + self.size * 0.05), round(y + asc - s * 0.12)), glyph)
        return self.size * 1.15


def load(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def video_frame(path, t):
    tmp = os.path.join(OUT, "_frame.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", path, "-frames:v", "1", tmp],
                   check=True)
    im = load(tmp)
    os.remove(tmp)
    return im


def background(im):
    """Blurred cover of the same shot, washed toward warm paper so brown text reads."""
    tmp = os.path.join(OUT, "_bg.png")
    im.save(tmp)
    bg = cover_crop(tmp, W, H, (0.5, 0.5)).filter(ImageFilter.GaussianBlur(40))
    os.remove(tmp)
    return Image.blend(bg, Image.new("RGB", (W, H), PAPER), 0.55)


def rounded_mask(w, h, r=34):
    m = Image.new("L", (w * 4, h * 4), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * 4 - 1, h * 4 - 1), radius=r * 4, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def layout(slide, index, total, src_size):
    """Draw text on the background; return the image and the card rectangle."""
    img = background(slide["_first"])
    d = ImageDraw.Draw(img)
    hf = manrope(26, "Regular")
    counter = f"{index}/{total}"
    d.text((W - MARGIN - d.textlength(counter, font=hf), 170), counter, font=hf, fill=HEADER)

    y = TOP
    width = W - 2 * MARGIN
    for k, (style, text) in enumerate(slide["blocks"]):
        make_font, color, step = STYLES[style]
        rich = Rich(make_font())
        if k:
            y += 26
        lines = rich.wrap(d, text, width - (60 if style == "mark" else 0))
        if style == "mark":  # paper marker plate, like the highlights in the hospital stories
            h = len(lines) * step + 30
            d.rounded_rectangle((MARGIN - 4, y - 8, W - MARGIN + 4, y + h - 8), radius=18,
                                fill=(250, 247, 242))
            y += 12
            for line in lines:
                rich.draw(img, d, (MARGIN + 26, y), line, color)
                y += step
            y += 18
        else:
            for line in lines:
                rich.draw(img, d, (MARGIN, y), line, color)
                y += step

    # Card: as large as fits between the text and the reply-bar zone
    top = y + 60
    max_w, max_h = W - 2 * 90, CARD_BOTTOM - top
    sw, sh = src_size
    scale = min(max_w / sw, max_h / sh)
    cw, ch = int(sw * scale) // 2 * 2, int(sh * scale) // 2 * 2
    cx, cy = (W - cw) // 2, top + (max_h - ch) // 2

    border = 12
    shadow = Image.new("L", img.size, 0)
    ImageDraw.Draw(shadow).rounded_rectangle(
        (cx - border + 4, cy - border + 16, cx + cw + border + 4, cy + ch + border + 16), radius=44, fill=90)
    img.paste(Image.new("RGB", img.size, (95, 75, 65)), (0, 0), shadow.filter(ImageFilter.GaussianBlur(26)))
    frame = rounded_mask(cw + 2 * border, ch + 2 * border, 34 + border)
    img.paste(Image.new("RGB", frame.size, (252, 250, 247)), (cx - border, cy - border), frame)
    return img, (cx, cy, cw, ch)


def render_photo(slide, index, total):
    im = load(os.path.join(SRC, slide["photo"]))
    slide["_first"] = im
    img, (cx, cy, cw, ch) = layout(slide, index, total, im.size)
    img.paste(im.resize((cw, ch), Image.LANCZOS), (cx, cy), rounded_mask(cw, ch))
    path = os.path.join(OUT, f"story_{index:02d}.jpg")
    img.save(path, quality=95)
    return path


def render_video(slide, index, total):
    src = os.path.join(SRC, slide["video"])
    first = video_frame(src, slide["frame"])
    slide["_first"] = first
    base, (cx, cy, cw, ch) = layout(slide, index, total, first.size)
    base_p, mask_p = os.path.join(OUT, "_base.png"), os.path.join(OUT, "_mask.png")
    base.save(base_p)
    rounded_mask(cw, ch).save(mask_p)
    path = os.path.join(OUT, f"story_{index:02d}.mp4")
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", src], capture_output=True, text=True, check=True).stdout.strip()
    graph = (f"[0:v]scale={cw}:{ch},format=rgba[v];[2:v]format=gray[m];[v][m]alphamerge[vm];"
             f"[1:v][vm]overlay={cx}:{cy}:shortest=1,fps=30,format=yuv420p[out]")
    # Still inputs are looped, so every stream is capped at the clip's duration
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-loop", "1", "-t", dur, "-i", base_p,
                    "-loop", "1", "-t", dur, "-i", mask_p, "-t", dur, "-filter_complex", graph, "-map", "[out]",
                    "-map", "0:a?", "-c:v", "libx264", "-crf", "19", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", path], check=True)
    # Still preview for the overview sheet
    still = base.copy()
    still.paste(first.resize((cw, ch), Image.LANCZOS), (cx, cy), rounded_mask(cw, ch))
    still.save(os.path.join(OUT, f"story_{index:02d}_preview.jpg"), quality=90)
    os.remove(base_p)
    os.remove(mask_p)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    total = len(SLIDES)
    for i, slide in enumerate(SLIDES, 1):
        print(render_video(slide, i, total) if "video" in slide else render_photo(slide, i, total))
    previews = [os.path.join(OUT, f"story_{i:02d}" + ("_preview.jpg" if "video" in s else ".jpg"))
                for i, s in enumerate(SLIDES, 1)]
    ims = [Image.open(p).resize((360, 640)) for p in previews]
    sheet = Image.new("RGB", (380 * len(ims) + 20, 680), (255, 255, 255))
    for k, im in enumerate(ims):
        sheet.paste(im, (20 + k * 380, 20))
    sheet.save(os.path.join(OUT, "00_overview.jpg"), quality=90)
