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
        "layout": "full", "text": "top",  # face sits low, wall above it is free
        "blocks": [
            ("bold", f"Ну что, мы наконец-то ввели морковку!{NB}🥕"),
            ("light", f"Сначала у{NB}дочки была немного странная реакция на{NB}новый вкус, "
                      f"но в{NB}целом морковка нам понравилась{NB}😂"),
        ],
    },
    {
        "video": "IMG_9362.mov", "frame": 9,
        "layout": "card", "text": "bottom",  # small card up top, text below it
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
        "layout": "full", "text": "bottom",  # face mid-left, blanket below is free
        "blocks": [
            ("bold", f"Если спросите, как у{NB}нас дела со{NB}сном, то через одно место!"),
            ("light", f"От укачиваний мы так и{NB}не смогли уйти. И{NB}я, если честно, вообще "
                      f"не понимаю, КАК от{NB}них уходят?!"),
            ("mark", f"Мамы, поделитесь опытом, потому что я{NB}уже не знаю, что делать."),
        ],
    },
    {
        "photo": "E7CA53F1-372D-41B8-AFE2-7AFFC580A2CE.png",
        "layout": "full", "text": "bottom",  # face on top, text over the pink tee
        "blocks": [
            ("bold", f"А сегодня маму наконец-то отпускают на{NB}массаж!"),
            ("light", f"Я начала ходить на{NB}расслабляющий массаж раз в{NB}неделю. Сегодня иду "
                      f"всего второй раз, но уже могу сказать, что это лучший час в{NB}моей "
                      f"жизни{NB}😂"),
        ],
    },
    {
        "photo": "0E168C5F-0AAE-4E41-96DE-CF882206C536.png",
        "layout": "card", "text": "top", "tilt": -3,  # close-up face: tilted print below text
        "blocks": [
            ("light", f"После родов у{NB}меня до{NB}сих пор безумно напряжено всё тело, поэтому "
                      f"этот час, когда можно просто лежать, расслабиться и{NB}ни о{NB}чём "
                      f"не{NB}думать,"),
            ("bold", f"для меня теперь отдельный вид удовольствия."),
        ],
    },
    {
        "photo": "0C3CDFFA-4B81-495C-96E2-31C2645E016B.png",
        "layout": "split", "focus": (0.5, 0.15),  # photo edge to edge on top, paper panel below
        "blocks": [
            ("light", f"С домашними тренировками пока всё нестабильно. То занимаюсь, то пропускаю. "
                      f"Иногда просто лень, а{NB}иногда настолько нет сил, что даже думать "
                      f"о{NB}тренировке не{NB}хочется."),
            ("bold", f"В общем, пытаюсь потихоньку возвращать в{NB}свою жизнь что-то для себя. "
                     f"Пока получается через раз, но хотя бы так!"),
        ],
    },
]

MARGIN, SAFE_TOP, SAFE_BOTTOM = 100, 250, 1640  # clear of Instagram's top bar and reply bar
STYLES = {  # font, colour, line step
    "bold": (lambda: ImageFont.truetype(CLASSIC, 56), BROWN, 70),
    "mark": (lambda: ImageFont.truetype(CLASSIC, 48), BROWN, 64),
    "light": (lambda: manrope(40, "Light"), BODY, 58),
}
GAP = 26


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
        img.alpha_composite(glyph, (round(x + self.size * 0.05), round(y + asc - s * 0.12)))
        return self.size * 1.15


def text_blocks(d, blocks, width):
    """Wrap every block; return [(style, lines)] and total height."""
    out, h = [], 0
    for k, (style, text) in enumerate(blocks):
        _, _, step = STYLES[style]
        rich = Rich(STYLES[style][0]())
        lines = rich.wrap(d, text, width - (60 if style == "mark" else 0))
        out.append((style, rich, lines))
        h += (GAP if k else 0) + len(lines) * step + (30 if style == "mark" else 0)
    return out, h


def draw_blocks(img, d, wrapped, x, y, width):
    for k, (style, rich, lines) in enumerate(wrapped):
        _, color, step = STYLES[style]
        if k:
            y += GAP
        if style == "mark":  # paper marker plate, like the highlights in the hospital stories
            h = len(lines) * step + 30
            d.rounded_rectangle((x - 4, y - 8, x + width + 4, y + h - 8), radius=18,
                                fill=(250, 247, 242, 255))
            y += 12
            for line in lines:
                rich.draw(img, d, (x + 26, y), line, color)
                y += step
            y += 18
        else:
            for line in lines:
                rich.draw(img, d, (x, y), line, color)
                y += step
    return y


def load(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def video_frame(path, t):
    tmp = os.path.join(OUT, "_frame.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", path, "-frames:v", "1", tmp],
                   check=True)
    im = load(tmp)
    os.remove(tmp)
    return im


def cover(im, w, h, focus=(0.5, 0.5)):
    tmp = os.path.join(OUT, "_cover.png")
    im.save(tmp)
    out = cover_crop(tmp, w, h, focus)
    os.remove(tmp)
    return out


def blurred_background(im):
    """Blurred cover of the same shot, washed toward warm paper so brown text reads."""
    bg = cover(im, W, H).filter(ImageFilter.GaussianBlur(40))
    return Image.blend(bg, Image.new("RGB", (W, H), PAPER), 0.55)


def rounded_mask(w, h, r=34):
    m = Image.new("L", (w * 4, h * 4), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * 4 - 1, h * 4 - 1), radius=r * 4, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def counter(d, index, total, fill=HEADER):
    hf = manrope(26, "Regular")
    label = f"{index}/{total}"
    d.text((W - MARGIN - d.textlength(label, font=hf), 170), label, font=hf, fill=fill)


def wash(start, end, top_side):
    """Paper-coloured RGBA gradient: solid on the text side, fading out toward the photo."""
    a = np.zeros((H, 1), np.float32)
    span = end - start
    ramp = (np.linspace(1, 0, span) ** 1.3) * 0.86
    if top_side:
        a[:start, 0] = 0.86
        a[start:end, 0] = ramp
    else:
        a[end:, 0] = 0.86
        a[start:end, 0] = ramp[::-1]
    layer = Image.new("RGBA", (W, H), PAPER + (0,))
    layer.putalpha(Image.fromarray((np.repeat(a, W, 1) * 255).astype(np.uint8)))
    return layer


def full_overlay(slide, index, total):
    """Transparent layer for full-screen slides: paper wash + text on the free side."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    width = W - 2 * MARGIN
    wrapped, h = text_blocks(d, slide["blocks"], width)
    if slide["text"] == "top":
        y = SAFE_TOP
        layer.alpha_composite(wash(y + h - 60, y + h + 170, True))
    else:
        y = SAFE_BOTTOM - h
        layer.alpha_composite(wash(y - 280, y + 60, False))
    d = ImageDraw.Draw(layer)
    counter(d, index, total)
    draw_blocks(layer, d, wrapped, MARGIN, y, width)
    return layer


def card_base(slide, index, total, first, src_size):
    """Blurred background, text on one side and a framed card on the other ("distant cover")."""
    img = blurred_background(first).convert("RGBA")
    d = ImageDraw.Draw(img)
    counter(d, index, total)
    width = W - 2 * MARGIN
    wrapped, h = text_blocks(d, slide["blocks"], width)
    if slide["text"] == "top":
        draw_blocks(img, d, wrapped, MARGIN, SAFE_TOP, width)
        area = (SAFE_TOP + h + 70, SAFE_BOTTOM)
    else:
        draw_blocks(img, d, wrapped, MARGIN, SAFE_BOTTOM - h, width)
        area = (SAFE_TOP - 20, SAFE_BOTTOM - h - 70)

    max_w, max_h = W - 2 * 110, area[1] - area[0]
    sw, sh = src_size
    scale = min(max_w / sw, max_h / sh)
    cw, ch = int(sw * scale) // 2 * 2, int(sh * scale) // 2 * 2
    cx, cy = (W - cw) // 2, area[0] + (max_h - ch) // 2
    return img, (cx, cy, cw, ch)


def framed_card(photo, tilt=0, border=12):
    """Photo with rounded corners in a white frame, optionally tilted; returns RGBA."""
    cw, ch = photo.size
    card = Image.new("RGBA", (cw + 2 * border, ch + 2 * border), (0, 0, 0, 0))
    card.paste((252, 250, 247, 255), (0, 0), rounded_mask(*card.size, 34 + border))
    card.paste(photo, (border, border), rounded_mask(cw, ch))
    return card.rotate(tilt, Image.BICUBIC, expand=True) if tilt else card


def drop_shadow(img, card, xy):
    shadow = Image.new("L", img.size, 0)
    shadow.paste(card.getchannel("A").point(lambda v: v * 90 // 255), (xy[0] + 4, xy[1] + 16))
    img.alpha_composite(Image.merge("RGBA", (*Image.new("RGB", img.size, (95, 75, 65)).split(),
                                             shadow.filter(ImageFilter.GaussianBlur(26)))))


def render_photo(slide, index, total):
    im = load(os.path.join(SRC, slide["photo"]))
    if slide["layout"] == "full":
        img = cover(im, W, H, slide.get("focus", (0.5, 0.5))).convert("RGBA")
        img.alpha_composite(full_overlay(slide, index, total))
    elif slide["layout"] == "split":
        img = Image.new("RGBA", (W, H), PAPER + (255,))
        d = ImageDraw.Draw(img)
        width = W - 2 * MARGIN
        wrapped, h = text_blocks(d, slide["blocks"], width)
        panel = SAFE_BOTTOM - h - 90  # photo edge to edge above, paper panel with text below
        img.paste(cover(im, W, panel, slide.get("focus", (0.5, 0.5))), (0, 0))
        d.line((MARGIN, panel + 45, MARGIN + 120, panel + 45), fill=BROWN, width=2)
        draw_blocks(img, d, wrapped, MARGIN, panel + 90, width)
        counter(d, index, total, (250, 247, 242))
    else:
        img, (cx, cy, cw, ch) = card_base(slide, index, total, im, im.size)
        tilt = slide.get("tilt", 0)
        if tilt:  # leave room for the rotated corners
            cw, ch = int(cw * 0.92) // 2 * 2, int(ch * 0.92) // 2 * 2
            cx, cy = (W - cw) // 2, cy + 20
        card = framed_card(im.resize((cw, ch), Image.LANCZOS), tilt)
        xy = (cx + cw // 2 - card.width // 2, cy + ch // 2 - card.height // 2)
        drop_shadow(img, card, xy)
        img.alpha_composite(card, xy)
    path = os.path.join(OUT, f"story_{index:02d}.jpg")
    img.convert("RGB").save(path, quality=95)
    return path


def ffmpeg(args, path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-map", "[out]", "-map", "0:a?",
                    "-c:v", "libx264", "-crf", "19", "-preset", "veryfast", "-c:a", "aac",
                    "-b:a", "160k", "-movflags", "+faststart", path], check=True)


def render_video(slide, index, total):
    src = os.path.join(SRC, slide["video"])
    first = video_frame(src, slide["frame"])
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", src], capture_output=True, text=True, check=True).stdout.strip()
    path = os.path.join(OUT, f"story_{index:02d}.mp4")
    layer_p, mask_p = os.path.join(OUT, "_layer.png"), os.path.join(OUT, "_mask.png")
    # Still inputs are looped, so every stream is capped at the clip's duration
    if slide["layout"] == "full":
        layer = full_overlay(slide, index, total)
        layer.save(layer_p)
        graph = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}[v];"
                 f"[v][1:v]overlay=0:0:shortest=1,fps=30,format=yuv420p[out]")
        ffmpeg(["-i", src, "-loop", "1", "-t", dur, "-i", layer_p, "-t", dur,
                "-filter_complex", graph], path)
        still = cover(first, W, H).convert("RGBA")
        still.alpha_composite(layer)
    else:
        base, (cx, cy, cw, ch) = card_base(slide, index, total, first, first.size)
        frame = framed_card(Image.new("RGB", (cw, ch)))  # white frame under the live video
        drop_shadow(base, frame, (cx - 12, cy - 12))
        base.alpha_composite(frame, (cx - 12, cy - 12))
        base.convert("RGB").save(layer_p)
        rounded_mask(cw, ch).save(mask_p)
        graph = (f"[0:v]scale={cw}:{ch},format=rgba[v];[2:v]format=gray[m];[v][m]alphamerge[vm];"
                 f"[1:v][vm]overlay={cx}:{cy}:shortest=1,fps=30,format=yuv420p[out]")
        ffmpeg(["-i", src, "-loop", "1", "-t", dur, "-i", layer_p, "-loop", "1", "-t", dur,
                "-i", mask_p, "-t", dur, "-filter_complex", graph], path)
        still = base.copy()
        still.paste(first.resize((cw, ch), Image.LANCZOS), (cx, cy), rounded_mask(cw, ch))
        os.remove(mask_p)
    still.convert("RGB").save(os.path.join(OUT, f"story_{index:02d}_preview.jpg"), quality=90)
    os.remove(layer_p)
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
