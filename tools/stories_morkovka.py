"""Daily-update stories in a minimal Pinterest-style look.

Every slide is a full-screen shot (photo -> JPG, video -> MP4), gently muted, with small white
type placed in a free zone away from faces. Each slide gets its own small accent: "• • •" dots,
a thin vertical bar, a thin rule above the heading, a white note card, a soft highlighter,
a tinted heading or a serif body. A barely-there dark gradient, stretched far, keeps type legible.

Usage: python3 tools/stories_morkovka.py
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_editorial import H, ROOT, W, cover_crop, manrope  # noqa: E402

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # only needed for .HEIC sources
    pass

SRC = os.environ.get("STORIES_SRC", "/tmp/claude-0/-home-user-design-montazh-so-vkusom/"
                     "407108f0-3ce4-585b-bf12-4b6e65a8be4f/scratchpad/drive/Мои сторис")
OUT = os.path.join(ROOT, "stories_morkovka")
CLASSIC = os.path.join(ROOT, "fonts", "commercial-safe", "Inter", "InterDisplay-Bold.otf")
SERIF = os.path.join(ROOT, "fonts", "Forum-Regular.ttf")
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

NB = " "
WHITE = (255, 255, 255)
CREAM = (246, 234, 204)      # tinted heading, like the pale yellow "ВЧЕРА" reference
PALE_PINK = (250, 222, 229)
NOTE_TEXT = (70, 62, 58)
HIGHLIGHT = (240, 228, 170, 120)  # soft translucent marker

MUTE = (0.86, 0.95)  # saturation, brightness: calm, slightly muted photos

# A slide is a full-screen shot plus one or more text groups.
# group: anchor ("top"/"bottom", y), x, width, align, blocks [(style, text, options)]
SLIDES = [
    {
        "video": "IMG_9361.mov", "frame": 5,
        "groups": [{
            "anchor": ("top", 300), "x": 100, "width": 800, "align": "left",
            "blocks": [
                ("bold", f"ну что, мы наконец-то ввели морковку!{NB}🥕", {}),
                ("light", f"сначала у{NB}дочки была немного странная реакция на{NB}новый вкус, "
                          f"но в{NB}целом морковка нам понравилась", {"bar": True}),
            ],
        }],
    },
    {
        "video": "IMG_9362.mov", "frame": 9,
        "groups": [
            {
                "anchor": ("top", 330), "x": 100, "width": 820, "align": "left",
                "blocks": [
                    ("line", "", {}),
                    ("bold", f"завтра хочу попробовать дать индейку,", {"size": 46}),
                    ("light", f"а потом уже наконец-то начать миксовать разные вкусы: индейка, "
                              f"морковка, кабачок. мне кажется, так будет уже намного интереснее!",
                     {"size": 32}),
                ],
            },
            {
                "anchor": ("bottom", 1640), "x": 120, "width": 840, "align": "left",
                "blocks": [
                    ("note", f"а вот цветную капусту я{NB}пока решила вообще не давать. после нашей "
                             f"весёлой ночки что-то не{NB}хочется к{NB}ней возвращаться", {}),
                ],
            },
        ],
    },
    {
        "photo": "IMG_9485.HEIC",
        "groups": [{
            "anchor": ("top", 260), "x": 100, "width": 820, "align": "left",
            "blocks": [
                ("dots", "", {}),
                ("bold", f"если спросите, как у{NB}нас дела со{NB}сном, то{NB}через одно место!",
                 {"size": 46}),
                ("light", f"от укачиваний мы так и{NB}не смогли уйти. и{NB}я, если честно, вообще "
                          f"не понимаю, КАК от{NB}них уходят?!", {"size": 32}),
                ("hl", f"мамы, поделитесь опытом, потому что я{NB}уже не{NB}знаю, что делать", {}),
            ],
        }],
    },
    {
        "photo": "E7CA53F1-372D-41B8-AFE2-7AFFC580A2CE.png",
        "groups": [{
            "anchor": ("bottom", 1600), "x": 140, "width": 800, "align": "center",
            "blocks": [
                ("bold", f"а сегодня маму наконец-то отпускают на{NB}массаж!",
                 {"color": CREAM, "size": 50}),
                ("light", f"я начала ходить на{NB}расслабляющий массаж раз в{NB}неделю. сегодня иду "
                          f"всего второй раз, но уже могу сказать, что это лучший час в{NB}моей "
                          f"жизни{NB}😂", {"size": 32}),
            ],
        }],
    },
    {
        "photo": "0E168C5F-0AAE-4E41-96DE-CF882206C536.png",
        "groups": [{
            "anchor": ("bottom", 1620), "x": 100, "width": 820, "align": "left",
            "blocks": [
                ("light", f"после родов у{NB}меня до{NB}сих пор безумно напряжено всё тело, поэтому "
                          f"этот час, когда можно просто лежать, расслабиться и{NB}ни о{NB}чём "
                          f"не{NB}думать,", {"size": 34}),
                ("bold", f"для меня теперь отдельный вид удовольствия", {"color": PALE_PINK}),
            ],
        }],
    },
    {
        "photo": "0C3CDFFA-4B81-495C-96E2-31C2645E016B.png",
        "groups": [{
            "anchor": ("bottom", 1620), "x": 100, "width": 830, "align": "left",
            "blocks": [
                ("light", f"с домашними тренировками пока всё нестабильно. то занимаюсь, "
                          f"то пропускаю. иногда просто лень, а{NB}иногда настолько нет сил, что "
                          f"даже думать о{NB}тренировке не{NB}хочется.", {}),
                ("bold", f"в общем, пытаюсь потихоньку возвращать в{NB}свою жизнь что-то для себя. "
                         f"пока получается через раз, но{NB}хотя бы так!", {"size": 44}),
            ],
        }],
    },
]

FONTS = {  # style: (font file or maker, default size, line spacing factor)
    "bold": (lambda n: ImageFont.truetype(CLASSIC, n), 50, 1.24),
    "light": (lambda n: manrope(n, "Light"), 34, 1.42),
    "serif": (lambda n: ImageFont.truetype(SERIF, n), 40, 1.3),
    "note": (lambda n: manrope(n, "Regular"), 30, 1.45),
    "hl": (lambda n: ImageFont.truetype(CLASSIC, n), 36, 1.4),
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


def layout_group(d, group):
    """Wrap a group's blocks; return the drawable items and the group's height."""
    items, h = [], 0
    for k, (style, text, opt) in enumerate(group["blocks"]):
        if k:
            h += GAP
        if style == "dots":
            items.append((style, None, [], 0, opt))
            h += 22
            continue
        if style == "line":
            items.append((style, None, [], 0, opt))
            h += 6
            continue
        make, size, spacing = FONTS[style]
        rich = Rich(make(opt.get("size", size)))
        inset = 44 if opt.get("bar") else 64 if style == "note" else 0
        lines = rich.wrap(d, text, group["width"] - inset)
        step = round(rich.size * spacing)
        items.append((style, rich, lines, step, opt))
        h += len(lines) * step + (56 if style == "note" else 0)
    return items, h


def draw_group(layer, group):
    d = ImageDraw.Draw(layer)
    items, h = layout_group(d, group)
    side, edge = group["anchor"]
    y = edge if side == "top" else edge - h
    x0, width, center = group["x"], group["width"], group["align"] == "center"

    def line_x(rich, line, indent=0):
        return x0 + (width - rich.length(d, line)) / 2 if center else x0 + indent

    for k, (style, rich, lines, step, opt) in enumerate(items):
        if k:
            y += GAP
        if style == "dots":  # "• • •" like a typing bubble
            for i in range(3):
                cx = x0 + 8 + i * 30
                d.ellipse((cx - 7, y + 4, cx + 7, y + 18), fill=WHITE)
            y += 22
        elif style == "line":  # thin rule above the heading
            d.rectangle((x0, y, x0 + 150, y + 4), fill=WHITE)
            y += 6
        elif style == "note":  # white note card with small dark text
            text_w = max(rich.length(d, line) for line in lines)
            d.rounded_rectangle((x0, y, x0 + text_w + 64, y + len(lines) * step + 56), radius=26,
                                fill=(255, 255, 255, 240))
            ty = y + 26
            for line in lines:
                rich.draw(layer, d, (x0 + 32, ty), line, NOTE_TEXT)
                ty += step
            y += len(lines) * step + 56
        else:
            color = opt.get("color", WHITE)
            indent = 44 if opt.get("bar") else 0
            if opt.get("bar"):  # thin vertical bar beside the paragraph
                d.rectangle((x0, y + 6, x0 + 4, y + len(lines) * step - 6), fill=WHITE)
            for line in lines:
                lx = line_x(rich, line, indent)
                if style == "hl":  # soft marker behind each line only
                    _, t, _, b = d.textbbox((lx, y), line, font=rich.font)
                    d.rounded_rectangle((lx - 10, t - 8, lx + rich.length(d, line) + 10, b + 8),
                                        radius=8, fill=HIGHLIGHT)
                rich.draw(layer, d, (lx, y), line, color)
                y += step
    return (edge, edge + h) if side == "top" else (edge - h, edge)


def with_shadow(layer, strength=110):
    """Soft dark shadow under the type so white reads on light spots."""
    alpha = layer.getchannel("A").filter(ImageFilter.GaussianBlur(10)).point(lambda v: v * strength // 255)
    shadow = Image.new("RGBA", layer.size, (20, 14, 12, 0))
    shadow.putalpha(alpha)
    shadow.alpha_composite(layer)
    return shadow


def dark_veil(spans):
    """Barely-there darkening around each text group, stretched far so it never shows as a band."""
    a = np.zeros(H, np.float32)
    ys = np.arange(H)
    for top, bottom in spans:
        dist = np.maximum(np.maximum(top - ys, ys - bottom), 0)
        a = np.maximum(a, 0.4 * np.clip(1 - dist / 650, 0, 1) ** 1.6)
    veil = Image.new("RGBA", (W, H), (20, 14, 12, 0))
    veil.putalpha(Image.fromarray((np.repeat(a[:, None], W, 1) * 255).astype(np.uint8)))
    return veil


def overlay(slide):
    """Transparent layer: veil + all text groups."""
    text = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    spans = [draw_group(text, g) for g in slide["groups"]]
    layer = dark_veil(spans)
    layer.alpha_composite(with_shadow(text))
    return layer


def load(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def mute(im):
    im = ImageEnhance.Color(im).enhance(MUTE[0])
    return ImageEnhance.Brightness(im).enhance(MUTE[1])


def cover(im, focus=(0.5, 0.5)):
    tmp = os.path.join(OUT, "_cover.png")
    im.save(tmp)
    out = cover_crop(tmp, W, H, focus)
    os.remove(tmp)
    return out


def render_photo(slide, index):
    img = mute(cover(load(os.path.join(SRC, slide["photo"])))).convert("RGBA")
    img.alpha_composite(overlay(slide))
    path = os.path.join(OUT, f"story_{index:02d}.jpg")
    img.convert("RGB").save(path, quality=95)
    return path


def render_video(slide, index):
    src = os.path.join(SRC, slide["video"])
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", src], capture_output=True, text=True, check=True).stdout.strip()
    layer = overlay(slide)
    layer_p = os.path.join(OUT, "_layer.png")
    layer.save(layer_p)
    path = os.path.join(OUT, f"story_{index:02d}.mp4")
    brightness = MUTE[1] - 1  # eq brightness is additive, roughly matching the photo grade
    graph = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
             f"eq=saturation={MUTE[0]}:brightness={brightness / 2:.3f}[v];"
             f"[v][1:v]overlay=0:0:shortest=1,fps=30,format=yuv420p[out]")
    # The still input is looped, so it is capped at the clip's duration
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-loop", "1", "-t", dur, "-i", layer_p,
                    "-t", dur, "-filter_complex", graph, "-map", "[out]", "-map", "0:a?",
                    "-c:v", "libx264", "-crf", "19", "-preset", "veryfast", "-c:a", "aac",
                    "-b:a", "160k", "-movflags", "+faststart", path], check=True)
    os.remove(layer_p)
    # Still preview for the overview sheet
    frame_p = os.path.join(OUT, "_frame.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(slide["frame"]), "-i", path,
                    "-frames:v", "1", frame_p], check=True)
    load(frame_p).save(os.path.join(OUT, f"story_{index:02d}_preview.jpg"), quality=90)
    os.remove(frame_p)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i, slide in enumerate(SLIDES, 1):
        print(render_video(slide, i) if "video" in slide else render_photo(slide, i))
    previews = [os.path.join(OUT, f"story_{i:02d}" + ("_preview.jpg" if "video" in s else ".jpg"))
                for i, s in enumerate(SLIDES, 1)]
    ims = [Image.open(p).resize((360, 640)) for p in previews]
    sheet = Image.new("RGB", (380 * len(ims) + 20, 680), (255, 255, 255))
    for k, im in enumerate(ims):
        sheet.paste(im, (20 + k * 380, 20))
    sheet.save(os.path.join(OUT, "00_overview.jpg"), quality=90)
