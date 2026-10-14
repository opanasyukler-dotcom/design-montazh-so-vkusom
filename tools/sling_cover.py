"""«Обзор на слинг» — Reels cover in the video's look: full source frame (no dark haze over the photo), a long black spray
rising from the very bottom, Manrope type set over it (Light kicker, ExtraBold title, hairline, BLUSH «100/10»),
all kept inside the 3:4 profile-grid crop. Renders 2160x3840 PNG + 1080x1920 JPG.
  python3 -I sling_cover.py still.png fonts_dir out_basename [zoom] [cx] [cy]
"""
import sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

K = 2
W, H = 1080 * K, 1920 * K
WHITE, BLUSH = (255, 255, 255), (247, 223, 221)

def main(still, fonts, out, zoom=1.0, cx=0.5, cy=0.5):
    im = cv2.cvtColor(cv2.imread(still), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LANCZOS4)
    if zoom > 1:
        c = (cx * W, cy * H); M = np.float32([[zoom, 0, c[0] * (1 - zoom)], [0, zoom, c[1] * (1 - zoom)]])
        im = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    a = im.astype(np.float32) / 255
    a = np.clip((a - 0.5) * 1.04 + 0.5, 0, 1)                                    # same light grade as the video
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    spray = np.clip((y - 0.50) / 0.50, 0, 1) ** 1.6 * 0.80                        # black spray, densest at the very bottom
    a = a * (1 - spray)
    img = Image.fromarray((a * 255 + 0.5).astype(np.uint8))

    def F(name, size): return ImageFont.truetype(f"{fonts}/Manrope-{name}.ttf", int(size * K))
    def tracked(d, xy, txt, font, fill, track):
        x, yy = xy
        for ch in txt: d.text((x, yy), ch, font=font, fill=fill); x += font.getlength(ch) + track * font.size
        return x
    # soft drop shadow layer only under the glyphs (no haze box)
    txt = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(txt)
    X = 84 * K
    tracked(d, (X, 1300 * K), "обзор на", F("Light", 52), WHITE, 0.14)
    d.text((X - 8 * K, 1340 * K), "слинг", font=F("ExtraBold", 210), fill=WHITE)
    d.rectangle((X, 1600 * K, X + 380 * K, 1600 * K + int(1.8 * K)), fill=(255, 255, 255, 220))
    f = F("Regular", 42); x = X + d.textlength("лучшая покупка для мамы  ", font=f)
    d.text((X, 1622 * K), "лучшая покупка для мамы", font=f, fill=WHITE)
    fb = F("ExtraBold", 50); d.text((x, 1614 * K), "100", font=fb, fill=BLUSH)
    d.text((x + d.textlength("100", font=fb) + 6 * K, 1630 * K), "/10", font=F("Light", 34), fill=BLUSH)
    al = np.asarray(txt.split()[3], np.float32) / 255
    sh = cv2.GaussianBlur(al, (0, 0), 7 * K) * 0.35
    sh = np.roll(sh, 3 * K, axis=0)
    base = np.asarray(img, np.float32) * (1 - sh[..., None])
    rgb = np.asarray(txt.convert("RGB"), np.float32)
    base = base * (1 - al[..., None]) + rgb * al[..., None]
    out_im = Image.fromarray(np.clip(base + 0.5, 0, 255).astype(np.uint8))
    out_im.save(out + "_4k.png")
    out_im.resize((1080, 1920), Image.LANCZOS).save(out + ".jpg", quality=95, subsampling=0)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(float(v) for v in sys.argv[4:7]))
