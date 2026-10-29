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
    ("03_chose_lipofilling", 10.67, 19.20, "text3"),  # footage cropped low (patient in frame), headline on the wall top-left
    ("04_donor_zones", 19.20, 23.07),
    # headline band (top of the reel) stacked over the footage band (bottom of the reel): full 4:5, no side bars
    ("05_volume", 23.07, 29.20, "stack", (0.07, 0.36), (0.585, 1.0)),
    ("06_not_a_lift", 29.20, 36.60, "stack", (0.06, 0.335), (0.378, 1.0)),
    ("07_this_task", 36.60, None, "text"),  # footage cropped low, headline redrawn on top of it
]


G = "0x%02x%02x%02x" % ct.GRAPHITE


def stack(name, a, b, top, bot):
    """top band at native scale + bottom band filling the rest of the 4:5 frame, soft seam between them"""
    import numpy as np, cv2
    SH = 1920
    dur = (b or float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", REEL],
                                     capture_output=True, text=True).stdout)) - a
    t0, t1 = int(top[0] * SH), int(top[1] * SH); b0, b1 = int(bot[0] * SH), int(bot[1] * SH)
    ha = t1 - t0; hb = H - ha; seam = 60
    rd = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", REEL,
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    tmp = f"{OUT}/_{name}_v.mp4"
    wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30",
                           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", tmp],
                          stdin=subprocess.PIPE)
    k = np.linspace(0, 1, seam, dtype=np.float32)[:, None, None]; k = k * k * (3 - 2 * k)
    while True:
        buf = rd.stdout.read(W * SH * 3)
        if len(buf) < W * SH * 3: break
        f = np.frombuffer(buf, np.uint8).reshape(SH, W, 3)
        A = f[t0:t1].astype(np.float32)
        Bsrc = f[b0:b1]
        if b1 - b0 <= hb:  # native scale; the seam rows come from just above the band, so nothing repeats
            ext = f[max(b1 - hb - seam, 0):b1].astype(np.float32)
            B = ext[-hb:]; Bseam = ext[:seam] if len(ext) >= hb + seam else B[:seam] * 0 + A[-1:]
        else:  # taller than the room left: scale the footage down over its own (already blurred) background
            s = hb / (b1 - b0); fw = int(W * s) // 2 * 2
            fg = cv2.resize(Bsrc, (fw, hb), interpolation=cv2.INTER_AREA).astype(np.float32)
            bg = cv2.GaussianBlur(cv2.resize(Bsrc, (W, hb), interpolation=cv2.INTER_AREA), (0, 0), 25).astype(np.float32)
            m = np.ones((hb, fw, 1), np.float32); e = 70
            r = np.linspace(0, 1, e, dtype=np.float32)[None, :, None]; m[:, :e] *= r; m[:, -e:] *= r[:, ::-1]
            x = (W - fw) // 2; B = bg.copy(); B[:, x:x + fw] = bg[:, x:x + fw] * (1 - m) + fg * m
            Bseam = np.repeat(bg[:1], seam, 0)  # plain blurred background under the seam
        out = np.empty((H, W, 3), np.float32)
        out[:ha] = A; out[ha:] = B
        out[ha - seam:ha] = A[ha - seam:] * (1 - k) + Bseam * k  # soft seam between the two bands
        wr.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
    wr.stdin.close(); wr.wait(); rd.wait()
    fo = max(dur - 0.12, 0)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", REEL,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", f"afade=t=in:d=0.06,afade=t=out:st={fo:.3f}:d=0.12",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", f"{OUT}/{name}.mp4"], check=True)
    os.remove(tmp)


def headline_overlay(path):
    """the reel's last headline, redrawn over the footage (check mark, title, navy pill)"""
    from PIL import Image, ImageDraw
    S = 3
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov.alpha_composite(ct.vgrad(170, 380, 1.0, 0))  # the top of the crop is already dark: hides the reel's word pill
    hi = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0)); d = ImageDraw.Draw(hi)
    cx, cy, r = W // 2 * S, 92 * S, 40 * S
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=ct.WHITE + (255,), width=5 * S)
    d.line([(cx - 18 * S, cy + 1 * S), (cx - 5 * S, cy + 14 * S), (cx + 20 * S, cy - 13 * S)], fill=ct.WHITE + (255,), width=6 * S, joint="curve")
    ov.alpha_composite(hi.resize((W, H), Image.LANCZOS))
    ct.Block(W - 2 * ct.MARGIN, "center").head("Именно эту задачу\nи решали", 92, 0).draw(ov, ct.MARGIN, 152)
    f = ct.mont(27, 500); t = "Подробнее в актуальном «Липофилинг»"; tw = f.getlength(t)
    d = ImageDraw.Draw(ov); y = 352
    d.rounded_rectangle(((W - tw) / 2 - 22, y, (W + tw) / 2 + 22, y + 50), radius=25, fill=(16, 46, 70, 235))
    d.text((W / 2, y + 25), t, font=f, fill=ct.WHITE, anchor="mm")
    ov.save(path)


def chose_overlay(path):
    """ВЫБРАЛИ / ЛИПОФИЛИНГ on the wall, top-left (surgeon stays clear): no shade, only the text's own soft shadow"""
    from PIL import Image, ImageDraw
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    x, y = 44, 70
    fb = ct.bebas(100)
    ct.shadow_text(ov, (x + 14, y), "ВЫБРАЛИ", fb, ct.WHITE, blur=8, a=170); y += 100
    tw = fb.getlength("ЛИПОФИЛИНГ")
    d.rectangle((x, y + 6, x + tw + 28, y + 104), fill=(16, 46, 70, 255))
    d.text((x + 14, y), "ЛИПОФИЛИНГ", font=fb, fill=ct.WHITE); y += 124
    fs = ct.mont(30, 500)
    for ln in ["собственная жировая ткань", "будет получена из области", "галифе, внутренних бёдер", "и коленей"]:
        ct.shadow_text(ov, (x + 14, y), ln, fs, ct.WHITE, blur=6, a=210); y += 40
    ov.save(path)


def cut(name, a, b, mode="crop", top=None, bot=None):
    if mode == "text3":
        ovp = f"{OUT}/_{name}_ov.png"; chose_overlay(ovp)
        dur = b - a; fo = max(dur - 0.12, 0)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", REEL, "-loop", "1", "-i", ovp,
                        # 6% push-in: the reel's word pill above the surgeon falls outside the frame, patient stays in
                        "-filter_complex", f"[0:v]scale=1146:-2:flags=lanczos,crop={W}:{H}:33:ih-{H},setsar=1[v];"
                                           "[v][1:v]overlay=0:0:shortest=1,format=yuv420p[o]",
                        "-map", "[o]", "-map", "0:a", "-t", f"{dur:.3f}",
                        "-af", f"afade=t=in:d=0.06,afade=t=out:st={fo:.3f}:d=0.12",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", f"{OUT}/{name}.mp4"], check=True)
        os.remove(ovp); return
    if mode == "stack": return stack(name, a, b, top, bot)
    if mode == "text":
        ovp = f"{OUT}/_{name}_ov.png"; headline_overlay(ovp)
        dur = (b or float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", REEL],
                                         capture_output=True, text=True).stdout)) - a
        fo = max(dur - 0.12, 0)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", REEL, "-loop", "1", "-i", ovp,
                        "-filter_complex", f"[0:v]crop={W}:{H}:0:1920-{H},setsar=1[v];[v][1:v]overlay=0:0:shortest=1,format=yuv420p[o]",
                        "-map", "[o]", "-map", "0:a", "-t", f"{dur:.3f}",
                        "-af", f"afade=t=in:d=0.06,afade=t=out:st={fo:.3f}:d=0.12",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", f"{OUT}/{name}.mp4"], check=True)
        os.remove(ovp); return
    dur = (b or float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", REEL],
                                     capture_output=True, text=True).stdout)) - a
    fo = max(dur - 0.12, 0)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", REEL,
                    "-vf", (f"crop={W}:{H}:0:{CROP_Y},setsar=1" if mode == "crop" else
                            f"scale=-2:{H}:flags=lanczos,pad={W}:{H}:(ow-iw)/2:0:color={G},setsar=1"),  # whole frame: content sits low
                    "-af", f"afade=t=in:d=0.06,afade=t=out:st={fo:.3f}:d=0.12",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", f"{OUT}/{name}.mp4"], check=True)


ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "carousel_lipofilling", "src")
LILAC_D = (247, 207, 217)  # soft pastel pink: bullets, highlight frames, step badges


def before_badge(path, cx, cy, r=165):
    """round inset with the patient's "before" photo and a navy НАШЕ ДО tag"""
    from PIL import Image, ImageDraw
    S = 3
    im = Image.open(os.path.join(ASSETS, "before_front.jpg")).convert("RGB")
    sq = im.crop((245, 190, 1225, 1170)).resize((2 * r * S, 2 * r * S), Image.LANCZOS)
    m = Image.new("L", sq.size, 0); ImageDraw.Draw(m).ellipse((0, 0, sq.width - 1, sq.height - 1), fill=255)
    hi = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0)); d = ImageDraw.Draw(hi)
    ring = 7 * S
    d.ellipse(((cx - r) * S - ring, (cy - r) * S - ring, (cx + r) * S + ring, (cy + r) * S + ring), fill=ct.WHITE + (255,))
    hi.paste(sq, ((cx - r) * S, (cy - r) * S), m)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    # soft shadow under the badge
    sh = Image.new("L", (W, H), 0); ImageDraw.Draw(sh).ellipse((cx - r - 4, cy - r + 6, cx + r + 4, cy + r + 14), fill=150)
    from PIL import ImageFilter
    sh = sh.filter(ImageFilter.GaussianBlur(14)); shadow = Image.new("RGBA", (W, H), ct.GRAPHITE + (0,)); shadow.putalpha(sh)
    ov.alpha_composite(shadow); ov.alpha_composite(hi.resize((W, H), Image.LANCZOS))
    d = ImageDraw.Draw(ov); f = ct.bebas(44); t = "НАШЕ ДО"; tw = f.getlength(t)
    y = cy + r - 26
    d.rounded_rectangle((cx - tw / 2 - 22, y, cx + tw / 2 + 22, y + 54), radius=14, fill=(16, 46, 70, 255))
    d.text((cx, y + 29), t, font=f, fill=ct.WHITE, anchor="mm")
    ov.save(path)


def add_badge(name, cx, cy, r=165):
    src = f"{OUT}/{name}.mp4"; tmp = f"{OUT}/_{name}_b.mp4"; png = f"{OUT}/_{name}_badge.png"
    before_badge(png, cx, cy, r)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-loop", "1", "-i", png,
                    "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1,format=yuv420p[o]", "-map", "[o]", "-map", "0:a",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-c:a", "copy", "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, src); os.remove(png)


def cta():
    """info slide in the surgeon's dark style: what the consultation covers, then how to book (annotated profile screenshots)"""
    from PIL import Image, ImageDraw
    ct.LILAC = LILAC_D  # the rule under the heading follows the same pink
    c = Image.new("RGBA", (W, H), ct.GRAPHITE + (255,))
    c.alpha_composite(ct.vgrad(0, H, 0, 0.55, (8, 10, 14)))  # slightly deeper towards the bottom
    ct.put_logo(c, y=196, alpha=0.95)
    d = ImageDraw.Draw(c); bw = 920; x = (W - bw) // 2
    y = ct.Block(bw, "center").head("На консультации разберём", 78, 14).rule(0.14, 28).draw(c, x, 284)
    items = ["подойдёт ли вам липофилинг", "есть ли донорские зоны", "какой объём можно добавить",
             "нужна ли подтяжка или импланты"]
    f = ct.mont(31, 500)
    for it in items:
        tw = f.getlength(it); x0 = (W - tw) / 2
        d.ellipse((x0 - 26, y + 13, x0 - 15, y + 24), fill=LILAC_D); d.text((x0, y), it, font=f, fill=ct.WHITE); y += 48
    y += 40
    fh = ct.bebas(54); t = "КАК ЗАПИСАТЬСЯ"; d.text(((W - fh.getlength(t)) / 2, y), t, font=fh, fill=ct.WHITE); y += 78
    # screenshots: profile header (step 1) and the links sheet (step 2)
    shots = [("profile_header.jpg", [(30, 805, 760, 890)], "1", "Нажмите на ссылку\nв шапке профиля"),
             ("profile_links.jpg", [(30, 180, 1000, 310), (30, 660, 1000, 790)], "2", "Выберите WhatsApp\nили Telegram")]
    hgt = 392; gap = 26
    ims = [Image.open(os.path.join(ASSETS, fn)).convert("RGB") for fn, *_ in shots]
    ws = [int(i.width * hgt / i.height) for i in ims]
    x0 = (W - sum(ws) - gap) // 2
    S = 3
    for (fn, boxes, num, cap), im_, w_ in zip(shots, ims, ws):
        k = hgt / im_.height
        card = im_.resize((w_ * S, hgt * S), Image.LANCZOS).convert("RGBA")
        hd = ImageDraw.Draw(card)
        for bx0, by0, bx1, by1 in boxes:
            hd.rounded_rectangle((bx0 * k * S, by0 * k * S, bx1 * k * S, by1 * k * S), radius=16 * S, outline=LILAC_D + (255,), width=4 * S)
        m = Image.new("L", card.size, 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, card.width - 1, card.height - 1), radius=22 * S, fill=255)
        card.putalpha(m); card = card.resize((w_, hgt), Image.LANCZOS)
        c.alpha_composite(card, (x0, y))
        d.rounded_rectangle((x0 - 1, y - 1, x0 + w_, y + hgt), radius=22, outline=LILAC_D + (140,), width=2)  # card edge on the dark ground
        d.ellipse((x0 - 18, y - 18, x0 + 36, y + 36), fill=LILAC_D, outline=ct.GRAPHITE, width=4)
        d.text((x0 + 9, y + 10), num, font=ct.bebas(40), fill=ct.INK, anchor="mm")
        fc = ct.mont(27, 500)
        for li, ln in enumerate(cap.split("\n")):
            d.text((x0 + w_ / 2, y + hgt + 24 + li * 36), ln, font=fc, fill=ct.WHITE, anchor="ma")
        x0 += w_ + gap
    ct.save_jpg(c, f"{OUT}/08_cta.jpg")


def booking_steps(c, y, hgt=340, gap=26, text=None):
    """the two annotated profile screenshots (step 1: link in the header, step 2: pick a messenger); returns the y below"""
    from PIL import Image, ImageDraw
    text = text or ct.WHITE
    d = ImageDraw.Draw(c)
    shots = [("profile_header.jpg", [(30, 805, 760, 890)], "1", "Нажмите на ссылку\nв шапке профиля"),
             ("profile_links.jpg", [(30, 180, 1000, 310), (30, 660, 1000, 790)], "2", "Выберите WhatsApp\nили Telegram")]
    ims = [Image.open(os.path.join(ASSETS, fn)).convert("RGB") for fn, *_ in shots]
    ws = [int(i.width * hgt / i.height) for i in ims]
    x0 = (W - sum(ws) - gap) // 2; S = 3
    for (fn, boxes, num, cap), im_, w_ in zip(shots, ims, ws):
        k = hgt / im_.height
        card = im_.resize((w_ * S, hgt * S), Image.LANCZOS).convert("RGBA"); hd = ImageDraw.Draw(card)
        for bx0, by0, bx1, by1 in boxes:
            hd.rounded_rectangle((bx0 * k * S, by0 * k * S, bx1 * k * S, by1 * k * S), radius=16 * S, outline=LILAC_D + (255,), width=4 * S)
        m = Image.new("L", card.size, 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, card.width - 1, card.height - 1), radius=22 * S, fill=255)
        card.putalpha(m); card = card.resize((w_, hgt), Image.LANCZOS)
        c.alpha_composite(card, (x0, y))
        d.rounded_rectangle((x0 - 1, y - 1, x0 + w_, y + hgt), radius=22, outline=LILAC_D + (140,), width=2)
        d.ellipse((x0 - 18, y - 18, x0 + 36, y + 36), fill=LILAC_D, outline=ct.GRAPHITE, width=4)
        d.text((x0 + 9, y + 10), num, font=ct.bebas(40), fill=ct.INK, anchor="mm")
        for li, ln in enumerate(cap.split("\n")):
            ct.shadow_text(c, (x0 + w_ / 2 - ct.mont(26, 500).getlength(ln) / 2, y + hgt + 22 + li * 34), ln, ct.mont(26, 500), text)
        x0 += w_ + gap
    return y + hgt + 22 + 2 * 34


def important():
    """blurred surgeon in the OR, ВАЖНО accent: no answers in Direct, write to a messenger; profile navigation; room left for a link"""
    from PIL import Image, ImageDraw, ImageFilter
    ct.LILAC = LILAC_D
    im = Image.open(os.path.join(ASSETS, "surgeon_or.jpg")).convert("RGB")
    bg = ct.fit_cover(im, W, H, cy=0.35).filter(ImageFilter.GaussianBlur(26)).convert("RGBA")
    bg.alpha_composite(Image.new("RGBA", (W, H), ct.GRAPHITE + (150,)))  # dim the light wall so white type reads
    c = bg; d = ImageDraw.Draw(c)
    ct.put_logo(c, y=96, alpha=0.95)
    f = ct.bebas(46); t = "ВАЖНО"; tw = f.getlength(t); y = 168
    d.rounded_rectangle(((W - tw) / 2 - 26, y, (W + tw) / 2 + 26, y + 60), radius=30, fill=LILAC_D + (255,))
    d.text((W / 2, y + 33), t, font=f, fill=ct.INK, anchor="mm")
    y = ct.Block(W - 2 * ct.MARGIN, "center").head("Я не могу отвечать\nвам в директе", 86, 18).rule(0.14, 24) \
        .body("Пишите сразу в удобном для вас мессенджере", 33, 0).draw(c, ct.MARGIN, y + 92)
    y = booking_steps(c, y + 60)
    ct.save_jpg(c, f"{OUT}/09_important.jpg")  # free space below the steps is left for the link


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for n, a, b, *m in SLIDES:
        cut(n, a, b, *m); print("done", n, flush=True)
    add_badge("05_volume", 870, 1080); add_badge("07_this_task", 870, 1110)
    cta(); print("done cta")
    important(); print("done important")
