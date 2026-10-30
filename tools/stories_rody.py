"""@mcg.va — stories «про мои роды» (1080x1920), one short thought per story: photo stories → JPG, video stories → MP4.
Type: main = Montserrat ExtraBold, second = thin Montserrat 300, accents = thin Montserrat Italic; all small.
The picture stays full-bleed and sharp: small text pieces sit in different parts of the frame, away from faces (MediaPipe,
face_boxes.py in the mpenv) — white with a soft shadow on dark areas, ink on light ones. Voice videos keep their sound and get
phrase subtitles.
  python3 -I stories_rody.py MATERIALS_DIR OUT_DIR MPENV_PYTHON MODELS_DIR [N ...]
"""
import io, json, os, re, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import W, H, SAFE_TOP, SAFE_BOTTOM, FONT_DIR, wrap

HERE = os.path.dirname(os.path.abspath(__file__))
MONT = os.path.join(FONT_DIR, "Montserrat-VariableFont_wght.ttf")
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
MARGIN, GAP, FACE_PAD = 70, 44, 50
WHITE, INK, SHADE = (255, 255, 255), (24, 20, 18), (22, 15, 12)

def mont_it(size, wght):
    f = ImageFont.truetype(os.path.join(FONT_DIR, "Montserrat-Italic-VariableFont_wght.ttf"), size); f.set_variation_by_axes([wght]); return f

def mont(size, wght):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([wght]); return f

STYLE = {"main": (lambda: mont(52, 700), 1.15, 760),          # font, line lead (× size), column width — sizes as in the blog's stories
         "second": (lambda: mont(40, 400), 1.32, 720),
         "script": (lambda: mont_it(44, 300), 1.3, 720)}                 # accent: thin Montserrat Italic

def emoji_img(ch, h):
    f = ImageFont.truetype(EMOJI, 109); im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((80, 80), ch, font=f, embedded_color=True, anchor="mm"); im = im.crop(im.getbbox())
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)

class Block:
    """one text piece: wrapped lines, its size, later its place and colour"""
    def __init__(self, kind, text):
        mk, lead, self.cw = STYLE[kind]; self.kind, self.f = kind, mk()
        self.emo = text[-1] if ord(text[-1]) > 0x2600 else None; self.text = text[:-1].rstrip() if self.emo else text
        self.step = self.f.size * lead; self.rewrap(self.cw)

    def rewrap(self, cw):
        self.lines = wrap(self.text, self.f, cw - (80 if self.emo else 0))
        self.w = int(max(self.f.getlength(l) for l in self.lines) + (80 if self.emo else 0))
        self.h = int(self.step * (len(self.lines) - 1) + self.f.size * 1.15)

def hits(r, boxes, pad):
    return any(not (r[2] + pad <= b[0] or r[0] - pad >= b[2] or r[3] + pad <= b[1] or r[1] - pad >= b[3]) for b in boxes)

def person_share(r, person):
    """share of the rectangle covered by the person mask (20 px cells)"""
    g = person[max(r[1] // 20, 0):r[3] // 20 + 1, max(r[0] // 20, 0):r[2] // 20 + 1]
    return float(g.mean()) if g.size else 0.0

def place(blocks, people, at=None, one_side=False):
    """try the wished / evenly spread heights first; if a piece found no room below the previous one, pack them higher"""
    n = len(blocks)
    for tg in ([at] if at else []) + [None, [i * 0.22 for i in range(n)], [0.0] * n]:
        if _place(blocks, people, tg, one_side): return
    _place(blocks, people, at, one_side)

def _place(blocks, people, at=None, one_side=False):
    """reading order top → bottom; each piece goes to a free spot near its share of the height, sides alternating;
    never over a face, and over the body only if there is no other room; at = wished heights (0..1 of the safe zone) per piece"""
    faces, person = people
    fx = np.mean([(b[0] + b[2]) / 2 for b in faces]) if faces else W / 2
    side = "left" if fx >= W / 2 else "right"; n = len(blocks); span = SAFE_BOTTOM - SAFE_TOP; prev = SAFE_TOP - GAP; used = []; ok = True
    for i, b in enumerate(blocks):
        tgt = SAFE_TOP + span * (at[i] if at else (i + 0.5) / n) - b.h / 2; best = None
        reserve = sum(c.h + GAP for c in blocks[i + 1:])                   # leave room below for the pieces still to come
        for lim, cw in [(l, c) for l in (0.04, 0.12, 0.3, 1.01) for c in (b.cw, int(b.cw * 0.72), int(b.cw * 0.52))]:
            b.rewrap(cw)                                                   # first a narrower column, only then over the body
            for s in (side, "right" if side == "left" else "left"):
                x = MARGIN if s == "left" else W - MARGIN - b.w
                for y in range(int(max(prev + GAP, SAFE_TOP)), SAFE_BOTTOM - b.h - reserve + 1, 8):
                    r = (x, y, x + b.w, y + b.h)
                    if hits(r, faces, FACE_PAD) or hits(r, used, GAP // 2): continue
                    ps = person_share(r, person)
                    if ps > lim: continue
                    sc = abs(y - tgt) + (0 if s == side else 260) + 900 * ps
                    if best is None or sc < best[0]: best = (sc, x, y, s, cw)
            if best: break
        if best is None: ok = False                                        # no room below the last piece: any free spot, still off faces
        if best is None:
            for cw in (b.cw, int(b.cw * 0.72), int(b.cw * 0.52)):
                b.rewrap(cw)
                for s in ("left", "right"):
                    x = MARGIN if s == "left" else W - MARGIN - b.w
                    for y in range(SAFE_TOP, SAFE_BOTTOM - b.h + 1, 8):
                        r = (x, y, x + b.w, y + b.h)
                        if hits(r, faces, FACE_PAD) or hits(r, used, GAP // 2): continue
                        sc = abs(y - tgt) + 900 * person_share(r, person)
                        if best is None or sc < best[0]: best = (sc, x, y, s, cw)
        if best is None:
            b.rewrap(b.cw); x = MARGIN if side == "left" else W - MARGIN - b.w; y = int(min(max(prev + GAP, SAFE_TOP), SAFE_BOTTOM - b.h)); s = side
        else: _, x, y, s, cw = best; b.rewrap(cw)
        b.x, b.y, b.side = x, y, s; used.append((x, y, x + b.w, y + b.h)); prev = max(prev, y + b.h)
        side = s if one_side else ("right" if s == "left" else "left")
    return ok

def colour(blocks, frame):
    """ink on calm light ground, white on calm dark ground; anything busy or mid-grey → white over a light black haze"""
    a = np.asarray(frame.convert("L"), np.float32)
    for b in blocks:
        r = a[b.y:b.y + b.h, b.x:b.x + b.w]; m, sd = r.mean(), r.std()
        b.light = m > 172 and sd < 38; b.haze = not b.light and (m > 100 or sd > 42)

def draw(blocks, size=(W, H)):
    """RGBA layer with the text pieces (soft glyph shadow under white, faint glow under ink)"""
    txt = Image.new("RGBA", size, (0, 0, 0, 0)); sh = Image.new("L", size, 0); gl = Image.new("L", size, 0)
    d, ds, dg = ImageDraw.Draw(txt), ImageDraw.Draw(sh), ImageDraw.Draw(gl)
    for b in blocks:
        col = INK if b.light else WHITE; under = dg if b.light else ds
        for j, l in enumerate(b.lines):
            yb = b.y + j * b.step + b.f.size * 0.85; lw = b.f.getlength(l) + (80 if b.emo and j == len(b.lines) - 1 else 0)
            x = b.x if b.side == "left" else b.x + b.w - lw
            d.text((x, yb), l, font=b.f, fill=col + (255,), anchor="ls"); under.text((x, yb + 2), l, font=b.f, fill=150, anchor="ls")
            if b.emo and j == len(b.lines) - 1:
                e = emoji_img(b.emo, int(b.f.size * 0.8)); txt.alpha_composite(e, (int(x + b.f.getlength(l) + 16), int(yb - e.height * 0.9)))
    out = Image.new("RGBA", size, (0, 0, 0, 0)); hz = Image.new("L", size, 0); dh = ImageDraw.Draw(hz)
    for b in blocks:                                                       # the light black haze under hard-to-read pieces
        if getattr(b, "haze", False): dh.rounded_rectangle((b.x - 46, b.y - 34, b.x + b.w + 46, b.y + b.h + 34), 60, fill=255)
    if hz.getbbox():
        h_ = Image.new("RGBA", size, (8, 6, 5, 0)); h_.putalpha(hz.filter(ImageFilter.GaussianBlur(38)).point(lambda v: v * 0.42))
        out.alpha_composite(h_)
    s1 = Image.new("RGBA", size, SHADE + (0,)); s1.putalpha(sh.filter(ImageFilter.GaussianBlur(7))); out.alpha_composite(s1)
    s2 = Image.new("RGBA", size, (255, 255, 255, 0)); s2.putalpha(gl.filter(ImageFilter.GaussianBlur(9)).point(lambda v: v * 0.6))
    out.alpha_composite(s2); out.alpha_composite(txt)
    return out

def cover(im, fx=0.5, fy=0.5, zoom=1.0, fit="cover"):
    if zoom < 1:                                                           # contained, smaller, on the picture's own corner colour
        s = min(W / im.width, H / im.height) * zoom; p = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        bg = Image.new("RGB", (W, H), im.getpixel((2, im.height // 2))); x, y = (W - p.width) // 2, int((H - p.height) * fy)
        bg.paste(p, (x, y)); cover.card = (x, y, x + p.width, y + p.height)         # the text keeps off the picture itself
        return bg
    cover.card = None
    if fit == "contain":                                                   # full width, the backdrop's own colour above, seam feathered
        s = W / im.width; p = im.resize((W, round(im.height * s)), Image.LANCZOS); y0 = H - p.height
        top = np.asarray(p, np.float32)[:12].mean((0, 1)); bg = Image.new("RGB", (W, H), tuple(int(v) for v in top))
        m = Image.fromarray((np.clip(np.arange(p.height) / 160, 0, 1)[:, None].repeat(W, 1) * 255).astype(np.uint8))
        bg.paste(p, (0, y0), m); return bg
    s = max(W / im.width, H / im.height) * zoom; im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    x = int(min(max(fx * im.width - W / 2, 0), im.width - W)); y = int(min(max(fy * im.height - H / 2, 0), im.height - H))
    return im.crop((x, y, x + W, y + H))

def faces_of(frames, avoid=()):
    """face boxes over the given 1080x1920 frames (mpenv), plus any boxes the story lists by hand"""
    with tempfile.TemporaryDirectory() as td:
        ps = []
        for i, f in enumerate(frames): p = os.path.join(td, f"{i}.png"); f.save(p); ps.append(p)
        r = subprocess.run([MPPY, os.path.join(HERE, "face_boxes.py"), MODELS] + ps, capture_output=True, text=True, check=True)
    j = json.loads(r.stdout.strip().splitlines()[-1])
    return j["faces"] + [list(a) for a in avoid], np.array(j["person"], np.float32)

def photo_story(src, blocks, out, fx=0.5, fy=0.5, zoom=1.0, avoid=(), at=None, one_side=False, fit="cover"):
    import pillow_heif; pillow_heif.register_heif_opener()
    bg = cover(ImageOps.exif_transpose(Image.open(src)).convert("RGB"), fx, fy, zoom, fit)
    if cover.card: avoid = tuple(avoid) + (cover.card,)
    bs = [Block(k, t) for k, t in blocks]; place(bs, faces_of([bg], avoid), at, one_side); colour(bs, bg)
    img = bg.convert("RGBA"); img.alpha_composite(draw(bs)); img.convert("RGB").save(out, quality=95, subsampling=0)
    return bs

def probe_dur(src):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                                capture_output=True, text=True).stdout)

def frame_at(src, t, crop):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", src, "-frames:v", "1", "-vf", crop, "-f", "image2pipe",
                          "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(raw)).convert("RGB")

def subs_cues(words_json, t0, t1, fix=(), drop=()):
    """phrase cues (≤ 3 words / 24 letters, broken at punctuation) from whisper words in [t0, t1); fix = word replacements"""
    ws = [w for w in json.load(open(words_json)) if t0 <= w[1] < t1 and not any(a <= w[1] < b for a, b in drop)]
    toks = []
    for w, a, b in ws:
        if w.startswith("-") and toks: toks[-1][0] += w; toks[-1][2] = b; continue
        toks.append([w, a, b])
    for i, t in enumerate(toks):                                           # sentence-case: whisper capitalises after its own pauses
        if i and not re.search(r"[.!?…»]$", toks[i - 1][0]) and t[0][:1].isupper() and t[0] not in ("Дьявол", "Prada"): t[0] = t[0].lower()
        for a, b in fix: t[0] = t[0].replace(a, b)
    cues, cur = [], []
    for t in toks:
        cur.append(t); txt = " ".join(c[0] for c in cur)
        if len(cur) >= 3 or len(txt) >= 22 or re.search(r"[,.!?…»]$", t[0]): cues.append(cur); cur = []
    if cur: cues.append(cur)
    out = []
    for i, c in enumerate(cues):
        end = cues[i + 1][0][1] if i + 1 < len(cues) and cues[i + 1][0][1] - c[-1][2] < 0.6 else c[-1][2] + 0.35
        out.append((c[0][1] - t0, end - t0, " ".join(x[0] for x in c).rstrip(",")))
    return out

def sub_patch(text):
    f = mont(44, 600); lines = wrap(text, f, 900); w = int(max(f.getlength(l) for l in lines)) + 40; h = 62 * len(lines) + 30
    lay = Image.new("RGBA", (w, h), (0, 0, 0, 0)); sh = Image.new("L", (w, h), 0); d, ds = ImageDraw.Draw(lay), ImageDraw.Draw(sh)
    for j, l in enumerate(lines):
        d.text((w / 2, 20 + j * 62 + 44), l, font=f, fill=WHITE + (255,), anchor="ms")
        ds.text((w / 2, 22 + j * 62 + 44), l, font=f, fill=190, anchor="ms")
    s = Image.new("RGBA", (w, h), SHADE + (0,)); s.putalpha(sh.filter(ImageFilter.GaussianBlur(6))); s.alpha_composite(lay)
    return s

def privacy_pass(src, ss, dur, crop, out):
    """the cropped clip with the chest blurred, tracked frame by frame (chest_blur.py in the mpenv)"""
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{ss:.2f}", "-t", f"{dur:.2f}", "-i", src, "-vf", crop + ",fps=30",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    cb = subprocess.Popen([MPPY, os.path.join(HERE, "chest_blur.py"), MODELS, str(W), str(H), "30"], stdin=dec.stdout,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p", out], stdin=cb.stdout, check=True)
    dec.wait(); cb.wait()

def video_story(src, blocks, out, ss=0.0, dur=None, fx=0.5, fy=0.5, zoom=1.0, avoid=(), voice=None, at=None, one_side=False,
                chest=False, blur=False):
    """the clip full-bleed; the text pieces fade in; voice=(words.json, fixes, drops) keeps the sound and adds phrase subtitles;
    chest=True blurs the chest (tracked), blur=True blurs the whole picture"""
    probe = frame_at(src, ss, "null"); sw, sh_ = probe.size
    s = max(W / sw, H / sh_) * zoom; pw, ph = round(sw * s / 2) * 2, round(sh_ * s / 2) * 2
    x = int(min(max(fx * pw - W / 2, 0), pw - W)); y = int(min(max(fy * ph - H / 2, 0), ph - H))
    crop = f"scale={pw}:{ph}:flags=lanczos,crop={W}:{H}:{x}:{y}"
    words = sum(len(t.split()) for _, t in blocks)
    if dur is None: dur = float(np.clip(words / 2.5 + 2.0, 5, 10))
    dur = min(dur, probe_dur(src) - ss)
    frames = [frame_at(src, ss + dur * 0.9 * k / 5, crop) for k in range(6)]
    people = faces_of(frames, avoid); faces = people[0]; sy = None
    if voice:                                                              # subtitles low, unless a face is there; the text keeps off them
        sy = 1420 if not hits((90, 1380, W - 90, 1560), faces, 0) else 300
        people = (faces + [(60, sy - 40, W - 60, sy + 150)], people[1])
    bs = [Block(k, t) for k, t in blocks]; place(bs, people, at, one_side)
    colour(bs, Image.fromarray(np.median(np.stack([np.asarray(f) for f in frames]), 0).astype(np.uint8)))
    with tempfile.TemporaryDirectory() as td:
        lay = os.path.join(td, "text.png"); draw(bs).save(lay)
        ins = ["-ss", f"{ss:.2f}", "-t", f"{dur:.2f}", "-i", src, "-loop", "1", "-t", f"{dur:.2f}", "-i", lay]
        vin = f"[0:v]{crop},fps=30"
        if chest:                                                          # the blurred clip replaces the picture; sound stays from src
            pv = os.path.join(td, "private.mp4"); privacy_pass(src, ss, dur, crop, pv); ins += ["-i", pv]; vin = f"[{ins.count('-i') - 1}:v]null"
        if blur: vin += ",gblur=sigma=34"
        fg = f"{vin},format=yuv420p[v0];[1:v]format=rgba,fade=in:st=0.2:d=0.5:alpha=1[t];[v0][t]overlay=0:0[v1]"
        last = "v1"
        if voice:
            cues = subs_cues(*voice[:1], ss, ss + dur, *voice[1:]); n0 = ins.count("-i")        # first subtitle input
            for i, (a, b, t) in enumerate(cues):
                p = os.path.join(td, f"s{i}.png"); sub_patch(t).save(p); ins += ["-i", p]
            for i, (a, b, t) in enumerate(cues):
                fg += f";[{last}][{n0 + i}:v]overlay=(W-w)/2:{sy}:enable='between(t,{max(a, 0):.2f},{b:.2f})'[v{i + 2}]"; last = f"v{i + 2}"
            fg += f";[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[a]"
            amap = ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
        else: amap = ["-an"]
        fg += f";[{last}]format=yuv420p[vout]"
        subprocess.run(["ffmpeg", "-v", "error", "-y"] + ins + ["-filter_complex", fg, "-map", "[vout]"] + amap +
                       ["-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high", "-r", "30",
                        "-movflags", "+faststart", out], check=True)
    return bs

def card_story(src, blocks, out, icon=None, ss=0.0, fx=0.5, fy=0.5, at=None):
    """no repeated frame: the picture only as a soft blurred ground under the text (a still JPG, no animation)"""
    import pillow_heif; pillow_heif.register_heif_opener()
    if src.lower().endswith((".mov", ".mp4")): im = frame_at(src, ss, "null")
    else: im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    bg = cover(im, fx, fy).resize((W // 10, H // 10), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W, H), Image.BICUBIC)
    bg = Image.fromarray((np.asarray(bg, np.float32) * 0.62).astype(np.uint8))           # calm and darker: white type reads
    bs = [Block(k, t) for k, t in blocks]; place(bs, ([], np.zeros((H // 20, W // 20), np.float32)), at)
    for b in bs: b.light, b.haze = False, False
    img = bg.convert("RGBA"); img.alpha_composite(draw(bs)); img.convert("RGB").save(out, quality=95, subsampling=0)

V8208 = ("IMG_8208_m.json", (("прадо", "Prada"),), ())
V8214 = ("IMG_8214_m.json", (("прадо", "Prada"),), ((17.3, 19.5),))
V8207 = ("IMG_8207_m.json", (), ())

S = [  # (photo|video, file, options, [(kind, text), ...]) — one short thought per story, in reading order
    ("photo", "IMG_9141.JPG", dict(fit="contain", at=(0.03, 0.15), one_side=True), [("main", "Я тут наконец-то решилась рассказать"), ("script", "про свои роды")]),
    ("photo", "IMG_0764.JPG", dict(fy=0.5), [("second", "Если честно, очень долго откладывала эту тему,"),
                                             ("second", "потому что вообще не понимала, хочу ли вообще рассказывать.")]),
    ("video", "IMG_0751.MOV", dict(ss=5.0), [("second", "Всю беременность я думала, что самое страшное, что меня ждёт, —"),
                                             ("main", "это сами роды и тяжесть материнства, но…")]),
    ("photo", "IMG_7952.JPG", dict(fy=0.5), [("second", "Я готовилась к боли, к тому, что всё может пойти не по плану,"),
                                             ("second", "представляла себе миллион разных сценариев.")]),
    ("video", "IMG_0755.mov", dict(ss=1.0, avoid=((560, 300, 960, 700),)), [("main", "Но реальность оказалась"), ("script", "немного другой")]),
    ("video", "IMG_8207.mov", dict(voice=V8207, dur=9.0), [("main", "Мои роды начались ровно в ПДР!")]),
    ("photo", "IMG_8219.png", dict(fy=0.12, zoom=0.62), [("second", "Муж в этот день уехал на выступление,"),
                                             ("main", "а я осталась дома со схватками")]),
    ("video", "IMG_8216.mov", dict(ss=6.0), [("second", "Причём весь день я лежала, смотрела кино, ела, слушала музыку"),
                                             ("script", "и отдыхала в полной темноте")]),
    ("video", "IMG_8208.mov", dict(voice=V8208, dur=53.5, chest=True), []),
    ("video", "IMG_8214.mov", dict(voice=V8214, ss=2.0, dur=33.0, chest=True), [("script", "Вот примерно о таких родах я и мечтала")]),
    ("video", "IMG_8218.mov", dict(ss=2.5), [("second", "А потом у меня отошли воды, схватки стали сильнее,"),
                                             ("main", "и знаете, что я решила сделать?")]),
    ("video", "IMG_8217.mov", dict(ss=0.5, dur=7.0, chest=True), [("script", "Помыть полы и высушить голову после душа 😂")]),
    ("card", "IMG_8217.mov", dict(ss=17.0, icon="mop"), [("main", "Не спрашивайте, зачем."),
                                                       ("second", "У человека активные роды, а она уборку затеяла.")]),
    ("video", "IMG_8217.mov", dict(ss=34.0, dur=8.0), [("second", "Я расстраивалась и думала, что у меня максимум 5 см раскрытие"),
                                                       ("main", "(но на самом деле уже было 10 см)")]),
    ("video", "IMG_8217.mov", dict(ss=58.0, dur=6.0, chest=True), [("script", "Кстати, голову до конца я так и не высушила.")]),
    ("video", "IMG_6146.mov", dict(ss=2.0, dur=6.0, blur=True), [("main", "В итоге в роддом мы буквально летели."),
                                                      ("second", "Потому что я чуть не родила дома, а затем и в такси.")]),
    ("video", "IMG_6146.mov", dict(ss=4.5, dur=5.0, blur=True), [("second", "Просто бедный таксист."),
                                                      ("second", "Он сидел весь зелёный от страха.")]),
    ("photo", "IMG_2463.heic", dict(fy=0.5), [("second", "Меня просто забрала моя акушерка из дома, и мы поехали в роддом.")]),
    ("video", "IMG_6146.mov", dict(ss=7.0, dur=6.0, blur=True), [("second", "Мы не могли долго выехать со двора, и акушерка думала уже звонить "
                                                                 "в полицию, чтобы нас сопроводили."),
                                                      ("main", "Потому что у меня уже начинались потуги.")]),
    ("video", "IMG_6146.mov", dict(ss=10.5, dur=7.0, blur=True), [("second", "Меня встретили с каталкой, быстро отвезли в родовую,"),
                                                       ("main", "и уже через час после поступления я родила дочку")]),
    ("photo", "IMG_6225.heic", dict(fy=0.5), [("second", "И знаете что? Я считаю, что роды у меня прошли"), ("script", "просто шикарно!")]),
    ("photo", "102.jpg", dict(fy=0.45), [("second", "После такого опыта я реально готова идти за вторым.")]),
    ("photo", "IMG_0042.HEIC", dict(fy=0.5), [("main", "Но потом меня перевели в послеродовое отделение.")]),
    ("video", "IMG_6145.mov", dict(ss=1.8, dur=5.0), [("second", "И вот здесь началось то,"),
                                                      ("main", "к чему я вообще не была готова")]),
    ("photo", "IMG_6190.heic", dict(fy=0.45), [("second", "Я осталась одна с новорождённым ребёнком."),
                                               ("second", "Уставшая, без сна, без понимания, что делать дальше.")]),
    ("photo", "IMG_2460.heic", dict(fy=0.5), [("script", "И мне казалось, что хотя бы здесь мне помогут.")]),
    ("video", "IMG_6145.mov", dict(ss=9.5, dur=3.6), [("main", "Дочка всю ночь плакала,"), ("second", "я буквально падала с ног.")]),
    ("card", "IMG_6190.heic", dict(icon="drops"), [("second", "Она захлёбывалась водой."),
                                                                  ("second", "Я не понимала, что с ней происходит.")]),
    ("card", "IMG_6225.heic", dict(icon="qmark"), [
        ("second", "Оказывается, ребёнок может наглотаться околоплодных вод, и это нормально, что вода потом выходит."),
        ("second", "Почему-то в роддоме мне никто не мог этого объяснить."), ("script", "Я чуть не умерла со страха.")]),
    ("card", "IMG_6145.mov", dict(ss=2.6, icon="bell"), [("second", "Несколько раз звала медсестру, потому что не понимала, что происходит "
                                                                 "и как ей помочь.")]),
    ("card", "IMG_0042.HEIC", dict(icon="bolt"), [("second", "И вместо объяснений услышала:"),
                                                                 ("main", "«Привыкай, дальше будет хуже»"),
                                                                 ("script", "Я до сих пор помню эту фразу.")]),
    ("photo", "DSC00516.JPG", dict(fy=0.4), [("main", "И это была только первая ночь.")]),
    ("photo", "IMG_2194.HEIC", dict(fx=0.22, fy=0.5, at=(0.25, 0.5, 0.72), one_side=True), [("second", "Дальше были оскорбления из-за моего отказа от грудного вскармливания,"),
                                              ("second", "противоречивые замечания врачей"),
                                              ("second", "и постоянное ощущение, что я всё делаю неправильно.")]),
    ("photo", "IMG_6190.heic", dict(fx=0.5, fy=0.4, zoom=1.3, at=(0.0, 0.1)), [("second", "Хотя я только что родила первого ребёнка"),
                                                                ("script", "и просто пыталась разобраться")]),
    ("photo", "DSC00770.JPG", dict(fy=0.5, at=(0.0, 0.14), one_side=True, avoid=((760, 1010, 1080, 1920),)), [("second", "Самое обидное, что я ведь не ждала какого-то особенного отношения."),
                                             ("second", "Мне просто хотелось, чтобы мне объяснили, помогли, дали немного "
                                                        "восстановиться после родов.")]),
    ("card", "DSC00516.JPG", dict(icon="heart"), [("main", "Хотелось просто человеческого отношения,"),
                                                                  ("second", "потому что после родов ты настолько сильно уязвима, "
                                                                             "как никогда.")]),
    ("photo", "IMG_7929.HEIC", dict(fy=0.5), [("second", "Да, кому-то могло быть хуже."),
                                              ("second", "Но я здесь не сравниваю свою историю с чужой.")]),
    ("card", "IMG_0764.JPG", dict(icon="cloud"), [("script", "Для меня этот опыт оказался травмирующим,"),
                                                                 ("second", "и я потом ещё долго с этим разбиралась.")]),
    ("photo", "100.jpg", dict(fy=0.42, at=(0.0, 0.1), one_side=True), [("main", "Я решила сделать целую серию рилс,"),
                                         ("second", "потому что в одно видео всё это просто невозможно уместить.")]),
    ("card", "102.jpg", dict(icon="clapper"), [("second", "Расскажу и про сами роды, и про первую ночь, и про отношение "
                                                                      "персонала, с которым я столкнулась в роддоме Юдина.")]),
    ("card", "IMG_9141.JPG", dict(icon="wand"), [("second", "Возможно, кому-то моя история поможет чуть лучше "
                                                                           "подготовиться к тому, что бывает после родов."),
                                                                ("script", "Первую часть уже выложила")]),
]

if __name__ == "__main__":
    src, out, MPPY, MODELS = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
    only = {int(a) for a in sys.argv[5:]}
    for i, (kind, f, kw, blocks) in enumerate(S, 1):
        if only and i not in only: continue
        name = os.path.join(out, f"story_{i:02d}")
        for old in (name + ".jpg", name + ".mp4"):
            if os.path.exists(old): os.remove(old)
        if kind == "photo": photo_story(os.path.join(src, f), blocks, name + ".jpg", **kw)
        elif kind == "card": card_story(os.path.join(src, f), blocks, name + ".jpg", **kw)
        else:
            if kw.get("voice"): kw = dict(kw, voice=(os.path.join(src, "..", "asr", kw["voice"][0]),) + kw["voice"][1:])
            video_story(os.path.join(src, f), blocks, name + ".mp4", **kw)
        print("story", i, flush=True)
