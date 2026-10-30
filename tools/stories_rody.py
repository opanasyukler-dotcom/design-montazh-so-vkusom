"""@mcg.va — 10 stories «про мои роды» (1080x1920): photo stories → JPG, video stories → MP4 with the text over the clip.
Type: main = Montserrat ExtraBold, huge; second = thin Montserrat 300; script accents = Gogol handwriting.
White type with a soft glyph shadow over a gentle gradient spray from the text side (no boxes, no cards).
  python3 -I stories_rody.py MATERIALS_DIR OUT_DIR [N ...]
"""
import os, sys, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories_kit import W, H, SAFE_TOP, SAFE_BOTTOM, FONT_DIR, wrap

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SCRIPT = os.path.join(ROOT, "skills", "montage-svg-typography", "Шрифты", "Коллекция шрифтов", "Gogol-Regular.otf")
MONT = os.path.join(FONT_DIR, "Montserrat-VariableFont_wght.ttf")
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
X0, MAXW = 84, W - 2 * 84
SPRAY = (22, 15, 12)

def mont(size, wght):
    f = ImageFont.truetype(MONT, size); f.set_variation_by_axes([wght]); return f

STYLE = {"main": (lambda s: mont(s, 800), 92, 1.08, 30),            # font, size, line lead (× size), gap after the block
         "second": (lambda s: mont(s, 300), 40, 1.38, 26),
         "script": (lambda s: ImageFont.truetype(SCRIPT, s), 108, 0.98, 48)}

def emoji_img(ch, h):
    f = ImageFont.truetype(EMOJI, 109); im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((80, 80), ch, font=f, embedded_color=True, anchor="mm"); im = im.crop(im.getbbox())
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)

def split_emoji(t):
    """«text 😂» → ("text", "😂"): emoji only at the end of a block"""
    if t and ord(t[-1]) > 0x2600: return t[:-1].rstrip(), t[-1]
    return t, None

def layout(blocks, scale=1.0):
    """→ list of (kind, font, line, y-offset, lead), total height"""
    ops, y = [], 0
    for i, (kind, text) in enumerate(blocks):
        mk, size, lead, gap = STYLE[kind]; f = mk(round(size * scale)); text, emo = split_emoji(text)
        for j, l in enumerate(wrap(text, f, MAXW - (90 if emo else 0))):
            ops.append((kind, f, l, y, emo if emo and j == len(wrap(text, f, MAXW - 90)) - 1 else None)); y += f.size * lead
        if i < len(blocks) - 1: y += gap * scale
    return ops, y

def fit_text(blocks):
    for sc in np.arange(1.0, 0.6, -0.03):                                 # shrink only if the text would leave the safe zone
        ops, h = layout(blocks, sc)
        if h <= SAFE_BOTTOM - SAFE_TOP - 40: return ops, h

def geometry(blocks, pos):
    """text block top y0 and height h; the sharp picture zone (z0, z1) is the rest of the frame, overlapping the text by FEATHER"""
    ops, h = fit_text(blocks); h = int(round(h))
    y0 = SAFE_TOP + 30 if pos == "top" else SAFE_BOTTOM - 20 - h
    z = (y0 + h + 40 - FEATHER // 2, H) if pos == "top" else (0, y0 - 40 + FEATHER // 2)
    return ops, y0, h, (int(z[0]), int(z[1]))

FEATHER = 300

def zone_mask(pos, zh):
    """alpha of the sharp picture: 1 inside, fading out over FEATHER towards the text"""
    m = np.ones(zh, np.float32); r = np.linspace(0, 1, FEATHER) ** 1.6
    if pos == "top": m[:FEATHER] = r
    else: m[-FEATHER:] = r[::-1]
    return Image.fromarray((np.repeat(m[:, None], W, 1) * 255).astype(np.uint8))

def text_layer(ops, y0, h, pos):
    """RGBA layer: a light spray on the text side + white type with a soft shadow"""
    a = np.zeros(H, np.float32); yy = np.arange(H)
    if pos == "top": e0, e1 = y0 + h + 200, y0 + 100; a = np.clip((e0 - yy) / (e0 - e1), 0, 1)
    else: e0, e1 = y0 - 200, y0 + 100; a = np.clip((yy - e0) / (e1 - e0), 0, 1)
    a = (a ** 1.3) * 0.30
    lay = Image.new("RGBA", (W, H), SPRAY + (0,)); lay.putalpha(Image.fromarray((np.repeat(a[:, None], W, 1) * 255).astype(np.uint8)))
    txt = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sh = Image.new("L", (W, H), 0)
    d, ds = ImageDraw.Draw(txt), ImageDraw.Draw(sh)
    for kind, f, l, y, emo in ops:
        yb = y0 + y + f.size * 0.8                                         # baseline
        d.text((X0, yb), l, font=f, fill=(255, 255, 255, 255), anchor="ls")
        ds.text((X0, yb + 3), l, font=f, fill=150 if kind != "main" else 120, anchor="ls")
        if emo:
            e = emoji_img(emo, int(f.size * (0.8 if kind == "script" else 0.95)))
            txt.alpha_composite(e, (int(X0 + f.getlength(l) + 18), int(yb - e.height * 0.92)))
    shadow = Image.new("RGBA", (W, H), SPRAY + (0,)); shadow.putalpha(sh.filter(ImageFilter.GaussianBlur(8)))
    lay.alpha_composite(shadow); lay.alpha_composite(txt)
    return lay

def place(sw, sh, zh, fx, fy, zoom=1.0):
    """scale (k) and crop offset of a sw×sh picture so it covers a W×zh zone with the focus (fx, fy) near the zone centre"""
    k = max(W / sw, zh / sh) * zoom; pw, ph = sw * k, sh * k
    x = int(min(max(fx * pw - W / 2, 0), pw - W)); y = int(min(max(fy * ph - zh / 2, 0), ph - zh))
    return k, round(pw), round(ph), x, y

def backdrop(im):
    """the same picture, cover-filled, heavily blurred and a little darker: the calm ground for the type"""
    s = max(W / im.width, H / im.height); b = im.resize((round(im.width * s / 8), round(im.height * s / 8)), Image.BILINEAR)
    b = b.filter(ImageFilter.GaussianBlur(5)).resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
    b = b.crop(((b.width - W) // 2, (b.height - H) // 2, (b.width - W) // 2 + W, (b.height - H) // 2 + H))
    return Image.fromarray((np.asarray(b, np.float32) * 0.72).astype(np.uint8))

def photo_story(src, blocks, pos, out, fx=0.5, fy=0.4, zoom=1.0):
    import pillow_heif; pillow_heif.register_heif_opener()
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    ops, y0, h, (z0, z1) = geometry(blocks, pos); zh = z1 - z0
    k, pw, ph, x, y = place(im.width, im.height, zh, fx, fy, zoom)
    sharp = im.resize((pw, ph), Image.LANCZOS).crop((x, y, x + W, y + zh))
    img = backdrop(im); img.paste(sharp, (0, z0), zone_mask(pos, zh)); img = img.convert("RGBA")
    img.alpha_composite(text_layer(ops, y0, h, pos)); img.convert("RGB").save(out, quality=95, subsampling=0)

def video_story(src, blocks, pos, out, ss=0.0, fx=0.5, fy=0.4, zoom=1.0):
    """the clip (muted) sharp in its zone over its own blurred copy; the static type fades in; length = time to read it"""
    words = sum(len(t.split()) for _, t in blocks); dur = float(np.clip(words / 2.6 + 2.5, 7, 22))
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src], capture_output=True, text=True)
    clip = float(pr.stdout) - ss
    fr = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(ss), "-i", src, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                        capture_output=True, check=True).stdout
    import io; sw, sh = Image.open(io.BytesIO(fr)).size                     # displayed (rotated) size
    slow = min(max(dur / clip, 1.0), 1.6)                                   # a short clip plays a bit slower, then loops if still short
    ops, y0, h, (z0, z1) = geometry(blocks, pos); zh = z1 - z0
    k, pw, ph, x, y = place(sw, sh, zh, fx, fy, zoom); pw += pw % 2; ph += ph % 2
    lay, msk = out + ".layer.png", out + ".mask.png"
    text_layer(ops, y0, h, pos).save(lay); zone_mask(pos, zh).save(msk)
    vf = (f"[0:v]setpts={slow:.3f}*PTS,fps=30,split[a][b];"
          f"[b]scale={W // 8}:{H // 8}:force_original_aspect_ratio=increase,crop={W // 8}:{H // 8},gblur=sigma=5,"
          f"scale={W}:{H}:flags=bicubic,colorchannelmixer=rr=.72:gg=.72:bb=.72[bg];"
          f"[a]scale={pw}:{ph}:flags=lanczos,crop={W}:{zh}:{x}:{y},format=rgba[f];[2:v]format=gray[m];[f][m]alphamerge[fm];"
          f"[bg][fm]overlay=0:{z0}[c];[1:v]format=rgba,fade=in:st=0.25:d=0.6:alpha=1[t];[c][t]overlay=0:0,format=yuv420p")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(ss), "-stream_loop", "-1", "-i", src, "-loop", "1", "-i", lay,
                    "-loop", "1", "-i", msk, "-filter_complex", vf, "-t", f"{dur:.2f}", "-an", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "17", "-profile:v", "high", "-movflags", "+faststart", out], check=True)
    os.remove(lay); os.remove(msk)

STORIES = {
    1: ("photo", "IMG_9141.JPG", "top", dict(fy=0.45, zoom=1.35), [
        ("main", "Я тут наконец-то решилась рассказать"),
        ("script", "про свои роды"),
        ("second", "Если честно, очень долго откладывала эту тему, потому что вообще не понимала, хочу ли вообще рассказывать.")]),
    2: ("video", "IMG_0751.MOV", "bottom", dict(ss=2.0), [
        ("second", "Всю беременность я думала, что самое страшное, что меня ждёт, — это сами роды и тяжесть материнства, но…"),
        ("second", "Я готовилась к боли, к тому, что всё может пойти не по плану, представляла себе миллион разных сценариев."),
        ("main", "Но реальность оказалась"),
        ("script", "немного другой")]),
    3: ("video", "IMG_8216.mov", "bottom", dict(), [
        ("main", "Мои роды начались ровно в ПДР!"),
        ("second", "Муж в этот день уехал на выступление, а я осталась дома со схватками."),
        ("second", "Причём весь день я лежала, смотрела кино, ела, слушала музыку и отдыхала в полной темноте."),
        ("script", "Вот примерно о таких родах я и мечтала")]),
    4: ("video", "IMG_8217.mov", "top", dict(), [
        ("second", "А потом у меня отошли воды, схватки стали сильнее,"),
        ("main", "и знаете, что я решила сделать?"),
        ("script", "Помыть полы и высушить голову после душа 😂"),
        ("second", "Не спрашивайте, зачем. У человека активные роды, а она уборку затеяла. Я расстраивалась и думала, что у меня "
                   "максимум 5 см раскрытие (но на самом деле уже было 10 см)."),
        ("second", "Кстати, голову до конца я так и не высушила.")]),
    "5a": ("video", "IMG_6146.mov", "top", dict(fy=0.5), [
        ("main", "В итоге в роддом мы буквально летели"),
        ("second", "Потому что я чуть не родила дома, а затем и в такси. Просто бедный таксист. Он сидел весь зелёный от страха. "
                   "Меня просто забрала моя акушерка из дома, и мы поехали в роддом. Мы не могли долго выехать со двора, и акушерка "
                   "думала уже звонить в полицию, чтобы нас сопроводили. Потому что у меня уже начинались потуги.")]),
    "5b": ("photo", "IMG_6190.heic", "bottom", dict(fy=0.40), [
        ("second", "Меня встретили с каталкой, быстро отвезли в родовую, и уже через час после поступления я родила дочку."),
        ("second", "И знаете что? Я считаю, что роды у меня прошли"),
        ("script", "просто шикарно!"),
        ("second", "После такого опыта я реально готова идти за вторым.")]),
    6: ("video", "IMG_6145.mov", "top", dict(), [
        ("second", "Но потом меня перевели в послеродовое отделение."),
        ("main", "И вот здесь началось то, к чему я вообще не была готова"),
        ("second", "Я осталась одна с новорождённым ребёнком. Уставшая, без сна, без понимания, что делать дальше."),
        ("script", "И мне казалось, что хотя бы здесь мне помогут")]),
    "7a": ("video", "IMG_8214.mov", "bottom", dict(fy=0.35), [
        ("main", "Дочка всю ночь плакала,"),
        ("second", "я буквально падала с ног. Она захлёбывалась водой. Я не понимала, что с ней происходит. Оказывается, ребёнок "
                   "может наглотаться околоплодных вод, и это нормально, что вода потом выходит. Почему-то в роддоме мне никто "
                   "не мог этого объяснить."),
        ("script", "Я чуть не умерла со страха")]),
    "7b": ("photo", "IMG_6225.heic", "top", dict(fy=0.55), [
        ("second", "Несколько раз звала медсестру, потому что не понимала, что происходит и как ей помочь. И вместо объяснений услышала:"),
        ("main", "«Привыкай, дальше будет хуже»"),
        ("script", "Я до сих пор помню эту фразу")]),
    8: ("photo", "IMG_0042.HEIC", "top", dict(fy=0.5), [
        ("main", "И это была только первая ночь"),
        ("second", "Дальше были оскорбления из-за моего отказа от грудного вскармливания, противоречивые замечания врачей и постоянное "
                   "ощущение, что я всё делаю неправильно."),
        ("second", "Хотя я только что родила первого ребёнка"),
        ("script", "и просто пыталась разобраться")]),
    "9a": ("photo", "DSC00516.JPG", "bottom", dict(fy=0.30), [
        ("second", "Самое обидное, что я ведь не ждала какого-то особенного отношения. Мне просто хотелось, чтобы мне объяснили, "
                   "помогли, дали немного восстановиться после родов."),
        ("main", "Хотелось просто человеческого отношения,"),
        ("second", "потому что после родов ты настолько сильно уязвима, как никогда.")]),
    "9b": ("photo", "DSC00770.JPG", "top", dict(fy=0.55), [
        ("second", "Да, кому-то могло быть хуже. Но я здесь не сравниваю свою историю с чужой."),
        ("script", "Для меня этот опыт оказался травмирующим,"),
        ("second", "и я потом ещё долго с этим разбиралась.")]),
    10: ("photo", "100.jpg", "bottom", dict(fy=0.33), [
        ("main", "Я решила сделать целую серию рилс,"),
        ("second", "потому что в одно видео всё это просто невозможно уместить. Расскажу и про сами роды, и про первую ночь, и про "
                   "отношение персонала, с которым я столкнулась в роддоме Юдина."),
        ("second", "Возможно, кому-то моя история поможет чуть лучше подготовиться к тому, что бывает после родов."),
        ("script", "Первую часть уже выложила")]),
}

if __name__ == "__main__":
    src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True); order = list(STORIES)
    for key in [k for k in order if str(k) in sys.argv[3:]] or order:
        kind, f, pos, kw, blocks = STORIES[key]; name = os.path.join(out, f"story_{order.index(key) + 1:02d}")
        if kind == "photo": photo_story(os.path.join(src, f), blocks, pos, name + ".jpg", **kw)
        else: video_story(os.path.join(src, f), blocks, pos, name + ".mp4", **kw)
        print("story", key, "→", os.path.basename(name), flush=True)
