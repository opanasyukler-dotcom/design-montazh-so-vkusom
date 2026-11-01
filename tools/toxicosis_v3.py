"""@mcg.va — «Как справляться с токсикозом?», version 3: the animated montage built around her own talking video.
She stays on screen with her own sound; key words land over her head, her stickers pop in at her sides, word-by-word subtitles
from her speech run at the bottom, and each tip opens with a short card («01. ЛИМОН» … «05. ДУОНИКА») that wipes over and away.
Everything inside the Instagram Reels safe area. Fonts: Montserrat heavy / thin / thin italic; her palette.
  python3 toxicosis_v3.py BASE_VIDEO WORDS.json STICKERS_DIR CUTS_DIR MATERIALS_DIR OUT.mp4 [END_SECONDS]
"""
import json, math, os, re, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import toxicosis_v2 as V
from toxicosis_v2 import W, H, SR, CREAM, INK, BROWN, MAUVE, BLUSH, font, ease, ease_back, paste, text_img, bubble_img, shadowed, El, transform

FPS = 25
SAFE = V.SAFE                                                              # (60, 260, 950, 1480)
LEFT, RIGHT = 175, 860                                                     # sticker lanes beside her (her body is x ≈ 230…850)

def text_sh(text, st, size, col=CREAM):
    """light type over the video: a soft dark glyph shadow, no box"""
    t = text_img(text, st, size, col); a = t.getchannel("A"); pad = 24
    out = Image.new("RGBA", (t.width + 2 * pad, t.height + 2 * pad), (0, 0, 0, 0))
    sh = Image.new("RGBA", out.size, (20, 14, 10, 0)); m = Image.new("L", out.size, 0); m.paste(a, (pad, pad + 3))
    sh.putalpha(m.filter(ImageFilter.GaussianBlur(8)).point(lambda v: int(v * 0.75))); out.alpha_composite(sh); out.alpha_composite(t, (pad, pad))
    return out

def norm(w): return re.sub(r"[^\wё-]", "", w.lower())

# beats: (word key, earliest time, kind, opts). Kinds: T text over the video, IMG sticker/icon, BUB bubble, STAMP, CARD tip card,
# GLITCH, SHAKE, SFX, ROW (a row of stickers along the top)
BEATS = [
    ("как", 0, "T", dict(text="как справляться", st="L", size=52, x=540, y=400, until="токсикоз")),
    ("токсикозом", 0, "T", dict(text="С ТОКСИКОЗОМ?", st="H", size=92, x=540, y=490, until="токсикоз")),
    ("токсикозом", 0, "GLITCH", {}),
    ("отдельная", 2, "T", dict(text="(отдельная история)", st="I", size=58, x=540, y=430, until="это")),
    ("тошнит", 5, "T", dict(text="ТОШНИТ", st="H", size=130, x=540, y=470, until="хочешь")), ("тошнит", 5, "GLITCH", {}), ("тошнит", 5, "SHAKE", {}),
    ("запаха", 6, "IMG", dict(src="icon:smell", h=200, x=LEFT, y=760, until="хочешь", shadow=False)),
    ("запаха", 6, "BUB", dict(text="от любого запаха", x=740, y=640, side="r", until="хочешь")),
    ("хочешь", 8, "T", dict(text="хочешь есть,", st="L", size=58, x=540, y=400, until="поэтому")),
    ("можешь", 9, "T", dict(text="НО НЕ МОЖЕШЬ", st="H", size=84, x=540, y=490, until="поэтому")),
    ("можешь", 9, "IMG", dict(src="icon:plate", h=230, x=LEFT, y=900, until="поэтому")),
    ("можешь", 9, "IMG", dict(src="icon:cross", h=200, x=LEFT, y=900, until="поэтому", shadow=False)),
    ("чипсами", 11, "BUB", dict(text="чипсы", x=230, y=620, side="l", until="первое")),
    ("бармеладками", 12, "BUB", dict(text="кислые мармеладки", x=760, y=700, side="r", until="первое")),
    ("помогали", 13, "T", dict(text="(только они помогали)", st="I", size=54, x=540, y=430, until="первое")),
    ("первое", 14, "CARD", dict(num="01.", title="ЛИМОН", sticker="lemon")),
    ("достаточно", 15, "IMG", dict(src="icon:lemon", h=230, x=LEFT, y=820, until="также")),
    ("воду", 17, "IMG", dict(src="icon:water", h=260, x=RIGHT, y=880, until="также")),
    ("воду", 17, "IMG", dict(src="icon:slice", h=110, x=RIGHT, y=790, until="также", anim="drop")), ("воду", 17, "SFX", dict(name="splash")),
    ("всем", 18, "T", dict(text="(помогает не всем)", st="I", size=56, x=540, y=430, until="также")),
    ("боржоми", 19, "CARD", dict(num="02.", title="БОРЖОМИ", sticker="mineral")),
    ("боржоми", 20.8, "IMG", dict(src="icon:mineral", h=330, x=LEFT, y=860, until="кока")),
    ("короткий", 22, "IMG", dict(src="icon:clock", h=170, x=RIGHT, y=760, until="кока")),
    ("короткий", 22, "IMG", dict(src="icon:hands", h=170, x=RIGHT, y=760, until="кока", motion="spin", shadow=False)),
    ("короткий", 22, "T", dict(text="(ненадолго)", st="I", size=58, x=540, y=430, until="кока")),
    ("кока", 25, "CARD", dict(num="03.", title="КОКА-КОЛА", sticker="cola")),
    ("для", 26.5, "IMG", dict(src="icon:cola", h=260, x=LEFT, y=860, until="ну")),
    ("лучшим", 27, "T", dict(text="ЛУЧШИЙ ВАРИАНТ", st="H", size=80, x=540, y=470, until="волшебным")),
    ("волшебным", 30, "T", dict(text="(волшебный напиток)", st="I", size=58, x=540, y=430, until="ну")),
    ("волшебным", 30, "IMG", dict(src="icon:heart", h=150, x=RIGHT, y=760, until="ну", motion="pulse", shadow=False)),
    ("гранат", 32, "CARD", dict(num="04.", title="ГРАНАТ\nИ ЦИТРУСЫ", sticker="fruits")),
    ("минус", 33.5, "IMG", dict(src="icon:fruits", h=230, x=LEFT, y=860, until="также")),
    ("изжогу", 34, "T", dict(text="ИЗЖОГА", st="H", size=110, x=540, y=470, until="также")), ("изжогу", 34, "GLITCH", {}),
    ("изжогу", 34, "IMG", dict(src="icon:flame", h=200, x=RIGHT, y=780, until="также", motion="wiggle")),
    ("принимала", 35, "CARD", dict(num="05.", title="ДУОНИКА", sticker="blister")),
    ("если", 36.4, "IMG", dict(src="icon:blister", h=150, x=LEFT, y=820, until="также")),
    ("сильный", 36.5, "T", dict(text="(если токсикоз сильный)", st="I", size=52, x=540, y=430, until="назначает")),
    ("назначает", 40, "STAMP", dict(text="ТОЛЬКО С ВРАЧОМ", y=470, until="также")),
    ("врач", 40.5, "IMG", dict(src="icon:med", h=150, x=RIGHT, y=780, until="также", motion="pulse", shadow=False)),
    ("большими", 42, "T", dict(text="НЕ БОЛЬШИМИ ПОРЦИЯМИ", st="H", size=64, x=540, y=470, until="маленькими")),
    ("большими", 42, "IMG", dict(src="icon:plate", h=220, x=LEFT, y=880, until="вообще")),
    ("большими", 42, "IMG", dict(src="icon:cross", h=190, x=LEFT, y=880, until="вообще", shadow=False)),
    ("маленькими", 43, "T", dict(text="ПОНЕМНОГУ И ЧАЩЕ", st="H", size=72, x=540, y=470, until="вообще")),
    ("маленькими", 43, "IMG", dict(src="icon:snack", h=200, x=RIGHT, y=880, until="вообще")),
    ("единого", 45, "ROW", dict(names=("lemon", "mineral", "cola", "fruits", "blister"), y=360, h=120, until="может")),
    ("единого", 45, "T", dict(text="НЕТ ЕДИНОГО СРЕДСТВА", st="H", size=70, x=540, y=520, until="может")),
    ("может", 48, "T", dict(text="А ЧТО ПОМОГАЛО ВАМ?", st="H", size=72, x=540, y=470)),
    ("варианты", 49, "BUB", dict(text="а у меня…", x=230, y=700, side="l")),
    ("идеи", 50, "BUB", dict(text="мне помогало…", x=770, y=780, side="r")),
    ("комментариях", 50, "T", dict(text="(делитесь в комментариях)", st="I", size=50, x=540, y=600)),
]

def word_at(words, key, after):
    for w, a, b in words:
        if a >= after - 0.01 and norm(w).startswith(key): return a
    raise SystemExit(f"no «{key}» after {after}")

def card_img(o, pics):
    """a tip card: ink ground, a huge thin italic number, the heavy title, her sticker"""
    c = Image.new("RGBA", (W, H), INK + (255,))
    n = text_img(o["num"], "N", 230, BLUSH); paste(c, n, 90, 300)
    t = text_img(o["title"], "H", 104, CREAM); paste(c, t, 80, 560)
    st = pics.get("icon:" + o["sticker"], 440); paste(c, st, int(W / 2 - st.width / 2), 800)        # clear of the subtitles
    return c

def build(words, total, pics):
    els, events, shakes, glitches, cards = [], [], [], [], []
    for key, after, kind, o in BEATS:
        t = word_at(words, key, after); t1 = word_at(words, o["until"], t + 0.05) if o.get("until") else total
        if kind == "GLITCH": glitches.append(t); events.append((t, "shake")); continue
        if kind == "SHAKE": shakes.append(t); continue
        if kind == "SFX": events.append((t, o["name"])); continue
        if kind == "CARD":
            cards.append((t - 0.1, t + 0.85, card_img(o, pics))); events += [(t - 0.12, "whoosh"), (t + 0.05, "ding"), (t + 0.85, "whoosh")]; continue
        if kind == "ROW":
            n = len(o["names"])
            for i, nm in enumerate(o["names"]):
                e = El("IMG", t + 0.1 * i, t1, dict(anim="pop", motion="float", x=SAFE[0] + 80 + i * (SAFE[2] - SAFE[0] - 160) / (n - 1), y=o["y"]))
                e.img = pics.get("icon:" + nm, o["h"]); els.append(e); events.append((t + 0.1 * i, "pop"))
            continue
        el = El(kind, t, t1, dict(o, anim=o.get("anim", "pop"), motion=o.get("motion", "float" if kind == "IMG" else None)))
        if kind == "T": el.img = text_sh(o["text"], o["st"], o["size"]); el.kind = "IMG"; el.o["motion"] = None
        elif kind == "IMG": el.img = pics.get(o["src"], o["h"], None, o.get("shadow", True))
        elif kind == "BUB": el.img = bubble_img(o["text"], o["side"]); el.kind = "IMG"; el.o["motion"] = "float"
        elif kind == "STAMP": el.img = V.T.stamp_img(o["text"]); el.kind = "IMG"; el.o.update(x=540, anim="pop", motion=None); shakes.append(t)
        events.append((t, "thud" if kind == "STAMP" else "pop"))
        V.fit_safe(el); els.append(el)
    return els, events, shakes, glitches, cards

def captions(words):
    """phrases of ≤ 3 words that fit the safe width; the spoken word on a pill"""
    cf = font(54, 700); out, cur = [], []
    ws = []
    for w in words:                                                        # a bare «—» rides on the word before it
        if not re.search(r"\w", w[0]) and ws: ws[-1] = (ws[-1][0] + " " + w[0], ws[-1][1], w[2])
        else: ws.append(w)
    for w in ws:
        if cur and cf.getlength(" ".join(x[0] for x in cur + [w])) > 820: out.append(cur); cur = []
        cur.append(w)
        if len(cur) >= 3 or re.search(r"[.,!?…]$", w[0]): out.append(cur); cur = []
    if cur: out.append(cur)
    return out, cf

def render(base, words, total, pics, out_v):
    els, events, shakes, glitches, cards = build(words, total, pics)
    phrases, cf = captions(words)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", base, "-t", f"{total:.2f}", "-vf", f"scale={W}:{H}:flags=lanczos,fps={FPS}",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", out_v], stdin=subprocess.PIPE)
    n = W * H * 3
    for fi in range(int(total * FPS)):
        buf = dec.stdout.read(n)
        if len(buf) < n: break
        t = fi / FPS; fr = Image.frombuffer("RGB", (W, H), buf).convert("RGBA")
        for el in els:
            if t < el.t0 or t > el.t1 + 0.2: continue
            r = transform(el.img, t, el)
            if r: im, dx, dy = r; paste(fr, im, int(el.o["x"] + dx - im.width / 2), int(el.o["y"] + dy - im.height / 2))
        for c0, c1, img in cards:                                          # tip card wipes up over the video and away
            if c0 <= t <= c1 + 0.18:
                if t < c0 + 0.16: y = int(H * (1 - ease((t - c0) / 0.16)))
                elif t > c1: y = -int(H * ease((t - c1) / 0.18))
                else: y = 0
                paste(fr, img, 0, y)
        ph = next((p for p in phrases if p[0][1] - 0.05 <= t <= p[-1][2] + 0.2), None)
        if ph:                                                             # subtitles: the spoken word on a blush pill
            d = ImageDraw.Draw(fr); ws_ = [cf.getlength(w[0]) for w in ph]; tw = sum(ws_) + 20 * (len(ph) - 1); x = W / 2 - tw / 2; y = 1390
            for (w, a, b), wd in zip(ph, ws_):
                on = a - 0.03 <= t <= b + 0.05
                if on:
                    k = 1 + 0.1 * (1 - ease((t - a) / 0.15)); pw, phh = wd * k + 34, 80 * k
                    d.rounded_rectangle((x + wd / 2 - pw / 2, y - phh / 2, x + wd / 2 + pw / 2, y + phh / 2), radius=int(phh / 2), fill=BLUSH)
                    d.text((x + wd / 2, y), w, font=cf, fill=INK, anchor="mm")
                else:
                    d.text((x + wd / 2 + 2, y + 3), w, font=cf, fill=(20, 14, 10), anchor="mm"); d.text((x + wd / 2, y), w, font=cf, fill=CREAM, anchor="mm")
                x += wd + 20
        img = np.asarray(fr.convert("RGB"))
        g = [s for s in glitches if 0 <= t - s < 0.35]
        if g:
            k_ = 1 - (t - g[-1]) / 0.35; o_ = int(14 * k_); img = img.copy(); img[..., 0] = np.roll(img[..., 0], o_, 1); img[..., 2] = np.roll(img[..., 2], -o_, 1)
        sh = [s for s in shakes if 0 <= t - s < 0.3]
        if sh: k_ = 1 - (t - sh[-1]) / 0.3; img = np.roll(img, (int(10 * k_ * math.sin(t * 90)), int(12 * k_ * math.cos(t * 77))), (0, 1))
        enc.stdin.write(np.ascontiguousarray(img).tobytes())
    enc.stdin.close(); enc.wait(); dec.kill()
    return events

def mix(base, total, events, video, out):
    v = V.T.FX.read(base)[:int(total * SR)]; bank = V.T.sfx_bank(); N = len(v); bus = np.zeros((N, 2), np.float32)
    for t, name in events:
        x = bank[name](); i = int(t * SR); m = min(len(x), N - i)
        if m > 0 and i >= 0: bus[i:i + m] += x[:m] * 0.8
    lv = np.convolve(np.abs(v).mean(1), np.ones(960) / 960, "same"); g = np.where(lv > V.T.FX.db(-40), V.T.FX.db(-5), 1.0)
    y = v + bus * g[:, None].astype(np.float32); y *= min(1, V.T.FX.db(-1) / (np.abs(y).max() + 1e-9))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-i", video, "-map", "1:v", "-map", "0:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                   input=y.astype(np.float32).tobytes(), check=True)

if __name__ == "__main__":
    base, words_json, stickers, cuts, mats, out = sys.argv[1:7]; end = float(sys.argv[7]) if len(sys.argv) > 7 else None
    words = [tuple(w) for w in json.load(open(words_json))]
    total = end or float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", base], capture_output=True, text=True).stdout)
    tmp = out + ".video.mp4"
    events = render(base, words, total, V.Pics(cuts, mats, stickers), tmp)
    mix(base, total, events, tmp, out); os.remove(tmp)
