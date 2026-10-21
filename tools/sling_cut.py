"""«Обзор на слинг» — voice clean cut: drops the false start («Короче, я в полном восторге, желаю только…»),
shrinks every pause to a short breath, 8 ms crossfades at the joins. Writes voice_cut.wav + cwords.json (word times on the cut).
Work dir: v48.wav, words.json (faster-whisper word times), db.npy (10 ms loudness).
  python3 -I sling_cut.py
"""
import json, wave
import numpy as np

SR = 48000
KEEP = [(0, 108), (116, 143)]           # word index ranges
TH, MINSIL, KEEPSIL = -45, 0.14, 0.05   # quiet below TH for ≥ MINSIL → keep KEEPSIL on each side
HEAD, TAIL = 0.06, 0.35

Wd = json.load(open("words.json")); db = np.load("db.npy")
w = wave.open("v48.wav"); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768

iv = []
for a, b in KEEP:
    x0, y0 = Wd[a][0] - HEAD, Wd[b][1] + (TAIL if b == len(Wd) - 1 else 0.08)
    fr = db[int(x0 * 100):int(y0 * 100)] < TH; c = x0; i = 0; n = len(fr)
    while i < n:
        if fr[i]:
            j = i
            while j < n and fr[j]: j += 1
            if (j - i) / 100 >= MINSIL and i > 0 and j < n:
                iv.append((c, x0 + i / 100 + KEEPSIL)); c = x0 + j / 100 - KEEPSIL
            i = j
        else: i += 1
    iv.append((c, y0))

XF = int(0.008 * SR); out = np.zeros(0, np.float32); tmap = []
for s, e in iv:
    seg = x[int(s * SR):int(e * SR)].copy()
    t0 = len(out) / SR
    if len(out) and XF:
        r = np.linspace(0, 1, XF, dtype=np.float32)
        out[-XF:] = out[-XF:] * (1 - r) + seg[:XF] * r; seg = seg[XF:]; t0 -= XF / SR
        tmap.append((s, e, t0)); out = np.concatenate([out, seg])
    else:
        tmap.append((s, e, t0)); out = np.concatenate([out, seg])
with wave.open("voice_cut.wav", "wb") as o:
    o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR); o.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes())

def m(t):
    for s, e, t0 in tmap:
        if s - 0.02 <= t <= e + 0.02: return round(t0 + min(max(t, s), e) - s, 3)
    best = min(tmap, key=lambda z: min(abs(t - z[0]), abs(t - z[1]))); s, e, t0 = best
    return round(t0 + min(max(t, s), e) - s, 3)
CW = [[m(Wd[i][0]), m(Wd[i][1]), Wd[i][2]] for a, b in KEEP for i in range(a, b + 1)]
json.dump(CW, open("cwords.json", "w"), ensure_ascii=False)
print("dur", round(len(out) / SR, 2), "joins", len(iv), "removed", round(sum(e - s for s, e in iv) and (Wd[-1][1] - Wd[0][0]) - sum(e - s for s, e in iv), 2))
