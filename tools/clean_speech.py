"""Tight speech edit: keep only the spoken words (every pause and «а/э/м» hesitation dropped), soft micro-crossfades
at the joins, and an optional voice change for chosen speaker intervals (the patient) so she cannot be recognised.

usage: python3 clean_speech.py SRC ASR.json OUT WINDOWS [--disguise a-b,...] [--cut a-b,...] [--drawn]
  WINDOWS: comma-separated source ranges to take words from, e.g. 4.2-33.0  or 37.5-44.8,52.3-61.9
  --disguise: source ranges spoken by the patient -> formant + slight pitch shift (rubberband, see VOICE), the surgeon stays natural
  --cut: source ranges to drop by hand (an «ага», a drawn «а-а» the ASR glued to a word)
  --drawn: also drop over-long steady vowels («ааа», «мыыы») found by voice_tools.drawn_vowels (first 0.10 s kept)
Writes OUT (video+audio) and OUT.words.json (word timings mapped to the new timeline, for the subtitles).
"""
import json, os, subprocess, sys
import numpy as np

SRC, ASRJ, OUT, WIN = sys.argv[1:5]


def ranges(flag):
    if flag not in sys.argv: return []
    return [tuple(map(float, r.split("-"))) for r in sys.argv[sys.argv.index(flag) + 1].split(",") if r]


DIS, CUT = ranges("--disguise"), ranges("--cut")
# disguise: formants up 16 % (clearly another timbre), pitch up 11 % -> a different, lighter female voice, still
# natural (no low «monster» of a pitch drop); the second pass brings the pitch back down with the formants kept
VOICE = ("rubberband=pitch=1.16:formant=shifted:transients=smooth,"
         "rubberband=pitch=0.957:formant=preserved:transients=smooth")
HAND = list(CUT)                                          # words under these go from the subtitles too
WINS = [tuple(map(float, r.split("-"))) for r in WIN.split(",")]
FPS, SR = 30.0, 48000
PRE, POST, MERGE = 0.06, 0.14, 0.12          # kept around each word; joins closer than MERGE are not cut
FILLERS = {"э", "ээ", "эм", "м", "мм", "ммм", "а-а", "э-э", "хм", "эээ"}

words = []
for s in json.load(open(ASRJ)):
    for a, b, t in s["w"]:
        tt = t.strip().lower().strip(".,!?…")
        if tt in FILLERS: continue
        if any(a >= w0 - 0.02 and b <= w1 + 0.05 for w0, w1 in WINS): words.append([a, b, t.strip()])
segs = []
for a, b, t in words:
    a, b = a - PRE, b + POST
    if segs and a - segs[-1][1] < MERGE: segs[-1][1] = max(segs[-1][1], b)
    else: segs.append([a, b])
# second pass on the sound itself: ASR word spans often stretch over breaths and silences — cut every quiet run
# longer than 0.16 s inside the kept pieces (50 ms of air left on each side)
_pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", SRC, "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                      capture_output=True, check=True).stdout
_x = np.frombuffer(_pcm, np.float32); hop = 160                                   # 10 ms frames
_rms = np.sqrt(np.convolve(_x ** 2, np.ones(400) / 400, "same")[::hop] + 1e-12)
_db = 20 * np.log10(_rms / (np.percentile(_rms, 99) + 1e-9))
quiet = _db < np.percentile(_db, 10) + 7                                # adaptive: room noise floor + 7 dB
for a, b, t in words:                                     # never cut inside a recognised word (quiet speakers)
    c0, c1 = a + 0.2 * (b - a), b - 0.2 * (b - a)
    quiet[int(c0 * 100):int(c1 * 100) + 1] = False
refined = []
for a, b in segs:
    i0, i1 = int(a * 100), int(b * 100); cur = a; j = i0
    while j < i1:
        if quiet[j]:
            k2 = j
            while k2 < i1 and quiet[k2]: k2 += 1
            if (k2 - j) >= 16 and j > i0 and k2 < i1:
                refined.append([cur, j / 100 + 0.05]); cur = k2 / 100 - 0.05
            j = k2
        else: j += 1
    refined.append([cur, b])
if "--drawn" in sys.argv:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import voice_tools as vt
    lm, _, en = vt.features(_x)
    dv = []
    for a, b in vt.drawn_vowels(lm, en, min_len=0.22, keep=0.10):      # one stretched vowel split in two -> one cut
        if dv and a - dv[-1][1] < 0.3: dv[-1][1] = b
        else: dv.append([a, b])
    CUT += [tuple(r) for r in dv]
for c0, c1 in CUT:                                        # subtract the dropped ranges from the kept pieces
    nxt = []
    for a, b in refined:
        if c1 <= a or c0 >= b: nxt.append([a, b]); continue
        if c0 > a: nxt.append([a, c0])
        if c1 < b: nxt.append([c1, b])
    refined = nxt
print("cut:", " ".join(f"{a:.2f}-{b:.2f}" for a, b in CUT))
segs = [s_ for s_ in refined if s_[1] - s_[0] > 0.08]
# snap to frames so picture and sound stay in sync
segs = [[round(a * FPS) / FPS, round(b * FPS) / FPS] for a, b in segs]
segs = [[max(a, 0), b] for a, b in segs if b > a]

# audio: natural track + disguised track, pick per word
tmp = OUT + ".tmp"; os.makedirs(tmp, exist_ok=True)
nat, dis = os.path.join(tmp, "nat.f32"), os.path.join(tmp, "dis.f32")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", nat], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-vn", "-ac", "1", "-ar", str(SR),
                "-af", VOICE, "-f", "f32le", dis], check=True)
A, D = np.fromfile(nat, np.float32), np.fromfile(dis, np.float32)
n = min(len(A), len(D)); A, D = A[:n], D[:n]
mask = np.zeros(n, np.float32)
for a, b in DIS: mask[int(a * SR):int(b * SR)] = 1
k = int(0.02 * SR); mask = np.convolve(mask, np.ones(k) / k, "same")         # 20 ms blend at speaker changes
mix = A * (1 - mask) + D * mask
X = int(0.012 * SR); out = []
for a, b in segs:
    piece = mix[int(a * SR):int(b * SR)].copy()
    if len(piece) > 2 * X:
        piece[:X] *= np.linspace(0, 1, X); piece[-X:] *= np.linspace(1, 0, X)    # 12 ms fades: no clicks
    out.append(piece)
audio = np.concatenate(out); audio /= max(1.0, np.abs(audio).max() / 0.97)
wav = os.path.join(tmp, "a.f32"); audio.astype(np.float32).tofile(wav)

v = "".join(f"[0:v]trim={a:.4f}:{b:.4f},setpts=PTS-STARTPTS[v{i}];" for i, (a, b) in enumerate(segs))
v += "".join(f"[v{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[v]"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", wav,
                "-filter_complex", v + ";[1:a]loudnorm=I=-16:TP=-1.5[a]", "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-crf", "15", "-preset", "fast", "-pix_fmt", "yuv420p", "-r", "30",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", OUT], check=True)
# words on the new timeline
offs, off = [], 0.0
for a0, b0 in segs: offs.append(off); off += b0 - a0
mapped = []
for a, b, t in words:                                     # a word may span a removed silence: place it where it sounds
    if b - a > 0.02 and sum(max(0, min(b, c1) - max(a, c0)) for c0, c1 in HAND) >= 0.8 * (b - a): continue
    best = max(range(len(segs)), key=lambda i: min(b, segs[i][1]) - max(a, segs[i][0]))
    (a0, b0), o = segs[best], offs[best]
    if min(b, b0) - max(a, a0) > -0.1: mapped.append([max(a, a0) - a0 + o, max(min(b, b0), max(a, a0) + 0.05) - a0 + o, t])
json.dump([{"w": mapped}], open(OUT + ".words.json", "w"), ensure_ascii=False)
subprocess.run(["rm", "-rf", tmp])
print("wrote", OUT, f"{off:.1f}s", len(segs), "pieces")
