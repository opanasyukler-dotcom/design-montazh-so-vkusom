"""Tighten a talking-head video: keep the speech, drop pauses / false starts / fillers, and put the separately recorded voice
under the picture (the camera sound goes). Cuts sit at real silences of the voice track (energy), soft 20 ms audio edges,
and every other cut gets a gentle punch-in so the jump reads as an edit.
  python3 talk_cut.py VIDEO VOICE WORDS.json OUT.mp4 --sync T0:OFF0,T1:OFF1 [--drop A-B,A-B ...] [--gap 0.45] [--zoom 1.08]
  WORDS.json: [[word, start, end], ...] on the VOICE timeline; sync: voice time = video time + offset (two points, linear).
"""
import json, os, subprocess, sys, tempfile
import numpy as np

SR, FPS = 48000, 25

def read(path):
    return np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                                        capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()

def env_db(x, hop=480):
    m = np.abs(x).mean(1); n = len(m) // hop
    return 20 * np.log10(np.sqrt((m[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)   # 10 ms frames

def segments(words, drops, gap, e, floor):
    """speech runs on the voice timeline: words outside the dropped ranges, split at pauses longer than gap,
    each edge moved out to where the voice really starts / ends (energy), with a little air"""
    ws = [w for w in words if not any(a <= (w[1] + w[2]) / 2 <= b for a, b in drops)]
    runs, cur = [], [ws[0]]
    for w in ws[1:]:
        dropped_between = any(cur[-1][2] <= a and b <= w[1] + 0.05 for a, b in drops)
        if w[1] - cur[-1][2] > gap or dropped_between: runs.append(cur); cur = [w]
        else: cur.append(w)
    runs.append(cur); out = []
    thr = floor + 12                                                       # dB above the room floor = voice
    for r in runs:
        a, b = r[0][1], r[-1][2]; i = int(a * 100)
        while i > 0 and e[i - 1] > thr and a - (i - 1) / 100 < 0.35: i -= 1           # onset (whisper is often late)
        j = int(b * 100)
        while j < len(e) - 1 and e[j] > thr and j / 100 - b < 0.5: j += 1             # decay of the last word
        out.append([max(i / 100 - 0.08, 0), j / 100 + 0.12])
    merged = [out[0]]
    for s in out[1:]:
        if s[0] - merged[-1][1] < 0.12: merged[-1][1] = s[1]
        else: merged.append(s)
    return merged

def main(video, voice, words_json, out, sync, drops, gap=0.45, zoom=1.08):
    (t0, o0), (t1, o1) = sync; off = lambda vt: o0 + (o1 - o0) * (vt - t0) / (t1 - t0)
    v = read(voice); e = env_db(v); floor = np.percentile(e, 8)
    segs = segments(json.load(open(words_json)), drops, gap, e, floor)
    # voice-time segments → frame-exact video ranges; the voice is cut at the same (rounded) moments
    plan, parts = [], []
    for k, (a, b) in enumerate(segs):
        va = round((a - off(a)) * FPS) / FPS; vb = round((b - off(b)) * FPS) / FPS
        if vb - va < 0.2: continue
        plan.append((va, vb, zoom if k % 2 else 1.0))
        x = v[int((va + off(va)) * SR):int((va + off(va)) * SR) + int(round((vb - va) * SR))].copy()
        f = int(0.02 * SR); r = np.linspace(0, 1, f)[:, None]; x[:f] *= r; x[-f:] *= r[::-1]; parts.append(x)
    y = np.concatenate(parts); total = sum(b - a for a, b, _ in plan)
    print(f"{len(plan)} pieces, {total:.1f} s (from {len(v) / SR:.1f} s)")
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(out))) as td:   # next to the output: 4K pieces are big
        raw = os.path.join(td, "voice.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", raw], input=y.tobytes(), check=True)
        meas = subprocess.run(["ffmpeg", "-hide_banner", "-i", raw, "-af", "loudnorm=I=-15:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                              capture_output=True, text=True).stderr
        mj = json.loads(meas[meas.rindex("{"):meas.rindex("}") + 1])
        ln = (f"loudnorm=I=-15:TP=-1.5:LRA=11:linear=true:measured_I={mj['input_i']}:measured_TP={mj['input_tp']}:"
              f"measured_LRA={mj['input_lra']}:measured_thresh={mj['input_thresh']}:offset={mj['target_offset']},alimiter=limit=0.89:level=false")
        lst = os.path.join(td, "list.txt")
        with open(lst, "w") as L:
            for i, (a, b, z) in enumerate(plan):                           # each piece re-encoded frame-exact, then joined
                p = os.path.join(td, f"p{i:03d}.mp4")
                vf = f"scale=iw*{z}:ih*{z}:flags=lanczos,crop=iw/{z}:ih/{z}:(iw-iw/{z})/2:(ih-ih/{z})*0.35," if z != 1.0 else ""
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", video, "-t", f"{b - a:.3f}", "-an",
                                "-vf", vf + "format=yuv420p", "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                                "-video_track_timescale", "12800", p], check=True)
                L.write(f"file '{p}'\n")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", raw, "-map", "0:v", "-map", "1:a",
                        "-af", ln, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out], check=True)
    json.dump(plan, open(out + ".cuts.json", "w"))

if __name__ == "__main__":
    a = sys.argv; opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    sync = [tuple(float(x) for x in p.split(":")) for p in opt("--sync", "").split(",")]
    drops = [tuple(float(x) for x in p.split("-")) for p in opt("--drop", "").split(",") if p]
    main(a[1], a[2], a[3], a[4], sync, drops, float(opt("--gap", 0.45)), float(opt("--zoom", 1.08)))
