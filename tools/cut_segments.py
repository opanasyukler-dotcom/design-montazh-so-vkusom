"""Cut and join segments of a video (video + audio, re-encoded, frame-accurate).

usage: python3 cut_segments.py SRC OUT a0-b0 [a1-b1 ...]
       python3 cut_segments.py SRC OUT --nopauses ASR.json [gap]   # drop every pause longer than gap (default 0.15 s)
In --nopauses mode the cuts are made inside the pauses (0.1 s kept before a word, 0.15 s after), so no word is clipped.
"""
import json, subprocess, sys

SRC, OUT = sys.argv[1], sys.argv[2]
if sys.argv[3] == "--nopauses":
    gap = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
    ws = [(w[0], w[1]) for s in json.load(open(sys.argv[4])) for w in s["w"]]
    segs = []
    for a, b in ws:
        a, b = max(a - 0.1, 0), b + 0.15
        if segs and a - segs[-1][1] < gap: segs[-1][1] = max(segs[-1][1], b)
        else: segs.append([a, b])
else:
    segs = [list(map(float, s.split("-"))) for s in sys.argv[3:]]
v = "".join(f"[0:v]trim={a}:{b},setpts=PTS-STARTPTS[v{i}];[0:a]atrim={a}:{b},asetpts=PTS-STARTPTS,afade=t=in:d=0.03,"
            f"afade=t=out:st={b - a - 0.04:.3f}:d=0.04[a{i}];" for i, (a, b) in enumerate(segs))
v += "".join(f"[v{i}][a{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=1[v][a]"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-filter_complex", v, "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-crf", "15", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", OUT], check=True)
print("wrote", OUT, "segments:", [(round(a, 2), round(b, 2)) for a, b in segs])
