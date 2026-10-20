"""Blur the patient's bare skin in an OR video (orange/tan skin in the lower part of the frame), keep everything else.

usage: python3 body_blur.py SRC OUT a b
Large skin-coloured areas below y0 are feathered and heavily blurred; small orange things (the fat in the tubing,
the surgeon's gloves) are left sharp.
"""
import subprocess, sys, json
import numpy as np, cv2

SRC, OUT, A, B = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                    "-of", "json", SRC], capture_output=True, text=True)
s = json.loads(r.stdout)["streams"][0]; w, h = s["width"], s["height"]; n, d = map(int, s["r_frame_rate"].split("/")); fps = n / d
y0 = int(h * 0.5)
dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(A), "-t", str(B - A), "-i", SRC, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       stdout=subprocess.PIPE)
tmp = OUT + ".v.mp4"
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
                        "-c:v", "libx264", "-crf", "14", "-preset", "fast", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
prev = None
while True:
    buf = dec.stdout.read(w * h * 3)
    if len(buf) < w * h * 3: break
    fr = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 4) & (hsv[..., 0] <= 22) & (hsv[..., 1] > 70) & (hsv[..., 2] > 60)).astype(np.uint8)
    m[:y0] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, k):
        if st[i, cv2.CC_STAT_AREA] > 12000: keep[lab == i] = 1
    keep = cv2.dilate(keep, np.ones((31, 31), np.uint8)).astype(np.float32)
    keep = cv2.GaussianBlur(keep, (0, 0), 14)
    prev = keep if prev is None else 0.6 * prev + 0.4 * keep                # steady from frame to frame
    bl = cv2.GaussianBlur(cv2.resize(cv2.resize(fr, (w // 16, h // 16), interpolation=cv2.INTER_AREA), (w, h)), (0, 0), 10)
    a = np.clip(prev * 1.3, 0, 1)[..., None]
    enc.stdin.write((fr * (1 - a) + bl * a).astype(np.uint8).tobytes())
enc.stdin.close(); enc.wait()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", str(A), "-t", str(B - A), "-i", SRC, "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", OUT], check=True)
subprocess.run(["rm", "-f", tmp]); print("wrote", OUT)
