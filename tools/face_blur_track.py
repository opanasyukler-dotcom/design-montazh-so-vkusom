"""Cut segments from a phone video, blur ONE tracked face (the patient) and write a clean intermediate clip.

usage: python3 face_blur_track.py MODEL.onnx SRC OUT pick a0-b0 [a1-b1 ...]
  pick: "top" = start from the highest face in the first frame (standing patient next to a seated doctor),
        "big" = start from the largest face.
Segments are joined in order; audio is cut the same way.  Missed detections hold the last box.
"""
import subprocess, sys, json
import numpy as np, cv2

MODEL, SRC, OUT, PICK = sys.argv[1:5]
SEGS = [tuple(map(float, s.split("-"))) for s in sys.argv[5:]]


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "json", path], capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]; n, d = map(int, s["r_frame_rate"].split("/"))
    return s["width"], s["height"], n / d


w, h, fps = probe(SRC)
det = cv2.FaceDetectorYN.create(MODEL, "", (w, h), 0.6, 0.3, 50)
tmp_v = OUT + ".v.mp4"
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
                        "-i", "-", "-c:v", "libx264", "-crf", "14", "-preset", "fast", "-pix_fmt", "yuv420p", tmp_v], stdin=subprocess.PIPE)
prev = None; misses = 0
for a, b in SEGS:
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a), "-i", SRC, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE)
    while True:
        buf = dec.stdout.read(w * h * 3)
        if len(buf) < w * h * 3: break
        fr = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        _, faces = det.detect(fr)
        boxes = [] if faces is None else [f[:4] for f in faces]
        if boxes:
            if prev is None:
                prev = min(boxes, key=lambda bx: bx[1]) if PICK == "top" else max(boxes, key=lambda bx: bx[2] * bx[3])
            else:
                c = np.array([prev[0] + prev[2] / 2, prev[1] + prev[3] / 2])
                best = min(boxes, key=lambda bx: np.linalg.norm(np.array([bx[0] + bx[2] / 2, bx[1] + bx[3] / 2]) - c))
                dist = np.linalg.norm(np.array([best[0] + best[2] / 2, best[1] + best[3] / 2]) - c)
                if dist < max(prev[2], prev[3]) * 1.6 or misses > 20:
                    prev = 0.5 * np.array(prev) + 0.5 * np.array(best); misses = 0
                else:
                    misses += 1
        else:
            misses += 1
        if prev is not None:
            x, y, bw, bh = prev
            cx, cy = x + bw / 2, y + bh / 2.2
            ax, ay = int(bw * 1.05), int(bh * 1.25)                      # generous ellipse: forehead to chin, ear to ear
            x0, y0, x1, y1 = max(int(cx - ax), 0), max(int(cy - ay), 0), min(int(cx + ax), w), min(int(cy + ay), h)
            if x1 > x0 and y1 > y0:
                roi = fr[y0:y1, x0:x1]
                small = cv2.resize(roi, (6, 8), interpolation=cv2.INTER_AREA)          # ~6x8 cells whatever the face size
                bl = cv2.GaussianBlur(cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR), (0, 0), max(8, (x1 - x0) / 12))
                m = np.zeros((y1 - y0, x1 - x0), np.float32)
                cv2.ellipse(m, (int(cx - x0), int(cy - y0)), (ax, ay), 0, 0, 360, 1, -1)
                m = cv2.GaussianBlur(m, (0, 0), 6)[..., None]
                fr[y0:y1, x0:x1] = (roi * (1 - m) + bl * m).astype(np.uint8)
        enc.stdin.write(fr.tobytes())
    dec.stdout.close(); dec.wait()
enc.stdin.close(); enc.wait()
# audio cut the same way
parts = "".join(f"[0:a]atrim={a}:{b},asetpts=PTS-STARTPTS[a{i}];" for i, (a, b) in enumerate(SEGS))
parts += "".join(f"[a{i}]" for i in range(len(SEGS))) + f"concat=n={len(SEGS)}:v=0:a=1[a]"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp_v, "-i", SRC, "-filter_complex", parts.replace("[0:a]", "[1:a]"),
                "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", OUT], check=True)
subprocess.run(["rm", "-f", tmp_v]); print("wrote", OUT)
