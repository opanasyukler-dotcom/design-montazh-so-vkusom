import cv2, json, numpy as np, subprocess, sys
src, trk, out = sys.argv[1:4]; T = json.load(open(trk)); prev = None
cap = cv2.VideoCapture(src); W_, H_ = int(cap.get(3)), int(cap.get(4))
only = [float(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else None
def blur(f, b):
    cx, cy, w, h = b; rx, ry = 1.0 * w, 1.05 * h
    x0, y0, x1, y1 = int(max(0, cx - rx * 1.6)), int(max(0, cy - ry * 1.6)), int(min(W_, cx + rx * 1.6)), int(min(H_, cy + ry * 1.6))
    if x1 <= x0 or y1 <= y0: return f
    reg = f[y0:y1, x0:x1].astype(np.float32)
    sm = cv2.resize(reg, (max(1, (x1 - x0) // 40), max(1, (y1 - y0) // 40)), interpolation=cv2.INTER_AREA)
    bl = cv2.GaussianBlur(cv2.resize(sm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC), (0, 0), max(8, w * 0.14))
    m = np.zeros((y1 - y0, x1 - x0), np.float32); cv2.ellipse(m, (int(cx - x0), int(cy - y0)), (int(rx), int(ry)), 0, 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(6, w * 0.12)); m = np.clip(m * 2.0, 0, 1)
    ya, yb = max(y0, 190), min(y1, 410)                                   # keep the burned-in captions sharp (glyph-sized white blobs in the caption row only)
    if yb > ya:
        hsv = cv2.cvtColor(f[ya:yb, x0:x1], cv2.COLOR_BGR2HSV); raw = ((hsv[..., 2] > 215) & (hsv[..., 1] < 35)).astype(np.uint8)
        n_, lab, st, _ = cv2.connectedComponentsWithStats(raw, 8); tm = np.zeros_like(raw)
        cand = [k for k in range(1, n_) if 252 <= st[k][1] + ya and st[k][1] + st[k][3] + ya <= 348 and st[k][3] < 62 and st[k][2] < 64 and st[k][4] > 12]
        if len(cand) >= 3:                                                # letters share a baseline or an x-height line; stray glints off the glasses don't
            tops = np.array([st[k][1] for k in cand]); bots = np.array([st[k][1] + st[k][3] for k in cand])
            mt, mb = np.median(tops), np.median(bots)
            keep = sorted([k for k, t_, b_ in zip(cand, tops, bots) if abs(t_ - mt) <= 4 or abs(b_ - mb) <= 4], key=lambda k: st[k][0])
            groups = []                                                   # a word is one tight run of letters; keep the biggest run only
            for k in keep:
                if groups and st[k][0] - (st[groups[-1][-1]][0] + st[groups[-1][-1]][2]) < 30: groups[-1].append(k)
                else: groups.append([k])
            keep = max(groups, key=len) if groups else []
            if len(keep) >= 2:
                for k in keep: tm[lab == k] = 1
        tm = cv2.GaussianBlur(cv2.dilate(tm, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 1.2); m[ya - y0:yb - y0] *= 1 - np.clip(tm * 1.5, 0, 1)
    m = m[..., None]
    f[y0:y1, x0:x1] = (reg * (1 - m) + bl * m).astype(np.uint8); return f
if only:
    tiles = []
    for t in only:
        i = int(round(t * 30)); cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if T[i]: f = blur(f, T[i])
        f = cv2.resize(f, (270, 480)); cv2.putText(f, f"{t:.1f}", (8, 30), 0, 1, (0, 255, 255), 2); tiles.append(f)
    while len(tiles) % 8: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])); sys.exit()
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W_}x{H_}', '-r', '30', '-i', '-', '-i', src,
                      '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-crf', '17', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-shortest', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i < len(T) and T[i]: f = blur(f, T[i])
    p.stdin.write(f.tobytes()); i += 1
p.stdin.close(); p.wait(); print(i)
