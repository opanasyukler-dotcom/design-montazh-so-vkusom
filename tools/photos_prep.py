import numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from tat import clean
lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path="pose.task"), running_mode=vision.RunningMode.IMAGE))
def pose_of(bgr):
    h, w = bgr.shape[:2]
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()))
    if not r.pose_landmarks: return None
    return np.array([[q.x * w, q.y * h] for q in r.pose_landmarks[0]], np.float32)
def fit(img):  # cover-fit to 1080x1920 on black
    h, w = img.shape[:2]; s = min(1080 / w, 1920 / h); im = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((1920, 1080, 3), np.uint8); y = (1920 - im.shape[0]) // 2; x = (1080 - im.shape[1]) // 2
    out[y:y + im.shape[0], x:x + im.shape[1]] = im; return out
from tat import tattoo_mask, remove4
def cl(img, top=0.0):
    p = pose_of(img)
    if p is not None: return clean(img, p)
    r = img
    for _ in range(3):
        m = tattoo_mask(r, None); m[: int(img.shape[0] * top)] = 0; r = remove4(r, m)
    return r
def bbox_fit(img):
    g = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 60).astype(np.uint8)
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(g, 8); k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    ys, xs = np.where(lab == k)
    img = img.copy(); img[cv2.dilate((lab == k).astype(np.uint8), np.ones((15, 15), np.uint8)) == 0] = 0
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; h = (y1 - y0) * 1.12; w = h * 9 / 16
    if w < (x1 - x0) * 1.1: w = (x1 - x0) * 1.1; h = w * 16 / 9
    X0, Y0 = int(cx - w / 2), int(cy - h / 2)
    pad = cv2.copyMakeBorder(img, 2000, 2000, 2000, 2000, cv2.BORDER_CONSTANT, value=0)
    crop = pad[Y0 + 2000:Y0 + 2000 + int(h), X0 + 2000:X0 + 2000 + int(w)]
    return cv2.resize(crop, (1080, 1920), interpolation=cv2.INTER_CUBIC)
for n, kind in [("434d627b", "col"), ("4f55eeac", "col"), ("e4a87217", "col"), ("3681468f", "bz"), ("af0d1e43", "bz"), ("34b6868e", "bz")]:
    im = cv2.imread(f"dl/{n}.jpg")
    if kind == "bz":
        h, w = im.shape[:2]; im[int(h * 0.94):, int(w * 0.7):] = 0   # Bazaart watermark
        im = bbox_fit(im); im = cl(im)
    else:
        im = fit(im); im[:960] = cl(im[:960].copy(), 0.1); im[960:] = cl(im[960:].copy(), 0.1)
    cv2.imwrite(f"p_{n}.png", im); print(n, "ok")
