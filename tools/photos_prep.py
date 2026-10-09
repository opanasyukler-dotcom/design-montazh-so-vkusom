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
import tat3; tat3.PROT[:] = [4.7, 2.3, -0.5]
from tat3 import ink_mask, remove_ink
from tat4 import dense
from tatblur import tattoo_region
from tat import nconv
def blur_mask(img, R, skin, sigma=16):
    f = img.astype(np.float32); bl, _ = nconv(f, skin.astype(np.float32), sigma)
    a = cv2.GaussianBlur(R.astype(np.float32), (0, 0), 6)[..., None] * skin[..., None]
    return np.clip(f * (1 - a) + bl * a, 0, 255).astype(np.uint8)
def cl(img, top=0.0, bottom=1.0, extra=None):
    p = pose_of(img)
    R, skin = tattoo_region(img, p)
    L0 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    R[cv2.dilate((L0 > 225).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0] = 0
    if p is None: R[: int(img.shape[0] * top)] = 0; R[int(img.shape[0] * bottom):] = 0
    if extra is not None: R |= extra & skin
    return blur_mask(img, R, skin)
def bbox_fit(img):
    g = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 60).astype(np.uint8)
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(g, 8); k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    ys, xs = np.where(lab == k)
    body = (lab == k).astype(np.uint8); cs, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body = np.zeros_like(body); cv2.drawContours(body, cs, -1, 1, -1)
    img = img.copy(); img[cv2.dilate(body, np.ones((15, 15), np.uint8)) == 0] = 0
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; h = (y1 - y0) * 1.12; w = h * 9 / 16
    if w < (x1 - x0) * 1.1: w = (x1 - x0) * 1.1; h = w * 16 / 9
    X0, Y0 = int(cx - w / 2), int(cy - h / 2)
    pad = cv2.copyMakeBorder(img, 2000, 2000, 2000, 2000, cv2.BORDER_CONSTANT, value=0)
    crop = pad[Y0 + 2000:Y0 + 2000 + int(h), X0 + 2000:X0 + 2000 + int(w)]
    return cv2.resize(crop, (1080, 1920), interpolation=cv2.INTER_CUBIC)
Z = {"3681468f": [("r", 860, 680, 1060, 1330), ("r", 40, 920, 150, 1320), ("c", 370, 1380, 45), ("c", 790, 815, 40)],
     "34b6868e": [("c", 490, 1040, 70)]}
for n, kind in [("434d627b", "col"), ("4f55eeac", "col"), ("e4a87217", "col"), ("3681468f", "bz"), ("af0d1e43", "bz"), ("34b6868e", "bz")]:
    im = cv2.imread(f"dl/{n}.jpg")
    if kind == "bz":
        h, w = im.shape[:2]; im[int(h * 0.94):, int(w * 0.7):] = 0   # Bazaart watermark
        im = bbox_fit(im); ex = None
        if n in Z:
            ex = np.zeros(im.shape[:2], np.uint8); L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0]
            for zz in Z[n]:
                if zz[0] == "r": cv2.rectangle(ex, zz[1:3], zz[3:5], 1, -1)
                else: cv2.circle(ex, zz[1:3], zz[3], 1, -1)
            dk = (L.astype(int) < cv2.medianBlur(L, 41).astype(int) - 6).astype(np.uint8) & ex
            ex = cv2.morphologyEx(cv2.dilate(dk, np.ones((15, 15), np.uint8)), cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
        im = cl(im, extra=ex)
    else:
        im = fit(im); im[:960] = cl(im[:960].copy(), 0.1, 0.85); im[960:] = cl(im[960:].copy(), 0.1, 0.85)
    def fill_zone(im, m):
        from tat import nconv
        f = im.astype(np.float32); L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0]
        R = cv2.dilate(m, np.ones((7, 7), np.uint8)); Wt = ((L > 70) & (R == 0)).astype(np.float32)
        low, _ = nconv(f, Wt, 8); hi = np.clip(f - cv2.GaussianBlur(f, (0, 0), 2), -1, 4)
        a = cv2.GaussianBlur(R.astype(np.float32), (0, 0), 1.6)[..., None]
        return np.clip(f * (1 - a) + (low + hi) * a, 0, 255).astype(np.uint8)
    cv2.imwrite(f"r_{n}.png", im); print(n, "ok")
