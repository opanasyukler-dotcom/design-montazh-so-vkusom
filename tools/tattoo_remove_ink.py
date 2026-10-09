import cv2, numpy as np
from tat import raw_skin, skin_mask2, nconv
K = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
PROT = [4.7, 4.0, 0.25]
def ink_mask(bgr, pose=None):
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    Lm, Am, Bm = [cv2.medianBlur(lab[..., i].astype(np.uint8), 31).astype(np.float32) for i in range(3)]
    Am -= 128; Bm -= 128
    rs = raw_skin(bgr); sk = cv2.morphologyEx(rs, cv2.MORPH_CLOSE, K(15))
    inner = cv2.erode(sk, K(9))
    dark = Lm - L
    chroma = np.sqrt(A ** 2 + B ** 2); chroma_m = np.sqrt(Am ** 2 + Bm ** 2)
    # ink: darker AND loses warm chroma (grey/blue); shadows/creases keep or gain chroma
    bl0 = cv2.morphologyEx((B < -10).astype(np.uint8), cv2.MORPH_OPEN, K(5))
    nb0, lb0, sb0, _ = cv2.connectedComponentsWithStats(bl0, 8)
    big = np.isin(lb0, [i for i in range(1, nb0) if sb0[i, 4] > 300]).astype(np.uint8)
    blue = cv2.dilate(big, K(21))                                       # stickers: large blue blobs only
    ink = (dark > 6) & (chroma < chroma_m * 0.9) & (inner > 0) & (L > 35) & (blue == 0)
    ink = ink.astype(np.uint8)
    g = L.astype(np.uint8)
    b7 = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, K(7)).astype(int); b21 = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, K(21)).astype(int)
    near = cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, K(2)), K(31))
    lines = (b7 > 5) & (b7 >= 0.55 * b21) & (near > 0) & (inner > 0) & (blue == 0)
    ink = (ink | lines).astype(np.uint8)
    bl = cv2.morphologyEx((B < -10).astype(np.uint8), cv2.MORPH_OPEN, K(5))   # breasts: protect around each sticker
    nb, lb, sb, cb = cv2.connectedComponentsWithStats(bl, 8)
    for i in range(1, nb):
        ar = sb[i, cv2.CC_STAT_AREA]
        if ar < 120: continue
        r = max(np.sqrt(ar / np.pi), sb[i, cv2.CC_STAT_HEIGHT] / 2)
        cx, cy = cb[i]
        cv2.ellipse(ink, (int(cx), int(cy + PROT[2] * r)), (int(PROT[0] * r), int(PROT[1] * r)), 0, 0, 360, 0, -1)
    if pose is not None:
        l, r = pose[11], pose[12]; sw = np.linalg.norm(l - r)
        ink[: int(min(l[1], r[1]) + 0.1 * sw)] = 0
        for wi, fi in ((15, (17, 19, 21)), (16, (18, 20, 22))):
            fc = np.mean([pose[k] for k in fi], 0); c = pose[wi] + (fc - pose[wi]) * 1.3
            cv2.circle(ink, tuple(int(v) for v in c), int(0.14 * sw), 0, -1)
    n, lab_, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    area = st[:, cv2.CC_STAT_AREA]; fill = area / np.maximum(st[:, cv2.CC_STAT_WIDTH] * st[:, cv2.CC_STAT_HEIGHT], 1)
    ok = (area >= 10); ok[0] = False   # drop solid round blobs (stickers)
    return ok[lab_].astype(np.uint8)
def remove_ink(bgr, m):
    rs = cv2.morphologyEx(raw_skin(bgr), cv2.MORPH_OPEN, K(3)); sk = skin_mask2(bgr)
    body = cv2.morphologyEx(rs, cv2.MORPH_CLOSE, K(9)) & sk
    R = cv2.morphologyEx(cv2.dilate(m, K(11)), cv2.MORPH_CLOSE, K(21)) & body
    f = bgr.astype(np.float32)
    Wt = (cv2.erode(rs, K(5)) & (1 - R)).astype(np.float32)
    low, den = nconv(f, Wt, 6); low2, _ = nconv(f, Wt, 16)
    w2 = np.clip(1 - den / 0.25, 0, 1); low = low * (1 - w2) + low2 * w2
    hi = np.clip(f - cv2.GaussianBlur(f, (0, 0), 2), -1.0, 4)
    a = cv2.GaussianBlur(R.astype(np.float32), (0, 0), 1.6)[..., None] * body[..., None]
    return np.clip(f * (1 - a) + (low + hi) * a, 0, 255).astype(np.uint8)
def clean3(bgr, pose=None, passes=4):
    r = bgr
    for _ in range(passes): r = remove_ink(r, ink_mask(r, pose))
    return r
