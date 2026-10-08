import cv2, numpy as np
K = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
def tattoo_mask(bgr, pose=None):
    h, w = bgr.shape[:2]
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb); hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Y, Cr, Cb = [ycc[..., i].astype(int) for i in range(3)]
    skin = ((Cr > 138) & (Cr < 180) & (Cb > 85) & (Cb < 128) & (Y > 60)).astype(np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, K(25)); skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, K(9))
    inner = cv2.erode(skin, K(13))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b7 = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, K(7)).astype(int)
    b21 = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, K(21)).astype(int)
    thin = b7 >= 0.55 * b21          # line-like, not broad shading
    weak = ((b7 > 5) & thin & (inner > 0)).astype(np.uint8)
    strong = (b7 > 13) & thin
    blue = cv2.dilate(((hsv[..., 0] > 95) & (hsv[..., 0] < 135) & (hsv[..., 1] > 60)).astype(np.uint8), K(25))
    weak[blue > 0] = 0
    if pose is not None:
        l, r = pose[11], pose[12]; sw = np.linalg.norm(l - r)
        weak[: int(min(l[1], r[1]) + 0.12 * sw)] = 0            # head, neck, necklace
        for wi, fi in ((15, (17, 19, 21)), (16, (18, 20, 22))):    # fingers only
            fc = np.mean([pose[k] for k in fi], 0); c = pose[wi] + (fc - pose[wi]) * 1.4
            cv2.circle(weak, tuple(int(v) for v in c), int(0.13 * sw), 0, -1)
    if pose is not None:   # dense dark tattoo fills on the arms
        arm = np.zeros_like(weak)
        for a_, b_ in ((11, 13), (13, 15), (12, 14), (14, 16)):
            cv2.line(arm, tuple(int(v) for v in pose[a_]), tuple(int(v) for v in pose[b_]), 1, max(3, int(0.32 * sw)))
        est = cv2.medianBlur(g, 41).astype(int)
        dk = ((g.astype(int) < est - 18) & (arm > 0) & (inner > 0) & (blue == 0)).astype(np.uint8)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(dk, 8)
        okd = (st2[:, cv2.CC_STAT_AREA] >= 6) & (st2[:, cv2.CC_STAT_AREA] <= 4000); okd[0] = False
        dkm = okd[lab2].astype(np.uint8)
        weak = weak | dkm; strong = strong | (dkm > 0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(weak, 8)
    has = np.zeros(n, bool); has[np.unique(lab[strong & (weak > 0)])] = True; has[0] = False
    ok = has & (st[:, cv2.CC_STAT_AREA] >= 8)
    m = ok[lab].astype(np.uint8)
    return cv2.dilate(m, K(7))
def remove(bgr, m):
    out = cv2.inpaint(bgr, m * 255, 7, cv2.INPAINT_TELEA)
    soft = cv2.GaussianBlur(cv2.dilate(m, K(9)).astype(np.float32), (0, 0), 3)[..., None]
    sm = cv2.bilateralFilter(out, 11, 30, 9)
    return np.clip(out * (1 - soft * 0.7) + sm * soft * 0.7, 0, 255).astype(np.uint8)
def remove2(bgr, m, inner_skin=None):
    R = cv2.morphologyEx(cv2.dilate(m, K(15)), cv2.MORPH_CLOSE, K(31))
    if inner_skin is not None: R = R & inner_skin
    f = bgr.astype(np.float32)
    low = cv2.inpaint(bgr, R * 255, 15, cv2.INPAINT_TELEA).astype(np.float32)
    low = cv2.GaussianBlur(low, (0, 0), 5) * R[..., None] + f * (1 - R[..., None])
    low = cv2.inpaint(np.clip(low, 0, 255).astype(np.uint8), cv2.erode(R, K(3)) * 0, 3, cv2.INPAINT_TELEA).astype(np.float32)
    hi = f - cv2.GaussianBlur(f, (0, 0), 2)
    hi = np.clip(hi, -2.5, 5)
    res = low + hi
    a = cv2.GaussianBlur(R.astype(np.float32), (0, 0), 4)[..., None]
    return np.clip(f * (1 - a) + res * a, 0, 255).astype(np.uint8)
def skin_inner(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = [ycc[..., i].astype(int) for i in range(3)]
    skin = ((Cr > 138) & (Cr < 180) & (Cb > 85) & (Cb < 128) & (Y > 60)).astype(np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, K(25)); skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, K(9))
    return cv2.erode(skin, K(7))
def skin_mask(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = [ycc[..., i].astype(int) for i in range(3)]
    skin = ((Cr > 138) & (Cr < 180) & (Cb > 85) & (Cb < 128) & (Y > 60)).astype(np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, K(25)); return cv2.morphologyEx(skin, cv2.MORPH_OPEN, K(9))
def nconv(f, Wt, s):
    num = cv2.GaussianBlur(f * Wt[..., None], (0, 0), s); den = cv2.GaussianBlur(Wt, (0, 0), s)[..., None]
    return num / np.maximum(den, 1e-4), den
def remove3(bgr, m):
    sk = skin_mask(bgr)
    R = cv2.morphologyEx(cv2.dilate(m, K(13)), cv2.MORPH_CLOSE, K(31)) & sk
    f = bgr.astype(np.float32); g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Wt = (sk & (1 - R)).astype(np.float32)
    for _ in range(2):  # refine: drop dark (tattoo shading) pixels from the reference
        est, _ = nconv(f, Wt, 14)
        eg = est @ np.float32([0.114, 0.587, 0.299])
        dark = (g < eg - 10) & (cv2.dilate(R, K(41)) > 0)
        R = (R | (dark & (sk > 0))).astype(np.uint8)
        R = cv2.morphologyEx(R, cv2.MORPH_CLOSE, K(9)) & sk
        Wt = (sk & (1 - R)).astype(np.float32)
    low, _ = nconv(f, Wt, 14)
    hi = f - cv2.GaussianBlur(f, (0, 0), 2); hi = np.clip(hi, -1.0, 4)
    res = low + hi
    a = cv2.GaussianBlur(R.astype(np.float32), (0, 0), 3)[..., None] * cv2.erode(sk, K(3))[..., None]
    return np.clip(f * (1 - a) + res * a, 0, 255).astype(np.uint8)
def skin_mask2(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb); hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Y, Cr, Cb = [ycc[..., i].astype(int) for i in range(3)]
    skin = ((Cr > 140) & (Cr < 180) & (Cb > 85) & (Cb < 125) & (Y > 60) & (hsv[..., 1] > 65)).astype(np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, K(25)); return cv2.morphologyEx(skin, cv2.MORPH_OPEN, K(9))
def raw_skin(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb); hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Y, Cr, Cb = [ycc[..., i].astype(int) for i in range(3)]
    return ((Cr > 140) & (Cr < 180) & (Cb > 85) & (Cb < 125) & (Y > 60) & (hsv[..., 1] > 65)).astype(np.uint8)
def remove4(bgr, m):
    sk = skin_mask2(bgr); rs = cv2.morphologyEx(raw_skin(bgr), cv2.MORPH_OPEN, K(3))
    body = cv2.dilate(rs, K(9)) & sk
    R = cv2.morphologyEx(cv2.dilate(m, K(11)), cv2.MORPH_CLOSE, K(21)) & body
    f = bgr.astype(np.float32)
    Wt = (cv2.erode(rs, K(5)) & (1 - R)).astype(np.float32)
    low, den = nconv(f, Wt, 9)
    low2, _ = nconv(f, Wt, 20)
    w2 = np.clip(1 - den / 0.25, 0, 1)            # where few references nearby, use wider estimate
    low = low * (1 - w2) + low2 * w2
    hi = f - cv2.GaussianBlur(f, (0, 0), 2); hi = np.clip(hi, -1.0, 4)
    a = cv2.GaussianBlur(R.astype(np.float32), (0, 0), 2.5)[..., None] * body[..., None]
    return np.clip(f * (1 - a) + (low + hi) * a, 0, 255).astype(np.uint8)
def clean(bgr, pose):
    r = bgr
    for _ in range(3): r = remove4(r, tattoo_mask(r, pose))
    return r
