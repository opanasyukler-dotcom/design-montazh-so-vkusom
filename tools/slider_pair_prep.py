import cv2, numpy as np, sys
W, H = 1080, 1920; YA, YB = 210, 1330                       # panel rows for shoulder top and fingertips (the same for ДО and ПОСЛЕ)
def place(img, m, y0, y1, cut_above):
    m = m.astype(np.float32) / 255; m[:cut_above] = 0
    s = (YB - YA) / (y1 - y0); rows = m[y0:y1]; xs = np.where(rows.max(0) > 0.5)[0]
    cols = [np.where(r > 0.5)[0] for r in rows[::10]]; cx = np.median([(c[0] + c[-1]) / 2 for c in cols if len(c)])
    M = np.float32([[s, 0, W / 2 - cx * s], [0, s, YA - y0 * s]])
    fg = cv2.warpAffine(img.astype(np.float32) / 255, M, (W, H), flags=cv2.INTER_AREA); a = cv2.warpAffine(cv2.GaussianBlur(m, (0, 0), 1.2), M, (W, H))[..., None]
    return (fg * a * 255).astype(np.uint8)
d = place(cv2.imread('IMG_1486.jpg'), cv2.imread('m1486.png', 0), 100, 1660, 70)
p = place(cv2.imread('IMG_1460.jpg'), cv2.imread('m1460.png', 0), 830, 2250, 0)
fade = np.clip((1540 - np.arange(H)) / 60.0, 0, 1)[:, None, None]       # both end at the same height (the ДО photo's floor line)
cv2.imwrite('sl_do.png', (d * fade).astype(np.uint8)); cv2.imwrite('sl_po.png', (p * fade).astype(np.uint8))
