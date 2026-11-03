"""The client's own before/after collages (IMG_9293 / 9294 / 9295: 3.5 months on top, before below, each half 1080x960,
the body large, no caption) remade with the 1 year 3 months photos (IMG_2052 / 2056 / 2058).

The before half is kept exactly from her collage.  The top half: her 3.5-month photo's placement in the collage is
recovered with SIFT + RANSAC (similarity transform photo -> collage); the 1-year photo is first brought onto the
3.5-month photo (same neck line, same neck..fingertips length, same body axis) and then put through that same transform,
so the 1-year body sits exactly where her 3.5-month body was.  The dark wall is pulled down to black like hers.

usage: python3 collages_1y3m_like_client.py SRC_DIR OUT_DIR       (SRC_DIR/drive holds <name>.png = the photos as RGB)
"""
import os, sys
import numpy as np, cv2
from PIL import Image

SRC, OUT = sys.argv[1:3]; DRIVE = os.path.join(SRC, "drive"); os.makedirs(OUT, exist_ok=True)
LM = {"IMG_9288.HEIC": (380, 3985), "IMG_9289.HEIC": (290, 3975), "IMG_9290.HEIC": (250, 3990),
      "IMG_2052.JPG": (160, 1730), "IMG_2056.JPG": (300, 2195), "IMG_2058.JPG": (290, 1910)}
SETS = [("IMG_9293.JPG", "IMG_9288.HEIC", "IMG_2052.JPG", "pryamo"), ("IMG_9294.PNG", "IMG_9289.HEIC", "IMG_2056.JPG", "3-4"),
        ("IMG_9295.PNG", "IMG_9290.HEIC", "IMG_2058.JPG", "sboku")]


def rgb(name):
    p = os.path.join(DRIVE, name + ".png") if not name.endswith(".PNG") and os.path.exists(os.path.join(DRIVE, name + ".png")) else os.path.join(DRIVE, name)
    return np.asarray(Image.open(p).convert("RGB"))


def to_black(im):
    """the wall pulled down to black, row by row (level = the darker of the outer 6 % strips)"""
    f = im.astype(np.float32) / 255; k = int(f.shape[1] * 0.06)
    bg = np.minimum(np.minimum(np.median(f[:, :k], 1), np.median(f[:, -k:], 1)), 0.35)[:, None, :]
    return (np.clip((f - bg) / (1 - bg), 0, 1) * 255).astype(np.uint8)


def axis(im, neck, tips):
    return float(np.median(np.nonzero(to_black(im)[neck:tips].mean(2) > 64)[1]))


def similarity(src, dst):
    """src photo -> dst (the collage half): SIFT matches, RANSAC similarity"""
    sift = cv2.SIFT_create(6000)
    g1 = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY); g2 = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    s = 1080 / src.shape[1]; g1s = cv2.resize(g1, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)   # match at a similar size
    k1, d1 = sift.detectAndCompute(g1s, None); k2, d2 = sift.detectAndCompute(g2, None)
    m = cv2.BFMatcher().knnMatch(d1, d2, k=2); good = [a for a, b in m if a.distance < 0.75 * b.distance]
    p1 = np.float32([k1[a.queryIdx].pt for a in good]) / s; p2 = np.float32([k2[a.trainIdx].pt for a in good])
    M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    print(f"  {len(good)} matches, {int(inl.sum())} inliers, scale {np.hypot(M[0, 0], M[1, 0]):.4f}")
    return M


for collage, mid, year, view in SETS:
    col = rgb(collage); top_her = col[:960]
    a35 = rgb(mid); a1y = rgb(year)
    M = similarity(a35, top_her)                                            # 3.5-month photo -> her top half
    (n35, t35), (n1, t1) = LM[mid], LM[year]
    sc = (t35 - n35) / (t1 - n1)                                            # 1-year photo -> 3.5-month photo
    ax35, ax1 = axis(a35, n35, t35), axis(a1y, n1, t1)
    A = np.array([[sc, 0, ax35 - sc * ax1], [0, sc, n35 - sc * n1], [0, 0, 1]])
    T = np.vstack([M, [0, 0, 1]]) @ A                                       # 1-year photo -> her top half
    top = cv2.warpAffine(to_black(a1y), T[:2], (1080, 960), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    out = col.copy(); out[:960] = top
    p = os.path.join(OUT, f"kollazh_1_god_3_mesyaca_{view}_kak_u_vas.jpg"); Image.fromarray(out).save(p, quality=95); print("wrote", p)
