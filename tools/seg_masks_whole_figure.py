"""body masks for the whole-figure reel photos (rembg isnet-general-use) -> $REEL_SRC/_work/<name>_mask.png
usage: REEL_SRC=... python3 tools/seg_masks_whole_figure.py IMG_0588 IMG_0594 ...
"""
import os
W_ = os.path.join(os.environ["REEL_SRC"], "_work")
import cv2, numpy as np, sys
from rembg import new_session, remove
sess = new_session("isnet-general-use")
for n in sys.argv[1:]:
    im = cv2.imread(os.path.join(W_, n + ".png")); h, w = im.shape[:2]; s = 1600 / h
    small = cv2.resize(im, (int(w * s), 1600))
    m = remove(small, session=sess, only_mask=True)
    m = cv2.resize(np.asarray(m), (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(W_, n + "_mask.png"), m); print(n, m.mean().round(1))
