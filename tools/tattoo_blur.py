import cv2, numpy as np
from tat3 import ink_mask
from tat4 import dense, script_zone, arm_zone, K
from tat import raw_skin, nconv
def tattoo_region(bgr, pose=None):
    m = dense(ink_mask(bgr, pose)).astype(np.uint8)
    rs0 = cv2.morphologyEx(raw_skin(bgr), cv2.MORPH_OPEN, K(5))
    cs, _ = cv2.findContours(cv2.morphologyEx(rs0, cv2.MORPH_CLOSE, K(9)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    skin = np.zeros_like(rs0); cv2.drawContours(skin, cs, -1, 1, -1)
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0]
    sz = script_zone(bgr) & cv2.erode(skin, K(11))
    m |= ((L.astype(int) < cv2.medianBlur(L, 31).astype(int) - 5) & (sz > 0)).astype(np.uint8)
    near = cv2.dilate(m, K(41)); az = arm_zone(bgr, pose) & cv2.erode(skin, K(11)) & near
    m |= ((L.astype(int) < cv2.medianBlur(L, 61).astype(int) - 6) & (az > 0)).astype(np.uint8)
    R = cv2.morphologyEx(cv2.dilate(m, K(17)), cv2.MORPH_CLOSE, K(35)) & skin
    return R, skin
def blur_tattoos(bgr, pose=None, sigma=14):
    R, skin = tattoo_region(bgr, pose)
    if not R.any(): return bgr
    f = bgr.astype(np.float32)
    bl, _ = nconv(f, skin.astype(np.float32), sigma)            # blur using skin pixels only (no background bleed)
    a = cv2.GaussianBlur(R.astype(np.float32), (0, 0), 5)[..., None] * skin[..., None]
    return np.clip(f * (1 - a) + bl * a, 0, 255).astype(np.uint8)
