import numpy as np, cv2
def retouch(img):
    """img float RGB 0..1 -> soften yellow/orange bruising + light skin smoothing"""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    skin = (h < 25) & (s > 40) & (v > 60)
    ref_s = np.median(s[skin]) if skin.any() else 90
    # bruise / staining: warmer-yellow hue and clearly more saturated than normal skin
    m = np.clip((s - ref_s * 1.12) / 45, 0, 1) * np.clip((h - 9) / 4, 0, 1) * (h < 32) * (v > 50)
    dark = np.clip((np.median(v[skin]) - v) / 60, 0, 1) * skin * np.clip((s - ref_s) / 40, 0, 1)
    m = np.maximum(m, dark * 0.8)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 6)
    hsv2 = hsv.copy()
    hsv2[..., 1] = s - (s - ref_s * 0.95) * 0.85 * m
    hsv2[..., 0] = h - (h - 11) * 0.6 * m
    hsv2[..., 2] = v + (np.median(v[skin]) - v).clip(0) * 0.35 * m
    out = cv2.cvtColor(np.clip(hsv2, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    sm = cv2.bilateralFilter(out, 9, 30, 9)
    sk = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 3)[..., None] * 0.45
    out = out * (1 - sk) + sm * sk
    return out.astype(np.float32) / 255
