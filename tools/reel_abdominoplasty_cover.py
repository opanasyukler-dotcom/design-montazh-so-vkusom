exec(open("reel.py").read().split('if __name__ == "__main__":')[0])
TCV = Title([("НАДО БЫЛО ПРОСТО", "H"), ("ДАВНО ПРИЙТИ", "P"), ("4 детей, близняшки и результат через 2 месяца", "S")], 112)
DEFER[0] = True; PENDING.clear(); LIGHT[0] = False
# same anatomical span on both sides: from the neck down to the fingertips (mid-thigh)
SPAN = {"B": (830, 2420, 1088), "A": (995, 2300, 1025)}
Y0, Y1 = 330, 1190
def half(img, key):
    top, bot, cx = SPAN[key]; s = (Y1 - Y0) / (bot - top)
    im = warp(img, s, cx, (top + bot) / 2, 270, (Y0 + Y1) / 2, (540, H))
    y = np.arange(H, dtype=np.float32)[:, None, None]
    m = np.clip((y - Y0 + 6) / 30, 0, 1) * np.clip((Y1 + 40 - y) / 110, 0, 1)     # soft cut at the neck and below the fingertips
    return im * m + np.float32([16, 16, 18]) / 255 * (1 - m)
fr = np.concatenate([half(BF["f"], "B"), half(AF["f"], "A")], 1)
fr = look(fr, 99); fr = fr * (1 - HAZE)
chip(fr, CH_DO, 40, 228, 9, 0); chip(fr, CH_PO, 580, 228, 9, 0)
TCV.draw(fr, 1260, 9, 0)
DEFER[0] = False
for args in PENDING: ablend(fr, *args)
blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
cv2.imwrite("cover.jpg", cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
