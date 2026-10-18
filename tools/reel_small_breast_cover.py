exec(open("reel.py").read().split('if __name__ == "__main__":')[0])
# cover: profile ДО | ПОСЛЕ big, the headline in the 3:4 safe zone of the profile grid
TCV = Title([("У ОЧЕНЬ МАЛЕНЬКОЙ ГРУДИ", "H"), ("ЕСТЬ ПРЕИМУЩЕСТВО", "P"), ("о котором многие не знают", "S")], 104)
DEFER[0] = True; PENDING.clear(); LIGHT[0] = False
def half(img, cx):
    s = 0.62; return warp(img, s, cx, 1180, 270, 1160, (540, H))        # torso centred, neck at the top of the photo zone
fr = np.concatenate([half(SDO, 645), half(SPO, 702)], 1)
y = np.arange(H, dtype=np.float32)[:, None, None]; m = np.clip((y - 560) / 70, 0, 1) * np.clip((1700 - y) / 90, 0, 1)
fr = fr * m + np.float32([16, 16, 18]) / 255 * (1 - m)                  # photos fade into black above and below
fr = look(fr, 99)
chip(fr, CH_DO, 40, 600, 9, 0); chip(fr, CH_PO, 580, 600, 9, 0)
TCV.draw(fr, 250, 9, 0)
DEFER[0] = False
for args in PENDING: ablend(fr, *args)
blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
cv2.imwrite("cover2.jpg", cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
