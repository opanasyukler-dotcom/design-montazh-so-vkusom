exec(open("reel.py").read().split('if __name__ == "__main__":')[0])
TCV = Title([("НАДО БЫЛО ПРОСТО", "H"), ("ДАВНО ПРИЙТИ", "P"), ("4 детей, близняшки и результат через 2 месяца", "S")], 112)
DEFER[0] = True; PENDING.clear(); LIGHT[0] = False
def half(img, key, Y0=420, Y1=1010):
    top, bot, cx = ANC[key]; s = (Y1 - Y0) / (bot - top)
    return warp(img, s, cx, (top + bot) / 2, 270, (Y0 + Y1) / 2, (540, H))
fr = np.concatenate([half(BF["f"], ("B", "f")), half(AF["f"], ("A", "f"))], 1)
fr = look(fr, 99); fr = fr * (1 - HAZE)
chip(fr, CH_DO, 40, 228, 9, 0); chip(fr, CH_PO, 580, 228, 9, 0)
TCV.draw(fr, 1240, 9, 0)
DEFER[0] = False
for args in PENDING: ablend(fr, *args)
blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
cv2.imwrite("cover.jpg", cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
