exec(open("scar.py").read().split('if __name__ == "__main__":')[0])
TCV = Title([("РУБЕЦ СТАЛ КРАСНЕЕ.", "H"), ("ЭТО НОРМАЛЬНО?", "P")], 116)
DEFER[0] = True; PENDING.clear(); LIGHT[0] = False
fr = vframe(84.6, 1.35, 560, 800, 1150); fr = look(fr, 99) * (1 - HAZE)
scar_card(fr, 15.0, -10)                                                  # red, month-2 scar on the card
TCV.draw(fr, 1180, 9, 0)
DEFER[0] = False
for args in PENDING: ablend(fr, *args)
blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
cv2.imwrite("cover_scar.jpg", cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
