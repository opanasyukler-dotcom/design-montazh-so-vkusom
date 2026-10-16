exec(open("reel.py").read().split('if __name__ == "__main__":')[0])
TCV = Title([("ЧТО ДЕЛАЮТ С ЖИРОМ", "H"), ("ПЕРЕД ЛИПОФИЛИНГОМ?", "P"), ("показываю прямо из операционной", "S")], 116)
DEFER[0] = True; PENDING.clear(); LIGHT[0] = False
fr = broll(53.0, 1.08, 900); fr = look(fr, 99) * (1 - HAZE)
TCV.draw(fr, 1170, 9, 0)
DEFER[0] = False
for args in PENDING: ablend(fr, *args)
blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
cv2.imwrite("cover.jpg", cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
