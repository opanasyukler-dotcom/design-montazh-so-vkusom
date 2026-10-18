exec(open("reel.py").read().split('if __name__ == "__main__":')[0])
TCV = Title([("У ОЧЕНЬ МАЛЕНЬКОЙ ГРУДИ", "H"), ("ЕСТЬ ПРЕИМУЩЕСТВО", "P"), ("о котором многие не знают", "S")], 100)
DEFER[0] = True; PENDING.clear(); LIGHT[0] = False
img = COLL2.copy(); img[420:590] = img[200:210].mean((0, 1)).astype(np.uint8)      # profile collage, its title removed
fr = img.astype(np.float32) / 255
chip(fr, CH_DO, 40, 620, 9, 0); chip(fr, CH_PO, 580, 620, 9, 0)
TCV.draw(fr, 250, 9, 0)
DEFER[0] = False
for args in PENDING: ablend(fr, *args)
blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
cv2.imwrite("cover.jpg", cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
