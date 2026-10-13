exec(open("reel.py").read().split('if __name__ == "__main__":')[0])
TCV = Title([("ГОД НАЗАД", "H"), ("Я БОЯЛАСЬ ИМПЛАНТОВ", "P"), ("а сейчас сама пришла за увеличением груди", "S")], 112)
DEFER[0] = True; PENDING.clear(); LIGHT[0] = False
a_ = body_view(BF["f"], SHO[("B", "f")], 520, 470, (540, H)); b_ = body_view(AF["f"], SHO[("A", "f")], 520, 470, (540, H))
fr = np.concatenate([a_, b_], 1); fr = look(fr, 99); fr = fr * (1 - HAZE)
chip(fr, CH_DO, 40, 228, 9, 0); chip(fr, CH_PO, 580, 228, 9, 0)
TCV.draw(fr, 1180, 9, 0)
DEFER[0] = False
for args in PENDING: ablend(fr, *args)
blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
cv2.imwrite("cover.jpg", cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
