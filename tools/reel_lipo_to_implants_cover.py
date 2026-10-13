exec(open("reel.py").read().split('if __name__ == "__main__":')[0])
TCV = Title([("ГОД НАЗАД", "H"), ("Я БОЯЛАСЬ ИМПЛАНТОВ", "P"), ("а сейчас сама пришла за увеличением груди", "S")], 112)
DEFER[0] = True; PENDING.clear(); LIGHT[0] = False
def soft(img, sh):
    h, w = img.shape[:2]; m = np.zeros((h, w), np.uint8); bd = int(h * 0.06); m[bd:h - int(h * 0.1), :] = 255
    m = cv2.GaussianBlur(m, (0, 0), h * 0.03); mm = np.repeat(m[..., None], 3, 2)
    return body_view(img, sh, 410, 560, (540, H)), body_view(mm, sh, 410, 560, (540, H))[..., :1]
(a_, ma), (b_, mb) = soft(BF["f"], SHO[("B", "f")]), soft(AF["f"], SHO[("A", "f")])
bgc = np.float32([16, 16, 18]) / 255; a_ = a_ * ma + bgc * (1 - ma); b_ = b_ * mb + bgc * (1 - mb)
fr = np.concatenate([a_, b_], 1); fr = look(fr, 99); fr = fr * (1 - HAZE)
chip(fr, CH_DO, 40, 228, 9, 0); chip(fr, CH_PO, 580, 228, 9, 0)
TCV.draw(fr, 1180, 9, 0)
DEFER[0] = False
for args in PENDING: ablend(fr, *args)
blend(fr, LOGO, 540 - LOGO.shape[1] / 2, 120, 0.9)
cv2.imwrite("cover.jpg", cv2.cvtColor((np.clip(fr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
