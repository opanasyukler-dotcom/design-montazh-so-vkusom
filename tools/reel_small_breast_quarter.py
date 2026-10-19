# 3/4 view ДО/ПОСЛЕ: ПОСЛЕ aligned to ДО by pose, light retouch + bruises lightened, skin tone matched, stickers, collage
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
src = open("side.py").read()
exec(src[src.index("SP ="):src.index("B = np.array(Image.open")])                   # sticker, put, warm, top_cut, debruise, retouch
exec(src[src.index("def deepen"):src.index("A = match_skin(A, B)")])               # deepen, match_skin
B = np.array(Image.open("dl/q_before.jpg").convert("RGB")); A = np.array(Image.open("dl/q_after.jpg").convert("RGB"))
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
_seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path="seg.tflite"), output_confidence_masks=True))
def cutout(img):                                                                      # person only, studio wall -> black
    r = _seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
    m = 1 - np.squeeze(r.confidence_masks[0].numpy_view()).astype(np.float32)
    m = cv2.GaussianBlur(np.clip((m - 0.3) / 0.4, 0, 1), (0, 0), 1.5)[..., None]
    return (img * m + 12 * (1 - m)).astype(np.uint8)
B = cutout(B); A = cutout(A)
# pose (shoulder mid / hip mid): ДО (647,352)-(726,1479), ПОСЛЕ (640,390)-(843,1611) -> scale + shift ПОСЛЕ onto ДО
s = np.hypot(79, 1127) / np.hypot(203, 1221)
M = np.float32([[s, 0, 647 - s * 640], [0, s, 352 - s * 390]])
A = debruise(A, cx=820, cy=790, ax=330, ay=230)                                       # yellow bruising + marker lines, softened
A = retouch(A, k_red=0.35, k_band=0.3, k_tex=0.05)
put(B, sticker(70), 706, 645); put(B, sticker(62), 952, 700); put(A, sticker(70), 814, 746); put(A, sticker(64), 1090, 716)
A = cv2.warpAffine(A, M, (B.shape[1], B.shape[0]), flags=cv2.INTER_AREA, borderValue=(12, 12, 14))
A = match_skin(A, B)
B = top_cut(warm(B, 0.6), 150); A = top_cut(warm(A, 0.6), 150)
bot = 1862; fb = np.clip((bot - np.arange(B.shape[0], dtype=np.float32)) / 50, 0, 1)[:, None, None]   # ПОСЛЕ ends here (hands leave the photo)
B = (B * fb + 12 * (1 - fb)).astype(np.uint8); A = (A * fb + 12 * (1 - fb)).astype(np.uint8)
cv2.imwrite("q_do.png", cv2.cvtColor(B, cv2.COLOR_RGB2BGR)); cv2.imwrite("q_po.png", cv2.cvtColor(A, cv2.COLOR_RGB2BGR))
W, H = 1080, 1920; col = np.zeros((H, W, 3), np.uint8); col[:] = (8, 8, 9)
def panel(img, cx):
    crop = img[130:1880, int(cx - 480):int(cx + 480)].astype(np.float32); h, w = crop.shape[:2]; e = 70
    m = (np.clip(np.arange(w) / e, 0, 1) * np.clip((w - 1 - np.arange(w)) / e, 0, 1))[None] * (np.clip((h - 1 - np.arange(h)) / e, 0, 1) * np.clip(np.arange(h) / 40, 0, 1))[:, None]
    crop = crop * m[..., None] + 8 * (1 - m[..., None]); return cv2.resize(crop.astype(np.uint8), (526, 958), interpolation=cv2.INTER_AREA)
col[620:1578, 14:540] = panel(B, 690); col[620:1578, 548:1074] = panel(A, 690)
im = Image.fromarray(col); d = ImageDraw.Draw(im); d.text((540, 540), "УВЕЛИЧЕНИЕ ГРУДИ ИМПЛАНТАМИ", font=osw(92), fill=(255, 255, 255), anchor="ms")
im.save("collage_q.jpg", quality=95)
