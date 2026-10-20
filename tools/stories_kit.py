"""Stories kit for @mcg.va (1080x1920). Type roles chosen by the blog:
  main   — the Instagram «Classic» bold look (Inter Bold): the main text
  second — thin Montserrat (wght 300): secondary lines
  detail — Montserrat Italic (wght 300): small details, asides, labels
Fonts live in fonts/stories (Inter, Montserrat: SIL OFL 1.1). Instagram safe zone: text between y ≈ 250 and y ≈ 1580.
"""
import os
from PIL import ImageFont

W, H = 1080, 1920
SAFE_TOP, SAFE_BOTTOM = 250, 1580
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "stories")
ROLES = {                       # file, default size, variable weight
    "main": ("Inter-Bold.ttf", 46, None),
    "second": ("Montserrat-VariableFont_wght.ttf", 38, 300),
    "detail": ("Montserrat-Italic-VariableFont_wght.ttf", 32, 300),
}
INK = (17, 17, 17)
_cache = {}

def font(role, size=None, weight=None):
    f, s, wg = ROLES[role]; size = size or s; wg = weight or wg
    key = (role, size, wg)
    if key not in _cache:
        fnt = ImageFont.truetype(os.path.join(FONT_DIR, f), size)
        if wg: fnt.set_variation_by_axes([wg])
        _cache[key] = fnt
    return _cache[key]

def wrap(text, fnt, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if fnt.getlength(t) <= maxw or not cur: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines
