def timg(txt, kind, size):
    f = osw(size) if kind in ("H", "P") else mont(500, 36); asc, desc = f.getmetrics(); tw = int(f.getlength(txt))
    im = Image.new("RGBA", (tw + 8, asc + desc + 4), (0, 0, 0, 0)); ImageDraw.Draw(im).text((4, 2 + asc), txt, font=f, fill=(255, 255, 255, 255), anchor="ls")
    return to_np(im)
def cols(kind): return (PINK, ROSE) if kind == "P" else (WHITE, BLUE)
class Pill:
    """outlined rounded frame opens from the centre, then the text wipes in left -> right (reference opening)"""
    def __init__(s, lines, size=84):
        s.ims = [(timg(t, k, size), k) for t, k in lines]
        s.w = max(i.shape[1] for i, _ in s.ims) + 90; s.h = sum(i.shape[0] for i, _ in s.ims) + 50
        im = Image.new("RGBA", (s.w + 8, s.h + 8), (0, 0, 0, 0))
        ImageDraw.Draw(im).rounded_rectangle((3, 3, s.w + 4, s.h + 4), radius=min(46, s.h // 2), outline=(255, 255, 255, 255), width=4)
        s.box = to_np(im)
    def draw(s, fr, yc, t, t0, t1):
        if t < t0 or t > t1 + 0.5: return
        pb = ease((t - t0) / 0.5); out = 1 - ease((t - t1) / 0.45) if t > t1 else 1.0
        bw = max(2, int(s.box.shape[1] * pb)); x0 = 540 - s.box.shape[1] / 2; y0 = yc - s.box.shape[0] / 2
        b = s.box.copy(); c = (s.box.shape[1] - bw) // 2; b[:, :c] = 0; b[:, c + bw:] = 0
        ablend(fr, b, x0, y0, out * 0.95, WHITE, BLUE)
        yy = y0 + 4 + 25
        for k, (im, kind) in enumerate(s.ims):
            pw = ease((t - t0 - 0.35 - 0.2 * k) / 0.55)
            if t > t1: pw = min(pw, 1 - ease((t - t1 - 0.05 * k) / 0.35))
            if pw > 0:
                vis = im.copy(); vis[:, int(vis.shape[1] * pw):] = 0
                ablend(fr, vis, 540 - im.shape[1] / 2, yy, 1.0, *cols(kind))
            yy += im.shape[0]
class SlideL:
    """left-aligned lines slide out of their own mask, one by one"""
    def __init__(s, lines, size=80, x=70): s.ims = [(timg(t, k, size), k) for t, k in lines]; s.x = x
    def draw(s, fr, y, t, t0, t1):
        if t < t0 or t > t1 + 0.5: return
        out = 1 - ease((t - t1) / 0.4) if t > t1 else 1.0
        for k, (im, kind) in enumerate(s.ims):
            p = ease((t - t0 - 0.28 * k) / 0.6)
            if p > 0:
                w = int(im.shape[1] * p); vis = np.zeros_like(im); vis[:, :w] = im[:, im.shape[1] - w:]
                ablend(fr, vis, s.x, y, out, *cols(kind))
                if kind == "P" and p > 0.3:                                 # thin powder bar grows under the key line
                    bar = np.zeros((6, int((im.shape[1] - 8) * ease((t - t0 - 0.28 * k - 0.25) / 0.6)) + 1, 4), np.float32); bar[..., 3] = 1
                    ablend(fr, bar, s.x + 4, y + im.shape[0] + 2, out, PINK, ROSE)
            y += im.shape[0] + 10
class Pop:
    """each line punches in from a bigger size"""
    def __init__(s, lines, size=110): s.ims = [(timg(t, k, size), k) for t, k in lines]
    def draw(s, fr, y, t, t0, t1):
        if t < t0 or t > t1 + 0.5: return
        out = 1 - ease((t - t1) / 0.4) if t > t1 else 1.0
        for k, (im, kind) in enumerate(s.ims):
            lt = t - t0 - 0.3 * k
            if lt > 0:
                sc = 1 + 0.45 * (1 - ease(lt / 0.45)); im2 = cv2.resize(im, (int(im.shape[1] * sc), int(im.shape[0] * sc)))
                ablend(fr, im2, 540 - im2.shape[1] / 2, y + im.shape[0] / 2 - im2.shape[0] / 2, out * min(1, lt / 0.2), *cols(kind))
            y += im.shape[0] + 4
def brackets(fr, cx, cy, w, h, t, t0, t1):
    """viewfinder corners that close in on the area (reference)"""
    if t < t0 or t > t1 + 0.4: return
    p = ease((t - t0) / 0.7); al = min(1, (t - t0) / 0.2) * (1 - ease((t - t1) / 0.4) if t > t1 else 1)
    sc = 1.5 - 0.5 * p; hw, hh = w * sc / 2, h * sc / 2; L = 60
    m = np.zeros((H, W), np.uint8)
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = int(cx + sx * hw), int(cy + sy * hh)
            cv2.line(m, (x, y), (x - sx * L, y), 255, 5, cv2.LINE_AA); cv2.line(m, (x, y), (x, y - sy * L), 255, 5, cv2.LINE_AA)
    a = m.astype(np.float32)[..., None] / 255 * al; fr *= 1 - a; fr += a
CL = {}
def callout(fr, txt, t, t0, pt, lx, ly, side, t1=1e9):
    """diagnosis callout: pulsing dot -> thin line -> underlined small label (reference)"""
    if t < t0 or t > t1 + 0.4: return
    al = 1 - ease((t - t1) / 0.4) if t > t1 else 1.0
    if txt not in CL:
        f = mont(600, 42); asc, desc = f.getmetrics(); im = Image.new("RGBA", (int(f.getlength(txt)) + 8, asc + desc + 4), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((4, 2 + asc), txt, font=f, fill=(255, 255, 255, 255), anchor="ls"); CL[txt] = to_np(im)
    im = CL[txt]; tw = im.shape[1]
    ax = lx                                                                # line meets the label's outer end, underline runs inward
    pl = ease((t - t0 - 0.2) / 0.5); pu = ease((t - t0 - 0.6) / 0.4)
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (int(pt[0]), int(pt[1])), 8, 255, -1, cv2.LINE_AA)
    pr = ((t - t0) * 0.9) % 1; ring_m = np.zeros((H, W), np.uint8); cv2.circle(ring_m, (int(pt[0]), int(pt[1])), int(14 + 26 * pr), 255, 3, cv2.LINE_AA)
    ex, ey = pt[0] + (ax - pt[0]) * pl, pt[1] + (ly + 6 - pt[1]) * pl
    cv2.line(m, (int(pt[0]), int(pt[1])), (int(ex), int(ey)), 255, 3, cv2.LINE_AA)
    if pu > 0: cv2.line(m, (int(ax), int(ly + 6)), (int(ax - side * tw * pu), int(ly + 6)), 255, 3, cv2.LINE_AA)
    a = (m.astype(np.float32) + ring_m.astype(np.float32) * (1 - pr) * 0.8)[..., None] / 255 * al * min(1, (t - t0) / 0.15)
    fr *= 1 - np.clip(a, 0, 1); fr += np.clip(a, 0, 1)
    if pu > 0:
        x = lx if side < 0 else lx - tw; vis = im.copy(); vis[:, int(tw * pu):] = 0 if side < 0 else vis[:, int(tw * pu):]
        if side > 0: vis[:, :int(tw * (1 - pu))] = 0
        ablend(fr, vis, x, ly - im.shape[0] - 2, al, WHITE, BLUE)

def focus(fr, lt, dur=0.9):                                             # reference: start out of focus, pull in
    sg = 14 * (1 - ease(lt / dur)) ** 1.5
    return cv2.GaussianBlur(fr, (0, 0), sg) if sg > 0.5 else fr
