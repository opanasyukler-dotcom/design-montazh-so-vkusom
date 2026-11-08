"""@mcg.va — stories «женская тусовка» (1080x1920): one thought per story, her story type (Montserrat ExtraBold main,
thin second, thin italic accent), small text pieces away from faces (MediaPipe), light black haze only where hard to read.
Photos are landscape: compact subjects are cropped to 9:16 around fx, wide scenes go full width (fit=contain).
  python3 -I stories_tusovka.py PHOTOS_DIR OUT_DIR MPENV_PYTHON MODELS_DIR
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stories_rody as R

P = lambda h: next(f for f in os.listdir(SRC) if f.startswith(h))
S = [  # (photo id, options, blocks)
    ("54E3497C", dict(fx=0.5), [("second", "Сегодня у нас, как всегда,"), ("main", "женская тусовка!")]),
    ("1D8618DB", dict(zoom=0.99, fy=0.78, at=(0.1, 0.24), one_side=True), [("second", "Папа уехал на тренировку,"), ("second", "а мы с Агатиком уже успели столько всего сделать.")]),
    ("51EA4A83", dict(fx=0.55), [("main", "Попробовали тыкву!"), ("script", "Нам понравилось,")]),
    ("8441869F", dict(fx=0.48), [("second", "правда, потом вся эта тыква"), ("script", "оказалась на кровати.")]),
    ("A09CF331", dict(fx=0.48), [("second", "После прикорма, естественно, пошли принимать ванну,")]),
    ("9CC28CF9", dict(fx=0.5, at=(0.0, 0.09)), [("second", "потому что чистыми после еды"), ("main", "мы теперь не остаёмся.")]),
    ("329009F6", dict(fx=0.42), [("second", "Мама успела выпить холодный кофе"), ("script", "и даже позавтракать!")]),
    ("B61AB164", dict(fx=0.55, at=(0.97,)), [("main", "Уже успех.")]),
    ("BE0BD576", dict(fit="contain"), [("second", "Потом мы много-много играли,"),
                                       ("second", "а впереди у нас ещё любимая мамина прогулка под сон Агатика.")]),
    ("6391B2B6", dict(fit="contain"), [("main", "Обожаю гулять, пока она спит."),
                                       ("second", "Можно хотя бы немного побыть со своими мыслями"), ("script", "и насладиться тишиной.")]),
]

if __name__ == "__main__":
    SRC, out, R.MPPY, R.MODELS = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
    for i, (h, kw, blocks) in enumerate(S, 1):
        R.photo_story(os.path.join(SRC, P(h)), blocks, os.path.join(out, f"story_{i:02d}.jpg"), **kw); print(i)
