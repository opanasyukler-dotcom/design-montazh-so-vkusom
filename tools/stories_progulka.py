""""Good day / walk" stories in the same minimal look as stories_morkovka.

Reuses the stories_morkovka renderer: full-screen muted photo, small white type in a free zone
away from faces, one accent per slide, barely-there dark veil under the text.

Usage: python3 tools/stories_progulka.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stories_morkovka as base  # noqa: E402
from stories_morkovka import CREAM, NB  # noqa: E402

UPLOADS = "/root/.claude/uploads/407108f0-3ce4-585b-bf12-4b6e65a8be4f"
base.OUT = os.path.join(base.ROOT, "stories_progulka")

SLIDES = [
    {   # sleeping in the bassinet: face mid-frame, blanket below is free
        "photo": f"{UPLOADS}/7011345a-image.png",
        "groups": [{
            "anchor": ("bottom", 1620), "x": 100, "width": 830, "align": "left",
            "blocks": [
                ("dots", "", {}),
                ("bold", f"сегодня у{NB}нас на{NB}удивление достаточно хороший день, и{NB}я этому "
                         f"безумно рада!", {"size": 48}),
                ("light", f"но у{NB}нас, как всегда, закон подлости!", {}),
            ],
        }],
    },
    {   # blankets close-up, no faces: text mid-frame with a side bar
        "photo": f"{UPLOADS}/d1619f6e-image.png",
        "groups": [{
            "anchor": ("top", 620), "x": 100, "width": 840, "align": "left",
            "blocks": [
                ("light", f"когда Агата капризничает и{NB}я уже не{NB}знаю, куда себя деть, к{NB}нам "
                          f"почему-то никто не{NB}приходит.", {"bar": True}),
                ("bold", f"а когда у{NB}нас идеальный ребёнок, который прекрасно себя ведёт, сразу "
                         f"собирается полный состав помощников", {"size": 46}),
            ],
        }],
    },
    {   # stroller handle: thin rule, story, then the punchline in cream
        "photo": f"{UPLOADS}/fa09248c-image.png",
        "groups": [{
            "anchor": ("top", 300), "x": 100, "width": 820, "align": "left",
            "blocks": [
                ("line", "", {}),
                ("light", f"сегодня у{NB}нас были и{NB}няня, и{NB}мой брат, и{NB}мама. причём все "
                          f"практически одновременно!", {}),
                ("bold", f"ну где вы все, когда у{NB}нас дома дурдом?!", {"color": CREAM, "size": 56}),
            ],
        }],
    },
    {   # elevator selfie: face upper-middle; short line on top, the rest over the bassinet
        "photo": f"{UPLOADS}/027c5d3e-image.png",
        "groups": [
            {
                "anchor": ("top", 270), "x": 100, "width": 820, "align": "left",
                "blocks": [
                    ("light", "сейчас вот наконец-то вышли погулять", {"size": 36}),
                ],
            },
            {
                "anchor": ("bottom", 1640), "x": 100, "width": 860, "align": "left",
                "blocks": [
                    ("bold", f"и, кстати, я{NB}та самая мама, которая будет гулять с{NB}ребёнком даже "
                             f"в{NB}минус тридцать", {"size": 46}),
                    ("note", f"обожаю прогулки и{NB}вообще не{NB}представляю, как можно всю зиму "
                             f"сидеть дома!", {}),
                ],
            },
        ],
    },
]

if __name__ == "__main__":
    os.makedirs(base.OUT, exist_ok=True)
    paths = [base.render_photo(s, i) for i, s in enumerate(SLIDES, 1)]
    print("\n".join(paths))
    from PIL import Image
    ims = [Image.open(p).resize((360, 640)) for p in paths]
    sheet = Image.new("RGB", (380 * len(ims) + 20, 680), (255, 255, 255))
    for k, im in enumerate(ims):
        sheet.paste(im, (20 + k * 380, 20))
    sheet.save(os.path.join(base.OUT, "00_overview.jpg"), quality=90)
