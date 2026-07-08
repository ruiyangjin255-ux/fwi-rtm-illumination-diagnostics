from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


SRC = Path(r"D:\ryjin\paper_figures_source\vti\layer_thomsen_surface\vti_layer_thomsen_surface_record_2s_vx_vz.png")
OUT = SRC.with_name("vti_layer_thomsen_surface_record_2s_vx_vz_annotated.png")

img = Image.open(SRC).convert("RGB")
draw = ImageDraw.Draw(img)

font_paths = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
]
font = None
small_font = None
for fp in font_paths:
    try:
        font = ImageFont.truetype(fp, 48)
        small_font = ImageFont.truetype(fp, 42)
        break
    except OSError:
        pass
if font is None:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()


def label_box(text, xy, target, font_obj=None, box_pad=14):
    if font_obj is None:
        font_obj = font
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font_obj)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    rect = (x - box_pad, y - box_pad, x + w + box_pad, y + h + box_pad)
    draw.rectangle(rect, fill=(255, 255, 255), outline=(0, 0, 0), width=4)
    draw.text((x, y), text, fill=(0, 0, 0), font=font_obj)
    start = (rect[2], (rect[1] + rect[3]) // 2)
    if target[0] < x:
        start = (rect[0], (rect[1] + rect[3]) // 2)
    draw.line([start, target], fill=(0, 0, 0), width=4)
    draw.ellipse((target[0]-8, target[1]-8, target[0]+8, target[1]+8), fill=(0, 0, 0))


# Left panel: Vx. Strong direct qP, weaker reflected and converted events.
label_box("直达 qP 波", (955, 210), (1280, 285))
label_box("PP 反射波", (915, 770), (1310, 850), small_font)
label_box("PS 转换反射波", (815, 1175), (1220, 1210), small_font)

# Right panel: Vz. Direct qP is weak; PP and PS reflected events are dominant.
label_box("弱直达 qP 波", (2555, 250), (2790, 345), small_font)
label_box("PP 反射波", (2585, 760), (2825, 820))
label_box("PS 转换反射波", (2490, 1185), (2740, 1245), small_font)

img.save(OUT, dpi=(300, 300))
print(OUT)
