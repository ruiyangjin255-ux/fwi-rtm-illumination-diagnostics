from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont


SRC_DIR = Path(r"D:\ryjin\paper_figures_source\vti\c13_series_wavefield_only")
OUT_PATH = Path(r"D:\ryjin\paper_figures_source\vti\c13_series_wavefield_only\vti_c13_wavefield_panel_2x4_no1p5.png")

# Keep 8 panels: remove +1.5
ORDER = [
    ("m17p5", -17.5),
    ("m10", -10.0),
    ("m5p8", -5.8),
    ("p0", 0.0),
    ("p3", 3.0),
    ("p5p8", 5.8),
    ("p10", 10.0),
    ("p17p5", 17.5),
]

files = [SRC_DIR / f"vti_c13_{tag}_wavefield_vx_vz.png" for tag, _ in ORDER]
for fp in files:
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")

imgs = [Image.open(fp).convert("RGB") for fp in files]

# normalize size to the smallest canvas
min_w = min(im.width for im in imgs)
min_h = min(im.height for im in imgs)
imgs = [ImageOps.fit(im, (min_w, min_h), method=Image.Resampling.LANCZOS) for im in imgs]

rows, cols = 4, 2
pad_x = 48
pad_y = 64
label_h = 96
margin_l = 36
margin_r = 36
margin_t = 36
margin_b = 36

panel_w = min_w
panel_h = min_h
cell_w = panel_w
cell_h = panel_h + label_h

canvas_w = margin_l + cols * cell_w + (cols - 1) * pad_x + margin_r
canvas_h = margin_t + rows * cell_h + (rows - 1) * pad_y + margin_b

canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("arialbd.ttf", 98)
except OSError:
    font = ImageFont.load_default()

letters = "abcdefgh"

for idx, (im, (_, c13)) in enumerate(zip(imgs, ORDER)):
    r = idx // cols
    c = idx % cols
    x0 = margin_l + c * (cell_w + pad_x)
    y0 = margin_t + r * (cell_h + pad_y)
    canvas.paste(im, (x0, y0))

    label = f"({letters[idx]}) c13={c13:g}"
    text_bbox = draw.textbbox((0, 0), label, font=font)
    tw = text_bbox[2] - text_bbox[0]
    tx = x0 + (panel_w - tw) // 2
    ty = y0 + panel_h + 8
    draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

canvas.save(OUT_PATH, dpi=(300, 300))
print(str(OUT_PATH))
