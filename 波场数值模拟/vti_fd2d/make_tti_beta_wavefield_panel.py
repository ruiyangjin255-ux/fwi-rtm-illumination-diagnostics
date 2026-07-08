from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont


SRC_DIR = Path(r"D:\ryjin\paper_figures_source\vti\tti_beta_series_vti_settings")
OUT_PATH = SRC_DIR / "tti_beta_wavefield_panel.png"

ORDER = [
    ("-90", -90),
    ("-60", -60),
    ("-30", -30),
    ("+0", 0),
    ("+30", 30),
    ("+60", 60),
]

files = [SRC_DIR / f"tti_beta_{tag}_wavefield_vx_vz.png" for tag, _ in ORDER]
for fp in files:
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")

imgs = [Image.open(fp).convert("RGB") for fp in files]

min_w = min(im.width for im in imgs)
min_h = min(im.height for im in imgs)
imgs = [ImageOps.fit(im, (min_w, min_h), method=Image.Resampling.LANCZOS) for im in imgs]

rows, cols = 3, 2
pad_x = 48
pad_y = 70
label_h = 104
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

font_paths = [
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\arialbd.ttf",
]
font = None
for fp in font_paths:
    try:
        font = ImageFont.truetype(fp, 98)
        break
    except OSError:
        pass
if font is None:
    font = ImageFont.load_default()

letters = "abcdef"
for idx, (im, (_, beta)) in enumerate(zip(imgs, ORDER)):
    r = idx // cols
    c = idx % cols
    x0 = margin_l + c * (cell_w + pad_x)
    y0 = margin_t + r * (cell_h + pad_y)
    canvas.paste(im, (x0, y0))

    beta_text = f"+{beta}" if beta > 0 else str(beta)
    label = f"({letters[idx]}) β={beta_text}°"
    text_bbox = draw.textbbox((0, 0), label, font=font)
    tw = text_bbox[2] - text_bbox[0]
    tx = x0 + (panel_w - tw) // 2
    ty = y0 + panel_h + 8
    draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

canvas.save(OUT_PATH, dpi=(300, 300))
print(str(OUT_PATH))
