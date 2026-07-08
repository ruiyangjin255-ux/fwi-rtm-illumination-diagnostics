from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.models.seismic_backbones.local_mae import LocalSeismicMAE
from fwi_visionfm.models.tokenizers.mae_patch_tokenizer import batch_bridge_images
from fwi_visionfm.torch_backend import require_torch_backend


def _iter_batches(images: np.ndarray, *, batch_size: int):
    for start in range(0, int(images.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(images.shape[0]))
        yield images[start:end]


def _evaluate(model: LocalSeismicMAE, images: np.ndarray, *, device: str, batch_size: int) -> float:
    torch = require_torch_backend()
    losses = []
    model.module.eval()
    with torch.no_grad():
        for batch in _iter_batches(images, batch_size=batch_size):
            out = model(torch.as_tensor(batch, dtype=torch.float32, device=device))
            losses.append(float(out["reconstruction_loss"].detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def _write_preview(model: LocalSeismicMAE, images: np.ndarray, output_path: Path, *, device: str) -> None:
    from PIL import Image, ImageDraw

    torch = require_torch_backend()
    batch = torch.as_tensor(images[: min(3, len(images))], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(batch)
    recon = out["reconstruction"].detach().cpu().numpy()
    truth = batch.detach().cpu().numpy()
    masked = out["masked_input"].detach().cpu().numpy()
    rows = []
    for pred, target, masked_input in zip(recon, truth, masked):
        tiles = []
        for title, image in (("masked", masked_input.mean(axis=0)), ("recon", pred.mean(axis=0)), ("target", target.mean(axis=0))):
            image = np.asarray(image, dtype=np.float32)
            lo = float(image.min())
            hi = float(image.max())
            scaled = np.zeros_like(image, dtype=np.uint8) if hi == lo else np.clip((image - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            tile = Image.fromarray(scaled, mode="L").convert("RGB")
            canvas = Image.new("RGB", (tile.width, tile.height + 18), "white")
            canvas.paste(tile, (0, 18))
            ImageDraw.Draw(canvas).text((2, 2), title, fill=(0, 0, 0))
            tiles.append(canvas)
        row = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width
        rows.append(row)
    grid = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height
    grid.save(output_path)


def run_local_mae_pretrain(
    *,
    train_paths: list[str | Path],
    val_paths: list[str | Path],
    output_dir: str | Path,
    bridge: str,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    mask_type: str = "random_patch",
    input_size: int = 64,
    patch_size: int = 8,
    embed_dim: int = 128,
    depth: int = 4,
    num_heads: int = 4,
    decoder_embed_dim: int = 64,
    decoder_depth: int = 2,
    decoder_heads: int = 4,
    mask_ratio: float = 0.75,
) -> dict[str, Any]:
    torch = require_torch_backend()
    torch.manual_seed(int(seed))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if mask_type == "frequency_band" and bridge == "raw_envelope_spectrum3":
        result = {"status": "SKIPPED_INCOMPATIBLE_MASK", "skip_reason": "frequency_band mask is reserved for spectrogram-like inputs."}
        (output_root / "config.json").write_text(json.dumps({"bridge": bridge, "mask_type": mask_type, **result}, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_root / "run_log.txt").write_text(result["skip_reason"] + "\n", encoding="utf-8")
        return result
    train_payload = batch_bridge_images(train_paths, bridge, output_size=input_size)
    val_payload = batch_bridge_images(val_paths, bridge, output_size=input_size)
    model = LocalSeismicMAE(
        input_size=input_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
        mask_ratio=mask_ratio,
        mask_type=mask_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, int(epochs) + 1):
        model.module.train()
        batch_losses = []
        for batch in _iter_batches(train_payload["images"], batch_size=batch_size):
            x = torch.as_tensor(batch, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            out = model(x)
            loss = out["reconstruction_loss"]
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        val_loss = _evaluate(model, val_payload["images"], device=device, batch_size=batch_size)
        history.append({"epoch": epoch, "mask_type": mask_type, "train_reconstruction_loss": float(np.mean(batch_losses)) if batch_losses else 0.0, "val_reconstruction_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "encoder_state": model.module.state_dict(),
                    "bridge": bridge,
                    "seed": int(seed),
                    "input_size": int(input_size),
                    "patch_size": int(patch_size),
                    "embed_dim": int(embed_dim),
                    "depth": int(depth),
                    "num_heads": int(num_heads),
                    "decoder_embed_dim": int(decoder_embed_dim),
                    "decoder_depth": int(decoder_depth),
                    "decoder_heads": int(decoder_heads),
                    "mask_ratio": float(mask_ratio),
                },
                output_root / "best_mae_encoder.pt",
            )
    torch.save({"model_state": model.module.state_dict(), "bridge": bridge}, output_root / "last_mae_model.pt")
    with (output_root / "pretrain_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    metrics = {
        "reconstruction_loss": float(best_loss),
        "mask_ratio": float(mask_ratio),
        "mask_type": mask_type,
        "bridge": bridge,
        "input_shape": list(train_payload["images"].shape[1:]),
        "patch_size": int(patch_size),
        "epoch_best": int(best_epoch),
    }
    (output_root / "pretrain_val_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_preview(model, val_payload["images"], output_root / "reconstruction_preview_grid.png", device=device)
    config = {
        "bridge": bridge,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "mask_ratio": float(mask_ratio),
        "mask_type": mask_type,
        "input_size": int(input_size),
        "patch_size": int(patch_size),
        "embed_dim": int(embed_dim),
        "status": "SUCCESS",
    }
    (output_root / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "run_log.txt").write_text(f"status=SUCCESS\nbest_epoch={best_epoch}\n", encoding="utf-8")
    return {"status": "SUCCESS", "reconstruction_loss": float(best_loss), "output_dir": str(output_root)}
