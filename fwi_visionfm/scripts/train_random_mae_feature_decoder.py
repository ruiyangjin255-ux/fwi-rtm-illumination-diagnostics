from __future__ import annotations

import argparse
import json
from pathlib import Path

from fwi_visionfm.scripts.train_local_mae_feature_decoder import train_local_mae_decoder_from_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train decoder from random MAE encoder features.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decoder", default="unet_decoder")
    parser.add_argument("--loss", default="default_l1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_local_mae_decoder_from_cache(
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        decoder_name=args.decoder,
        loss_name=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
