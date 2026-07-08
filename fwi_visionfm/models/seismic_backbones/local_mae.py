from __future__ import annotations

import math
from typing import Any


def _require_torch():
    import torch

    return torch


def _build_2d_sincos_position_embedding(embed_dim: int, grid_size: int):
    torch = _require_torch()
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim for 2D sin-cos position embedding must be divisible by 4")
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(grid_h, grid_w, indexing="ij")
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / max(embed_dim // 4, 1)
    omega = 1.0 / (10000**omega)
    out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=1)
    return pos.unsqueeze(0)


class PatchEmbed:
    def __init__(self, *, input_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        torch = _require_torch()
        nn = torch.nn
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(int(in_chans), int(embed_dim), kernel_size=self.patch_size, stride=self.patch_size)

    def __call__(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerStack:
    def __init__(self, *, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float) -> None:
        torch = _require_torch()
        nn = torch.nn
        layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(num_heads),
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.module = nn.TransformerEncoder(layer, num_layers=int(depth))

    def __call__(self, x):
        return self.module(x)


class LocalSeismicMAEEncoder:
    def __init__(
        self,
        *,
        input_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        pos_embed_mode: str = "sincos",
    ) -> None:
        torch = _require_torch()
        nn = torch.nn
        self.patch_embed = PatchEmbed(input_size=input_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        if pos_embed_mode == "sincos":
            pos = _build_2d_sincos_position_embedding(embed_dim, self.patch_embed.grid_size)
            self.pos_embed = nn.Parameter(pos, requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = TransformerStack(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, x, mask=None):
        torch = _require_torch()
        tokens = self.patch_embed(x) + self.pos_embed.to(x.device)
        if mask is not None:
            tokens = tokens * (1.0 - mask.unsqueeze(-1).to(tokens.dtype))
        latent = self.norm(self.blocks(tokens))
        pooled = latent.mean(dim=1)
        return latent, pooled


class LocalSeismicMAEDecoder:
    def __init__(
        self,
        *,
        num_patches: int,
        patch_dim: int,
        encoder_embed_dim: int = 128,
        decoder_embed_dim: int = 64,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        torch = _require_torch()
        nn = torch.nn
        self.decoder_embed = nn.Linear(int(encoder_embed_dim), int(decoder_embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(decoder_embed_dim)))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, int(num_patches), int(decoder_embed_dim)))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = TransformerStack(embed_dim=decoder_embed_dim, depth=decoder_depth, num_heads=decoder_heads, mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(int(decoder_embed_dim))
        self.head = nn.Linear(int(decoder_embed_dim), int(patch_dim))

    def __call__(self, latent, mask):
        tokens = self.decoder_embed(latent)
        mask_tokens = self.mask_token.expand(tokens.shape[0], tokens.shape[1], -1)
        tokens = tokens * (1.0 - mask.unsqueeze(-1).to(tokens.dtype)) + mask_tokens * mask.unsqueeze(-1).to(tokens.dtype)
        decoded = self.norm(self.blocks(tokens + self.pos_embed.to(tokens.device)))
        return self.head(decoded)


class LocalSeismicMAE:
    def __init__(
        self,
        *,
        input_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 64,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        mask_ratio: float = 0.75,
        mask_type: str = "random_patch",
        pos_embed_mode: str = "sincos",
    ) -> None:
        torch = _require_torch()
        nn = torch.nn

        class _Module(nn.Module):
            def __init__(self, outer: "LocalSeismicMAE") -> None:
                super().__init__()
                self.outer = outer
                self.encoder_pos_embed = outer.encoder.pos_embed
                self.patch_proj = outer.encoder.patch_embed.proj
                self.encoder_blocks = outer.encoder.blocks.module
                self.encoder_norm = outer.encoder.norm
                self.decoder_embed = outer.decoder.decoder_embed
                self.mask_token = outer.decoder.mask_token
                self.decoder_pos_embed = outer.decoder.pos_embed
                self.decoder_blocks = outer.decoder.blocks.module
                self.decoder_norm = outer.decoder.norm
                self.decoder_head = outer.decoder.head

            def forward(self, x):
                return self.outer._forward_impl(self, x)

        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.mask_ratio = float(mask_ratio)
        self.mask_type = str(mask_type)
        self.encoder = LocalSeismicMAEEncoder(
            input_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pos_embed_mode=pos_embed_mode,
        )
        patch_dim = self.patch_size * self.patch_size * self.in_chans
        self.decoder = LocalSeismicMAEDecoder(
            num_patches=self.encoder.patch_embed.num_patches,
            patch_dim=patch_dim,
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            mlp_ratio=mlp_ratio,
        )
        self.module = _Module(self)

    def patchify(self, images):
        p = self.patch_size
        b, c, h, w = images.shape
        if h != self.input_size or w != self.input_size:
            raise ValueError(f"expected input_size={self.input_size}, got {(h, w)}")
        x = images.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(b, -1, p * p * c)

    def unpatchify(self, patches):
        torch = _require_torch()
        p = self.patch_size
        b, n, dim = patches.shape
        side = int(math.sqrt(n))
        c = dim // (p * p)
        x = patches.reshape(b, side, side, p, p, c).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(b, c, side * p, side * p)

    def random_mask(self, batch_size: int, num_patches: int, device: str):
        torch = _require_torch()
        keep = max(1, int(round(num_patches * (1.0 - self.mask_ratio))))
        noise = torch.rand(batch_size, num_patches, device=device)
        ids = noise.argsort(dim=1)
        mask = torch.ones(batch_size, num_patches, device=device)
        mask.scatter_(1, ids[:, :keep], 0.0)
        return mask

    def structured_mask(self, x):
        torch = _require_torch()
        b, _, _, _ = x.shape
        num_patches = self.encoder.patch_embed.num_patches
        side = int(math.sqrt(num_patches))
        if self.mask_type == "random_patch":
            return self.random_mask(b, num_patches, x.device)
        if self.mask_type == "time_block":
            width = max(1, int(round(side * self.mask_ratio)))
            start = torch.randint(0, max(side - width + 1, 1), (b,), device=x.device)
            mask2d = torch.zeros((b, side, side), device=x.device)
            for i in range(b):
                mask2d[i, :, start[i] : start[i] + width] = 1.0
            return mask2d.reshape(b, -1)
        if self.mask_type == "receiver_block":
            height = max(1, int(round(side * self.mask_ratio)))
            start = torch.randint(0, max(side - height + 1, 1), (b,), device=x.device)
            mask2d = torch.zeros((b, side, side), device=x.device)
            for i in range(b):
                mask2d[i, start[i] : start[i] + height, :] = 1.0
            return mask2d.reshape(b, -1)
        if self.mask_type == "trace_dropout":
            cols = max(1, int(round(side * self.mask_ratio)))
            mask2d = torch.zeros((b, side, side), device=x.device)
            for i in range(b):
                selected = torch.randperm(side, device=x.device)[:cols]
                mask2d[i, :, selected] = 1.0
            return mask2d.reshape(b, -1)
        if self.mask_type == "hybrid_seismic_mask":
            base = self.random_mask(b, num_patches, x.device).reshape(b, side, side)
            recv = self.structured_mask(x.new_zeros(x.shape).copy_(x)) if False else None
            time_mask = torch.zeros((b, side, side), device=x.device)
            recv_mask = torch.zeros((b, side, side), device=x.device)
            tw = max(1, int(round(side * self.mask_ratio * 0.5)))
            rh = max(1, int(round(side * self.mask_ratio * 0.5)))
            for i in range(b):
                ts = torch.randint(0, max(side - tw + 1, 1), (1,), device=x.device).item()
                rs = torch.randint(0, max(side - rh + 1, 1), (1,), device=x.device).item()
                time_mask[i, :, ts : ts + tw] = 1.0
                recv_mask[i, rs : rs + rh, :] = 1.0
            return torch.clamp(base + time_mask + recv_mask, 0.0, 1.0).reshape(b, -1)
        if self.mask_type == "frequency_band":
            band = max(1, side // 3)
            mask2d = torch.zeros((b, side, side), device=x.device)
            for i in range(b):
                start = torch.randint(0, max(side - band + 1, 1), (1,), device=x.device).item()
                mask2d[i, start : start + band, :] = 1.0
            return mask2d.reshape(b, -1)
        raise ValueError(f"unsupported mask_type: {self.mask_type}")

    def mask_to_image(self, mask):
        torch = _require_torch()
        side = int(math.sqrt(mask.shape[1]))
        patch_mask = mask.reshape(mask.shape[0], side, side).unsqueeze(1)
        expanded = torch.nn.functional.interpolate(patch_mask, size=(self.input_size, self.input_size), mode="nearest")
        return expanded

    def _forward_impl(self, wrapper: Any, x):
        torch = _require_torch()
        mask = self.structured_mask(x)
        tokens = wrapper.patch_proj(x).flatten(2).transpose(1, 2) + wrapper.encoder_pos_embed.to(x.device)
        latent = wrapper.encoder_norm(wrapper.encoder_blocks(tokens * (1.0 - mask.unsqueeze(-1))))
        pooled = latent.mean(dim=1)
        decoded = wrapper.decoder_embed(latent)
        mask_tokens = wrapper.mask_token.expand(decoded.shape[0], decoded.shape[1], -1)
        decoded = decoded * (1.0 - mask.unsqueeze(-1)) + mask_tokens * mask.unsqueeze(-1)
        decoded = wrapper.decoder_norm(wrapper.decoder_blocks(decoded + wrapper.decoder_pos_embed.to(x.device)))
        pred_patches = wrapper.decoder_head(decoded)
        target_patches = self.patchify(x)
        loss = ((pred_patches - target_patches) ** 2)
        masked_loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum().clamp_min(1.0) / pred_patches.shape[-1]
        reconstruction = self.unpatchify(pred_patches)
        mask_image = self.mask_to_image(mask)
        masked_input = x * (1.0 - mask_image)
        return {
            "reconstruction": reconstruction,
            "mask": mask,
            "masked_input": masked_input,
            "latent_features": pooled,
            "reconstruction_loss": masked_loss,
        }

    def encode_features(self, x):
        with _require_torch().no_grad():
            _, pooled = self.encoder(x, mask=None)
        return pooled

    def __call__(self, x):
        return self.module(x)

    def to(self, device: str):
        self.module.to(device)
        return self

    def parameters(self):
        return self.module.parameters()
