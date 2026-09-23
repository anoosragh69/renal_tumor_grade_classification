"""
Step 8: Baseline comparison models (plan §4 Step 8)
====================================================
Image-only 2D backbones from timm, trained under the exact Step 7d protocol
(same splits, same augmentation, same 200 epochs, same paper-Adam) for the
Table 3 comparison against the vViT.

    - ViT-S/16        (vit_small_patch16_224, augreg_in21k)
    - ConvNeXt-S      (convnext_small)
    - ResNeXt-50 32x4d(resnext50_32x4d, a1_in1k)

Adaptation deviations (paper does not specify; documented for plan §6):
    1. ImageNet-pretrained weights (paper silent on pretraining) — random init
       on ~9.5k tumour slices of 48 patients would be near-chance.
    2. Input is 1-channel 128x128 HU-windowed grayscale; timm backbones expect
       3-channel 224x224 -> stem conv folded to in_chans=1 by timm, input
       bilinearly resized 128 -> 224 inside ``TimmBaseline.forward``.
    3. Binary head: single BCE logit (matches vViT's per-head BCE protocol),
       not softmax/CE.

Supersedes the old Task-15 baselines (3D CNN / clinical MLP) — plan §8.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

# model_name -> (timm architecture name, short tag used for artifacts/logs)
BASELINE_MODELS = {
    "vit": "vit_small_patch16_224",
    "convnext": "convnext_small",
    "resnext": "resnext50_32x4d",
}

INPUT_SIZE = 128        # dataset crop
TIMM_INPUT_SIZE = 224   # native resolution of the pretrained backbones


class TimmBaseline(nn.Module):
    """timm 2D backbone -> single binary logit, image sector only.

    forward(batch_or_tensor): accepts either the sector-dict batch (uses the
    ``image`` key, (B,1,128,128)) or a raw (B,1,H,W) tensor.
    """

    def __init__(self, model_name: str = "vit", pretrained: bool = True):
        super().__init__()
        if model_name not in BASELINE_MODELS:
            raise ValueError(f"unknown baseline {model_name!r}; "
                             f"choose from {list(BASELINE_MODELS)}")
        import timm
        self.model_name = model_name
        self.backbone = timm.create_model(
            BASELINE_MODELS[model_name],
            pretrained=pretrained,
            in_chans=1,      # timm folds RGB stem filters to 1 channel
            num_classes=1,   # single BCE logit
        )

    def _images(self, x):
        if isinstance(x, dict):
            x = x["image"]
        return x

    def forward(self, x) -> torch.Tensor:
        x = self._images(x)                                    # (B,1,128,128)
        if x.shape[-1] != TIMM_INPUT_SIZE:
            x = F.interpolate(x, size=(TIMM_INPUT_SIZE, TIMM_INPUT_SIZE),
                              mode="bilinear", align_corners=False)
        return self.backbone(x).squeeze(-1)                    # (B,) logit


def _self_test():
    """Dummy-tensor forward/backward for each backbone (pretrained=False)."""
    print("=== Step 8 baselines self-test (scratch weights) ===")
    for name in BASELINE_MODELS:
        model = TimmBaseline(name, pretrained=False)
        x = torch.randn(2, 1, INPUT_SIZE, INPUT_SIZE)
        zb = {"image": x}
        out_t = model(x)
        out_d = model(zb)
        assert out_t.shape == out_d.shape == (2,), f"{name}: {out_t.shape}"
        loss = F.binary_cross_entropy_with_logits(
            out_d, torch.tensor([0.0, 1.0]))
        loss.backward()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:8s}: logits {tuple(out_t.shape)}, BCE {loss.item():.3f}, "
              f"params {n_params:,}")
    # pretrained weight download smoke (vit only, smallest)
    print("  pretrained download check (vit)...", flush=True)
    m = TimmBaseline("vit", pretrained=True)
    prob = torch.sigmoid(m(torch.rand(1, 1, INPUT_SIZE, INPUT_SIZE))).item()
    assert 0.0 <= prob <= 1.0
    print(f"  pretrained vit forward OK (sigmoid prob {prob:.3f})")
    print("Step 8 baselines self-test PASSED.")


if __name__ == "__main__":
    _self_test()
