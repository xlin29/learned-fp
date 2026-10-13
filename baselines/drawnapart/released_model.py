# DRAWNAPART code, redistributed with the authors' permission.
#
# The network in this file is a layer-for-layer PyTorch transcription of
# DRAWNAPART's get_clf_model(), from their released code:
#
#     https://github.com/drawnapart/drawnapart
#     fpstalker_drawnapart.ipynb, commit bddd4f68fed265078c63edcf8f72e602f43084f9
#
# and included here with the written permission of its authors (September
# 2026), on the condition that this link and the following attribution
# accompany it:
#
#     Tomer Laor, Naif Mehanna, Vitaly Dyadyuk, Antonin Durey, Pierre
#     Laperdrix, Clémentine Maurice, Yossi Oren, Romain Rouvoy, Walter
#     Rudametkin, and Yuval Yarom. "DRAWN APART: A Device Identification
#     Technique based on Remote GPU Fingerprinting." Network and Distributed
#     System Security Symposium (NDSS), 2022.
#
# The transcription itself is part of the LearnedFP artifact and is
# MIT-licensed with it.
"""DRAWNAPART's released network, in PyTorch (their get_clf_model()):

    Input((32, 32, 1))
    Conv2D(128, (4,4), activation='relu'); Dropout(0.119510); AveragePooling2D()   x3
    Flatten()
    Dense(256, activation='relu')
    Dense(256, activation=None)
    Lambda(l2_normalize)
    Dense(C, activation='softmax')

Convolutions do not pad (32 -> 29 -> 14 -> 11 -> 5 -> 2 -> 1) and the
classifier reads the L2-normalized embedding. Trained in Keras by
dp_keras_baseline.py; weights arrive through dp_keras_to_torch.py.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCKS = 3
CHANNELS = 128
KSIZE = 4
DROPOUT = 0.119510
EMBED = 256


class ReleasedDPCNN(nn.Module):
    """DRAWNAPART's classifier / embedding network. The architecture arguments
    are accepted for interface compatibility and ignored; the released model
    fixes them."""

    def __init__(self, nclasses: int, embed_dim: int = EMBED, blocks: int = BLOCKS,
                 channels: int = CHANNELS, ksize: int = KSIZE,
                 dropout: float = DROPOUT, activation: str = "relu"):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = 1
        for _ in range(BLOCKS):
            layers += [
                nn.Conv2d(in_ch, CHANNELS, KSIZE),      # valid padding, as Keras
                nn.ReLU(inplace=True),                  # Conv2D(activation='relu')
                nn.Dropout(DROPOUT),                    # element-wise, not Dropout2d
                nn.AvgPool2d(2),
            ]
            in_ch = CHANNELS
        layers.append(nn.Flatten())
        self.backbone = nn.Sequential(*layers)

        self.proj = nn.Sequential(
            nn.Linear(CHANNELS, EMBED),
            nn.ReLU(inplace=True),
            nn.Linear(EMBED, EMBED),
        )
        self.cls = nn.Linear(EMBED, nclasses)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.backbone(x)), p=2, dim=1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        # Keras puts the softmax layer after the l2_normalize lambda
        return self.cls(self.embedding(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)


def build_from_arch(arch: dict, nclasses: int, device: torch.device) -> ReleasedDPCNN:
    """Reconstruct the network from a dp_arch_params.json blob.

    Only nclasses and embed_dim are read; the rest of the architecture is
    fixed by the released model.
    """
    net = ReleasedDPCNN(
        nclasses=int(arch.get("nclasses", nclasses)),
        embed_dim=int(arch.get("embed_dim", EMBED)),
    )
    return net.to(device)


def selftest() -> None:
    net = ReleasedDPCNN(nclasses=17)
    x = torch.randn(4, 1, 32, 32)
    h = net.backbone(x)
    assert h.shape == (4, CHANNELS), f"backbone output {tuple(h.shape)} != (4, {CHANNELS})"
    assert net.embedding(x).shape == (4, EMBED)
    assert net.logits(x).shape == (4, 17)
    n = sum(p.numel() for p in net.parameters())
    print(f"[selftest] ok  backbone->{tuple(h.shape)}  params={n:,}")


if __name__ == "__main__":
    selftest()
