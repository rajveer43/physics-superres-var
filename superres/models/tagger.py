"""Small residual CNN used as jet tagger (qg), energy regressor (calo) and
classifier two-sample test. It adapts its down-sampling to the input grid, so
the same architecture runs on HR, every LR grid and every SR output."""
import torch
import torch.nn as nn

from .nn_utils import ConvNd, ResBlock, conv_out, norm


class Tagger(nn.Module):
    def __init__(self, ndim, in_ch, input_shape, widths, out_dim=1, dropout=0.1, circ=(), n_global=0):
        super().__init__()
        layers, c, cur = [ConvNd(ndim, in_ch, widths[0], circular_dims=circ)], widths[0], list(input_shape)
        for w in widths:
            layers.append(ResBlock(ndim, c, w, circ))
            c = w
            stride = tuple(2 if s >= 8 else 1 for s in cur)
            if any(st > 1 for st in stride):
                layers.append(ConvNd(ndim, c, c, 3, stride=stride, circular_dims=circ))
                cur = [conv_out(s, st) for s, st in zip(cur, stride)]
        pool = nn.AdaptiveAvgPool2d(1) if ndim == 2 else nn.AdaptiveAvgPool3d(1)
        layers += [norm(c), nn.SiLU(), pool, nn.Flatten()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c + n_global, c), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(c, out_dim))

    def forward(self, x, g=None):
        h = self.body(x)
        if g is not None:
            h = torch.cat([h, g], 1)
        return self.head(h)
