"""Multi-scale residual VQ-VAE (VAR paper, Algorithms 1 & 2).

The encoder maps a log-energy image to a latent map f. f is quantised into K
token maps r_1..r_K of increasing size with a shared codebook; each step
quantises what the previous scales have not explained yet (residual design).
The perceptual (LPIPS) and GAN losses of eq. (5) are dropped: they are tuned for
natural images and mean nothing for energy deposits. Instead a total-energy term
is added in the trainer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import ConvNd, ResBlock, Upsample, norm


def down(x, size):
    size = tuple(size)
    return x if tuple(x.shape[2:]) == size else F.interpolate(x, size=size, mode="area")


def up(x, size):
    size = tuple(size)
    if tuple(x.shape[2:]) == size:
        return x
    mode = "bicubic" if x.dim() == 4 else "trilinear"
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


class Encoder(nn.Module):
    def __init__(self, ndim, in_ch, widths, strides, z_ch, circ=(), n_res=1):
        super().__init__()
        assert len(widths) == len(strides) + 1, "need one more width than strides"
        layers, c = [ConvNd(ndim, in_ch, widths[0], circular_dims=circ)], widths[0]
        for i, w in enumerate(widths):
            for _ in range(n_res):
                layers.append(ResBlock(ndim, c, w, circ))
                c = w
            if i < len(strides):
                layers.append(ConvNd(ndim, c, c, 3, stride=tuple(strides[i]), circular_dims=circ))
        layers += [ResBlock(ndim, c, c, circ), norm(c), nn.SiLU(), ConvNd(ndim, c, z_ch, 3, circular_dims=circ)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, ndim, out_ch, widths, strides, z_ch, circ=(), n_res=1):
        super().__init__()
        c = widths[-1]
        layers = [ConvNd(ndim, z_ch, c, 3, circular_dims=circ), ResBlock(ndim, c, c, circ)]
        for i in reversed(range(len(widths))):
            for _ in range(n_res):
                layers.append(ResBlock(ndim, c, widths[i], circ))
                c = widths[i]
            if i > 0:
                layers.append(Upsample(ndim, c, strides[i - 1], circ))
        layers += [norm(c), nn.SiLU(), ConvNd(ndim, c, out_ch, 3, circular_dims=circ)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Phi(nn.Module):
    """phi_k in Algorithm 1: residual conv applied after up-interpolation."""

    def __init__(self, ndim, C, circ=(), ratio=0.5):
        super().__init__()
        self.conv = ConvNd(ndim, C, C, 3, circular_dims=circ)
        self.r = ratio

    def forward(self, h):
        return h * (1 - self.r) + self.conv(h) * self.r


class MultiScaleVQ(nn.Module):
    def __init__(self, ndim, V, C, scales, beta=0.25, circ=()):
        super().__init__()
        self.scales = [tuple(s) for s in scales]
        self.V, self.C, self.beta = V, C, beta
        self.embedding = nn.Embedding(V, C)
        nn.init.uniform_(self.embedding.weight, -1.0 / V, 1.0 / V)
        self.phis = nn.ModuleList(Phi(ndim, C, circ) for _ in self.scales)
        self.register_buffer("usage", torch.zeros(V))

    def _nearest(self, z):
        B, s = z.shape[0], z.shape[2:]
        flat = z.movedim(1, -1).reshape(-1, self.C).float()
        e = self.embedding.weight.float()
        d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ e.t() + e.pow(2).sum(1)[None]
        return d.argmin(1).view(B, *s)

    def embed(self, idx):
        return self.embedding(idx).movedim(-1, 1)

    def forward(self, f):
        final = f.shape[2:]
        f_ng = f.detach()
        f_rest, f_hat = f_ng.clone(), torch.zeros_like(f_ng)
        loss = 0.0
        for k, s in enumerate(self.scales):
            idx = self._nearest(down(f_rest, s))
            if self.training:
                counts = torch.bincount(idx.flatten(), minlength=self.V).float()
                self.usage.mul_(0.99).add_(counts, alpha=0.01)
            h = self.phis[k](up(self.embed(idx), final))
            f_hat = f_hat + h
            f_rest = f_rest - h.detach()
            loss = loss + F.mse_loss(f_hat.detach(), f) * self.beta + F.mse_loss(f_hat, f_ng)
        loss = loss / len(self.scales)
        f_hat = (f_hat.detach() - f_ng) + f  # straight-through to the encoder
        return f_hat, loss

    @torch.no_grad()
    def f_to_idx(self, f):
        final, f_rest, out = f.shape[2:], f.clone(), []
        for k, s in enumerate(self.scales):
            idx = self._nearest(down(f_rest, s))
            out.append(idx)
            f_rest = f_rest - self.phis[k](up(self.embed(idx), final))
        return out

    def idx_to_fhat(self, idx_list):
        final = self.scales[-1]
        f_hat = 0
        for k, idx in enumerate(idx_list):
            f_hat = f_hat + self.phis[k](up(self.embed(idx), final))
        return f_hat

    def idx_to_var_input(self, idx_list):
        """Teacher-forcing inputs for scales 2..K: (B, L - 1, C)."""
        final = self.scales[-1]
        f_hat, xs = 0, []
        for k in range(len(self.scales) - 1):
            f_hat = f_hat + self.phis[k](up(self.embed(idx_list[k]), final))
            xs.append(down(f_hat, self.scales[k + 1]).flatten(2).transpose(1, 2))
        return torch.cat(xs, 1)

    def next_input(self, k, f_hat, idx):
        f_hat = f_hat + self.phis[k](up(self.embed(idx), self.scales[-1]))
        if k == len(self.scales) - 1:
            return f_hat, None
        return f_hat, down(f_hat, self.scales[k + 1]).flatten(2).transpose(1, 2)

    @torch.no_grad()
    def init_from(self, f):
        """Initialise the codebook with random encoder feature vectors."""
        flat = f.movedim(1, -1).reshape(-1, self.C).float()
        pick = flat[torch.randint(len(flat), (self.V,), device=flat.device)]
        self.embedding.weight.data.copy_(pick + 0.01 * pick.std() * torch.randn_like(pick))
        self.usage.fill_(1.0)

    @torch.no_grad()
    def reinit_dead(self, f, thresh):
        dead = self.usage < thresh
        n = int(dead.sum())
        if n:
            flat = f.movedim(1, -1).reshape(-1, self.C).float()
            pick = flat[torch.randint(len(flat), (n,), device=flat.device)]
            self.embedding.weight.data[dead] = pick + 0.01 * pick.std() * torch.randn_like(pick)
            self.usage[dead] = 1.0
        return n


class VQVAE(nn.Module):
    def __init__(self, ndim, in_ch, cfg, circ=()):
        super().__init__()
        z = cfg["z_ch"]
        self.enc = Encoder(ndim, in_ch, cfg["widths"], cfg["strides"], z, circ, cfg["n_res"])
        self.quant_conv = ConvNd(ndim, z, z, 3, circular_dims=circ)
        self.vq = MultiScaleVQ(ndim, cfg["codebook_size"], z, cfg["scales"], cfg["beta"], circ)
        self.post_quant_conv = ConvNd(ndim, z, z, 3, circular_dims=circ)
        self.dec = Decoder(ndim, in_ch, cfg["widths"], cfg["strides"], z, circ, cfg["n_res"])

    def encode(self, x):
        f = self.quant_conv(self.enc(x))
        if tuple(f.shape[2:]) != self.vq.scales[-1]:
            raise ValueError(f"latent {tuple(f.shape[2:])} != last scale {self.vq.scales[-1]}")
        return f

    def decode_fhat(self, f_hat):
        """Raw decoder output in log-energy space; clamp at 0 before use."""
        return self.dec(self.post_quant_conv(f_hat))

    def forward(self, x):
        f = self.encode(x)
        f_hat, vq_loss = self.vq(f)
        return self.decode_fhat(f_hat), vq_loss, f.detach()

    @torch.no_grad()
    def img_to_idx(self, x):
        return self.vq.f_to_idx(self.encode(x))

    @torch.no_grad()
    def idx_to_img(self, idx_list):
        return self.decode_fhat(self.vq.idx_to_fhat(idx_list))
