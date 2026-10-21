"""RF transforms with a private RNG, independent of data ordering and dropout."""

import math
import torch
from torch import nn


class RandomPhaseRotation(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x, generator=None):
        if torch.rand((), generator=generator).item() >= self.p:
            return x
        angle = 2 * math.pi * torch.rand((), generator=generator).item()
        i, q = x[..., 0, :], x[..., 1, :]
        return torch.stack(
            (
                i * math.cos(angle) - q * math.sin(angle),
                i * math.sin(angle) + q * math.cos(angle),
            ),
            dim=-2,
        )


class AdditiveGaussianNoise(nn.Module):
    """Reduce nominal SNR by U[0,max_drop_db], using expected signal power.

    x already contains AWGN. Signal power estimate = P(x)*rho/(1+rho).
    Added complex noise power = Psignal*(1/rho_target - 1/rho).
    Finite-burst realized SNR varies. No renormalization after augmentation.
    """

    def __init__(self, p=0.4, max_drop_db=6.0):
        super().__init__()
        self.p = p
        self.max_drop_db = max_drop_db
        if max_drop_db < 0:
            raise ValueError("max_drop_db must be nonnegative")

    def forward(self, x, snr_db, generator=None):
        if torch.rand((), generator=generator).item() >= self.p:
            return x
        if x.ndim != 2:
            raise ValueError("Noise augmentation expects a single [2,L] sample")
        rho = 10.0 ** (float(snr_db) / 10)
        drop = self.max_drop_db * torch.rand((), generator=generator).item()
        target = rho * 10 ** (-drop / 10)
        signal_power = x.square().sum(dim=0).mean() * rho / (1 + rho)
        added_power = signal_power * (1 / target - 1 / rho)
        noise = torch.randn(
            x.shape, generator=generator, dtype=x.dtype, device=x.device
        )
        return x + noise * torch.sqrt(torch.clamp(added_power / 2, min=0))


class TimeShiftJitter(nn.Module):
    def __init__(self, p=0.4, max_shift=8):
        super().__init__()
        self.p = p
        self.max_shift = max_shift

    def forward(self, x, generator=None):
        if torch.rand((), generator=generator).item() >= self.p:
            return x
        k = int(
            torch.randint(
                -self.max_shift, self.max_shift + 1, (), generator=generator
            ).item()
        )
        return torch.roll(x, k, dims=-1)


class AmplitudeScaling(nn.Module):
    def __init__(self, p=0.3, scale_min=0.85, scale_max=1.15):
        super().__init__()
        self.p = p
        self.lo = scale_min
        self.hi = scale_max

    def forward(self, x, generator=None):
        if torch.rand((), generator=generator).item() >= self.p:
            return x
        return x * (
            self.lo + (self.hi - self.lo) * torch.rand((), generator=generator).item()
        )


class RFAugmentationPipeline(nn.Module):
    def __init__(
        self,
        phase_prob=0.5,
        noise_prob=0.4,
        shift_prob=0.4,
        scale_prob=0.3,
        seed=0,
        max_drop_db=6,
    ):
        super().__init__()
        for v in (phase_prob, noise_prob, shift_prob, scale_prob):
            if not 0 <= v <= 1:
                raise ValueError("Probabilities must be in [0,1]")
        self.generator = torch.Generator().manual_seed(seed)
        self.phase_rot = RandomPhaseRotation(phase_prob)
        self.noise_inj = AdditiveGaussianNoise(noise_prob, max_drop_db)
        self.time_shift = TimeShiftJitter(shift_prob)
        self.amp_scale = AmplitudeScaling(scale_prob)

    def forward(self, x, snr_db=None):
        if x.device.type != "cpu" or x.ndim != 2:
            raise ValueError(
                "Use pipeline per CPU dataset sample [2,L], with num_workers=0"
            )
        g = self.generator
        x = self.phase_rot(x, g)
        if self.noise_inj.p > 0:
            if snr_db is None:
                raise ValueError("Nominal snr_db is required for calibrated noise")
            x = self.noise_inj(x, snr_db, g)
        x = self.time_shift(x, g)
        return self.amp_scale(x, g)


CONFIGS = {
    "baseline": dict(phase_prob=0, noise_prob=0, shift_prob=0, scale_prob=0),
    "phase_only": dict(phase_prob=0.5, noise_prob=0, shift_prob=0, scale_prob=0),
    "noise_only": dict(phase_prob=0, noise_prob=0.4, shift_prob=0, scale_prob=0),
    "shift_only": dict(phase_prob=0, noise_prob=0, shift_prob=0.4, scale_prob=0),
    "scale_only": dict(phase_prob=0, noise_prob=0, shift_prob=0, scale_prob=0.3),
    "augmented_full": dict(
        phase_prob=0.5, noise_prob=0.4, shift_prob=0.4, scale_prob=0.3
    ),
}
