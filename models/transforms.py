import torch  # pyrefly: ignore [missing-import]
import torch.nn as nn  # pyrefly: ignore [missing-import]
import math
import numpy as np  # pyrefly: ignore [missing-import]

class RandomPhaseRotation(nn.Module):
    """
    Biến đổi Xoay pha ngẫu nhiên (Random Phase Rotation):
    Xoay góc pha theta in [0, 2pi) ngẫu nhiên cho tín hiệu I/Q.
    I' = I * cos(theta) - Q * sin(theta)
    Q' = I * sin(theta) + Q * cos(theta)
    """
    def __init__(self, p: float = 0.8):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor shape [2, 128] hoặc [N_batch, 2, 128]
        """
        if torch.rand(1).item() > self.p:
            return x

        theta = torch.rand(1).item() * 2 * math.pi
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        if x.dim() == 2:
            i_chan = x[0, :]
            q_chan = x[1, :]
            i_rot = i_chan * cos_t - q_chan * sin_t
            q_rot = i_chan * sin_t + q_chan * cos_t
            return torch.stack([i_rot, q_rot], dim=0)
        else:
            i_chan = x[:, 0, :]
            q_chan = x[:, 1, :]
            i_rot = i_chan * cos_t - q_chan * sin_t
            q_rot = i_chan * sin_t + q_chan * cos_t
            return torch.stack([i_rot, q_rot], dim=1)

class AdditiveGaussianNoise(nn.Module):
    """
    Tiêm thêm Nhiễu Gaussian ngẫu nhiên (Additive Gaussian Noise Injection):
    Tăng cường độ bền vững ở vùng SNR thấp (-10 dB đến +2 dB).
    """
    def __init__(self, std_min: float = 0.01, std_max: float = 0.08, p: float = 0.5):
        super().__init__()
        self.std_min = std_min
        self.std_max = std_max
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        std = torch.empty(1).uniform_(self.std_min, self.std_max).item()
        noise = torch.randn_like(x) * std
        return x + noise

class TimeShiftJitter(nn.Module):
    """
    Jitter dịch thời gian ngẫu nhiên (Time-shift Jitter):
    Dịch vòng tín hiệu dọc theo trục thời gian (sample length = 128).
    """
    def __init__(self, max_shift: int = 8, p: float = 0.5):
        super().__init__()
        self.max_shift = max_shift
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        if shift == 0:
            return x
        
        return torch.roll(x, shifts=shift, dims=-1)

class AmplitudeScaling(nn.Module):
    """
    Co giãn biên độ ngẫu nhiên (Amplitude Scaling):
    Nội suy biên độ trong khoảng [scale_min, scale_max].
    """
    def __init__(self, scale_min: float = 0.85, scale_max: float = 1.15, p: float = 0.5):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        scale = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()
        return x * scale

class RFAugmentationPipeline(nn.Module):
    """
    Pipeline tổng hợp tất cả các phép Data Augmentation trên miền RF vật lý cho Topic 4.
    """
    def __init__(
        self,
        phase_prob: float = 0.8,
        noise_prob: float = 0.7,
        shift_prob: float = 0.5,
        scale_prob: float = 0.5
    ):
        super().__init__()
        self.phase_rot = RandomPhaseRotation(p=phase_prob)
        self.noise_inj = AdditiveGaussianNoise(p=noise_prob)
        self.time_shift = TimeShiftJitter(p=shift_prob)
        self.amp_scale = AmplitudeScaling(p=scale_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.phase_rot(x)
        x = self.noise_inj(x)
        x = self.time_shift(x)
        x = self.amp_scale(x)
        return x
