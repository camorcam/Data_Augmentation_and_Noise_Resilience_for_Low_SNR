import torch  # pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]
from typing import Tuple, Dict

class WirelessIQDataset(Dataset):
    """
    PyTorch Dataset wrapper cho dữ liệu I/Q [N_batch, 2, 128].
    """
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, snr_data: np.ndarray, transform=None):
        """
        Tham số:
            x_data: np.ndarray shape [N, 2, 128]
            y_data: np.ndarray shape [N]
            snr_data: np.ndarray shape [N]
            transform: hàm transform (dùng cho Data Augmentation ở Phase 2)
        """
        self.x = torch.from_numpy(x_data).float()
        self.y = torch.from_numpy(y_data).long()
        self.snr = torch.from_numpy(snr_data).float()
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]
        snr_sample = self.snr[idx]

        if self.transform is not None:
            x_sample = self.transform(x_sample)

        return x_sample, y_sample, snr_sample

def split_dataset(
    dataset_dict: Dict[str, np.ndarray],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[WirelessIQDataset, WirelessIQDataset, WirelessIQDataset]:
    """
    Phân chia dữ liệu thành 70% Train, 15% Val, 15% Test với phân bổ ngẫu nhiên cố định seed.
    Đảm bảo Zero Data Leakage giữa tập Train, Val và Test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Tổng tỷ lệ phải bằng 1.0"
    
    x = dataset_dict['x']
    y = dataset_dict['y']
    snr = dataset_dict['snr']
    
    total_samples = len(x)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_samples)

    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_dataset = WirelessIQDataset(x[train_idx], y[train_idx], snr[train_idx])
    val_dataset = WirelessIQDataset(x[val_idx], y[val_idx], snr[val_idx])
    test_dataset = WirelessIQDataset(x[test_idx], y[test_idx], snr[test_idx])

    return train_dataset, val_dataset, test_dataset
