import torch  # pyrefly: ignore [missing-import]
import torch.nn as nn  # pyrefly: ignore [missing-import]


class ModulationClassifier1DCNN(nn.Module):
    """
    Kiến trúc 1D CNN 3 khối hoạt động trực tiếp trên tensor tín hiệu I/Q gốc [N_batch, 2, 128].
    Áp dụng chuẩn cho cả Mô hình Baseline (không Augmentation) và Mô hình Topic 4 (có Augmentation).
    """

    def __init__(self, num_classes: int = 8, dropout_rate: float = 0.3):
        super().__init__()

        # Khối Conv 1: Trích xuất đặc trưng bậc thấp (L=128 -> 64)
        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=64, kernel_size=7, stride=1, padding=3
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Khối Conv 2: Trích xuất biểu diễn thời gian-tần số (L=64 -> 32)
        self.block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Khối Conv 3: Trích xuất đặc trưng hình học chòm sao cao cấp (L=32 -> 1)
        self.block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Đầu Phân loại (Classifier Head)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  torch.Tensor shape [N_batch, 2, 128]
        Output: Logits tensor shape [N_batch, num_classes] (ví dụ: [N_batch, 8])
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = self.classifier(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Đếm tổng số tham số trainable của mô hình."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
