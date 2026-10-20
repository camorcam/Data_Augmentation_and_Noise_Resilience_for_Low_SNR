"""
Script Huấn luyện & Thực nghiệm So sánh cho Giai đoạn 3 (Phase 3):
- Huấn luyện Mô hình A: Baseline 1D CNN (không dùng Data Augmentation)
- Huấn luyện Mô hình B: Augmented 1D CNN (dùng RFAugmentationPipeline cho Topic 4)
- Lưu weights vào thư mục checkpoint/ (baseline_model.pth & augmented_model.pth)
"""
import sys
import os
import time
import json
import torch  # pyrefly: ignore [missing-import]
import torch.nn as nn  # pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]

# Cấu hình UTF-8 cho Windows console output
sys.stdout.reconfigure(encoding='utf-8')

# Import các module đã tạo ở Phase 1 và Phase 2
from dataset import (  # pyrefly: ignore [missing-import]
    generate_synthetic_dataset,
    WirelessIQDataset,
    MODULATION_CLASSES,
    SNR_GRID
)
from dataset.pytorch_dataset import split_dataset  # pyrefly: ignore [missing-import]
from models import (  # pyrefly: ignore [missing-import]
    ModulationClassifier1DCNN,
    RFAugmentationPipeline
)

def set_seed(seed: int = 42):
    """Cố định random seed để đảm bảo 100% khả năng tái tạo kết quả (Reproducibility)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Huấn luyện 1 epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch, _ in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(y_batch)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """Đánh giá mô hình trên tập Validation hoặc Test."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            running_loss += loss.item() * len(y_batch)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model(
    model_name: str,
    train_dataset: WirelessIQDataset,
    val_dataset: WirelessIQDataset,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.001,
    save_dir: str = 'checkpoint'
):
    """
    Hàm chính huấn luyện mô hình với thanh tiến trình sạch sẽ.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n⚡ Đang huấn luyện [{model_name}] trên thiết bị: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}.pth")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ModulationClassifier1DCNN(num_classes=len(MODULATION_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Lưu checkpoint có Val Acc tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_path)
            is_best = "★ [BEST]"
        else:
            is_best = ""

        if epoch % 5 == 0 or epoch == epochs or is_best:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% {is_best}")

    total_time = time.time() - start_time
    print(f"  ✓ Hoàn tất huấn luyện [{model_name}] trong {total_time:.2f}s | Val Acc tốt nhất: {best_val_acc*100:.2f}%")
    print(f"  ✓ Weights đã được lưu tại: {save_path}")

    return history, save_path

def main():
    print("==========================================================")
    print("🚀 BẮT ĐẦU GIAI ĐOẠN 3 (PHASE 3: TRAINING & BENCHMARKING)")
    print("==========================================================")

    # 1. Cố định random seed
    set_seed(42)

    # 2. Sinh tập dữ liệu I/Q baseband chuẩn
    print("\n[1] Chuẩn bị tập dữ liệu I/Q baseband (18,000 bursts)...")
    dataset_dict = generate_synthetic_dataset(
        samples_per_class_per_snr=150, # 150 mẫu * 8 lớp * 15 điểm SNR = 18,000 bursts
        seed=42
    )

    # 3. Phân chia 70% Train, 15% Val, 15% Test
    train_ds_raw, val_ds, test_ds = split_dataset(
        dataset_dict,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    print(f"  ✓ Tập Train gốc: {len(train_ds_raw)} mẫu | Val: {len(val_ds)} mẫu | Test: {len(test_ds)} mẫu")

    # 4. Tạo tập Train cho Mô hình Topic 4 (tích hợp RFAugmentationPipeline tinh chỉnh)
    aug_pipeline = RFAugmentationPipeline(
        phase_prob=0.5,
        noise_prob=0.4,
        shift_prob=0.4,
        scale_prob=0.3
    )
    
    # Tạo Dataset có Augmentation từ dữ liệu numpy của train_ds_raw
    train_ds_augmented = WirelessIQDataset(
        x_data=train_ds_raw.x.numpy(),
        y_data=train_ds_raw.y.numpy(),
        snr_data=train_ds_raw.snr.numpy(),
        transform=aug_pipeline
    )

    # 5. Huấn luyện Mô hình A (Baseline 1D CNN - Không dùng Augmentation)
    print("\n----------------------------------------------------------")
    print("🔹 BƯỚC 3.1: HUẤN LUYỆN MÔ HÌNH A (BASELINE 1D CNN)")
    print("----------------------------------------------------------")
    history_baseline, path_baseline = train_model(
        model_name="Baseline 1D CNN",
        train_dataset=train_ds_raw,
        val_dataset=val_ds,
        epochs=25,
        batch_size=64,
        lr=0.001
    )

    # 6. Huấn luyện Mô hình B (Augmented 1D CNN - Topic 4)
    print("\n----------------------------------------------------------")
    print("🔹 BƯỚC 3.2: HUẤN LUYỆN MÔ HÌNH B (AUGMENTED 1D CNN - TOPIC 4)")
    print("----------------------------------------------------------")
    history_augmented, path_augmented = train_model(
        model_name="Augmented 1D CNN",
        train_dataset=train_ds_augmented,
        val_dataset=val_ds,
        epochs=25,
        batch_size=64,
        lr=0.001
    )

    print("\n==========================================================")
    print("✅ HOÀN THÀNH HUẤN LUYỆN CẢ 2 MÔ HÌNH CHO GIAI ĐOẠN 3!")
    print("==========================================================")

if __name__ == '__main__':
    main()
