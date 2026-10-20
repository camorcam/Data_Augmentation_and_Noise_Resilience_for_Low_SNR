"""
Script Đánh giá & So sánh Thực nghiệm Chi tiết cho Giai đoạn 3 & 4 (Phase 3 & 4):
- Đánh giá Mô hình A (Baseline) vs Mô hình B (Augmented) trên tập Test
- Quét qua 15 điểm SNR Grid (-10 dB đến +18 dB)
- Tính toán Accuracy vs SNR, Macro-F1 score, Ma trận nhầm lẫn (Confusion Matrix)
- Xuất kết quả chi tiết ra checkpoint/benchmark_results.json
"""
import sys
import os
import json
import torch  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]
from sklearn.metrics import confusion_matrix, f1_score  # pyrefly: ignore [missing-import]

# Cấu hình UTF-8 cho Windows console output
sys.stdout.reconfigure(encoding='utf-8')

# Import các module
from dataset import (  # pyrefly: ignore [missing-import]
    generate_synthetic_dataset,
    WirelessIQDataset,
    MODULATION_CLASSES,
    SNR_GRID
)
from dataset.pytorch_dataset import split_dataset  # pyrefly: ignore [missing-import]
from models import ModulationClassifier1DCNN  # pyrefly: ignore [missing-import]

def load_model(checkpoint_path: str, num_classes: int = 8, device: str = 'cpu') -> torch.nn.Module:
    """Tải weights mô hình từ checkpoint."""
    model = ModulationClassifier1DCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_snr_grid(model, test_dataset: WirelessIQDataset, snr_grid=SNR_GRID, device='cpu'):
    """
    Đánh giá mô hình trên từng điểm SNR trong lưới 15 điểm.
    Trả về dict lưu accuracy, macro_f1 và confusion matrix cho từng điểm SNR.
    """
    model.eval()
    x_test = test_dataset.x.to(device)
    y_test = test_dataset.y.numpy()
    snr_test = test_dataset.snr.numpy()

    with torch.no_grad():
        logits = model(x_test)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    acc_per_snr = []
    f1_per_snr = []
    cm_per_snr = {}

    for snr in snr_grid:
        mask = (snr_test == snr)
        if np.sum(mask) == 0:
            continue

        y_true_snr = y_test[mask]
        y_pred_snr = preds[mask]

        acc = np.mean(y_true_snr == y_pred_snr)
        macro_f1 = f1_score(y_true_snr, y_pred_snr, average='macro', zero_division=0)
        cm = confusion_matrix(y_true_snr, y_pred_snr, labels=list(range(len(MODULATION_CLASSES))))
        
        # Chuẩn hóa theo dòng (row-normalized confusion matrix)
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)

        acc_per_snr.append(float(acc))
        f1_per_snr.append(float(macro_f1))
        cm_per_snr[str(snr)] = cm_norm.tolist()

    global_acc = float(np.mean(y_test == preds))
    global_f1 = float(f1_score(y_test, preds, average='macro', zero_division=0))

    return {
        'snr_grid': snr_grid,
        'acc_per_snr': acc_per_snr,
        'f1_per_snr': f1_per_snr,
        'cm_per_snr': cm_per_snr,
        'global_acc': global_acc,
        'global_f1': global_f1
    }

def main():
    print("==========================================================")
    print("📊 BẮT ĐẦU ĐÁNH GIÁ & SO SÁNH BENCHMARK (PHASE 3 & 4)")
    print("==========================================================")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Sinh tập dữ liệu test độc lập
    print("\n[1] Chuẩn bị tập dữ liệu Test độc lập...")
    dataset_dict = generate_synthetic_dataset(samples_per_class_per_snr=100, seed=42)
    _, _, test_ds = split_dataset(dataset_dict, seed=42)

    path_baseline = 'checkpoint/baseline_1d_cnn.pth'
    path_augmented = 'checkpoint/augmented_1d_cnn.pth'

    if not os.path.exists(path_baseline) or not os.path.exists(path_augmented):
        print("❌ Chưa tìm thấy file checkpoint. Vui lòng chạy train.py trước!")
        sys.exit(1)

    # 2. Load 2 mô hình
    print("\n[2] Tải mô hình Baseline và Mô hình Augmented...")
    model_baseline = load_model(path_baseline, device=device)
    model_augmented = load_model(path_augmented, device=device)

    # 3. Đánh giá trên lưới 15 điểm SNR
    print("\n[3] Đánh giá độ chính xác trên 15 điểm SNR (-10 dB đến +18 dB)...")
    res_baseline = evaluate_snr_grid(model_baseline, test_ds, device=device)
    res_augmented = evaluate_snr_grid(model_augmented, test_ds, device=device)

    print("\n" + "="*70)
    print(f"{'SNR (dB)':<10} | {'Baseline Acc (%)':<20} | {'Augmented Acc (%)':<20} | {'Gain (%)':<10}")
    print("-" * 70)
    for snr, acc_b, acc_a in zip(SNR_GRID, res_baseline['acc_per_snr'], res_augmented['acc_per_snr']):
        gain = (acc_a - acc_b) * 100.0
        print(f"{snr:<10} | {acc_b*100:<20.2f} | {acc_a*100:<20.2f} | {gain:<+10.2f}")
    print("=" * 70)

    print(f"\n🏆 Tổng quan toàn bộ tập Test:")
    print(f"  • Baseline Model  : Global Acc = {res_baseline['global_acc']*100:.2f}% | Macro-F1 = {res_baseline['global_f1']:.4f}")
    print(f"  • Augmented Model : Global Acc = {res_augmented['global_acc']*100:.2f}% | Macro-F1 = {res_augmented['global_f1']:.4f}")

    # 4. Lưu kết quả ra benchmark_results.json
    results = {
        'classes': MODULATION_CLASSES,
        'snr_grid': SNR_GRID,
        'baseline': res_baseline,
        'augmented': res_augmented
    }

    out_file = 'checkpoint/benchmark_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✓ Kết quả benchmark chi tiết đã được xuất ra file: {out_file}")
    print("\n==========================================================")
    print("✅ HOÀN THÀNH THỰC NGHIỆM SO SÁNH CHO GIAI ĐOẠN 3!")
    print("==========================================================")

if __name__ == '__main__':
    main()
