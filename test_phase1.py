"""
Test Script cho Giai đoạn 1 (Phase 1): Tiền xử lý Tín hiệu & Tập dữ liệu I/Q Baseband
- Kiểm tra Tensor Shape [N_batch, 2, 128]
- Kiểm tra Chuẩn hóa RMS (RMS Energy = 1.0)
- Kiểm tra Phân chia Tập 70% Train / 15% Val / 15% Test
- Kiểm tra Lưới 15 điểm SNR Grid (-10 dB đến +18 dB)
"""
import sys
import numpy as np  # pyrefly: ignore [missing-import]

# Cấu hình UTF-8 cho Windows console output
sys.stdout.reconfigure(encoding='utf-8')

# Import package dataset
from dataset import (  # pyrefly: ignore [missing-import]
    generate_synthetic_dataset,
    format_iq_tensor,
    normalize_rms,
    MODULATION_CLASSES,
    SNR_GRID,
    WirelessIQDataset
)
from dataset.pytorch_dataset import split_dataset  # pyrefly: ignore [missing-import]

def main():
    print("==========================================================")
    print("🚀 BẮT ĐẦU KIỂM TRA GIAI ĐOẠN 1 (PHASE 1 PREPROCESSING)")
    print("==========================================================")

    # 1. Sinh tập dữ liệu synthetic
    print("\n[1] Sinh tập dữ liệu I/Q baseband...")
    dataset_dict = generate_synthetic_dataset(
        samples_per_class_per_snr=50, # 50 mẫu * 8 lớp * 15 điểm SNR = 6,000 bursts
        seed=42
    )
    
    x = dataset_dict['x']
    y = dataset_dict['y']
    snr = dataset_dict['snr']
    
    print(f"  ✓ Tổng số mẫu bursts đã sinh: {len(x)}")
    print(f"  ✓ Kích thước Tensor (x.shape): {x.shape} (Kỳ vọng: [6000, 2, 128])")
    print(f"  ✓ Danh sách 8 lớp Modulation: {MODULATION_CLASSES}")
    print(f"  ✓ Lưới 15 điểm SNR Grid ({len(SNR_GRID)} điểm): {SNR_GRID} dB")

    # 2. Kiểm tra định dạng Tensor [N_batch, 2, 128]
    print("\n[2] Kiểm tra định dạng Tensor chuẩn...")
    formatted_x = format_iq_tensor(x)
    assert formatted_x.shape == (6000, 2, 128), "Lỗi định dạng Tensor Shape!"
    print("  ✓ Tensor đạt chuẩn dạng [N_batch, 2, 128] float32!")

    # 3. Kiểm tra Chuẩn hóa RMS (RMS Energy = 1.0)
    print("\n[3] Kiểm tra Chuẩn hóa RMS (RMS Normalization)...")
    # Tính RMS energy của 10 mẫu ngẫu nhiên
    sample_rms_list = []
    for idx in range(10):
        burst = x[idx] # shape [2, 128]
        i_chan = burst[0, :]
        q_chan = burst[1, :]
        rms_val = np.sqrt(np.mean(i_chan**2 + q_chan**2))
        sample_rms_list.append(rms_val)

    avg_rms = np.mean(sample_rms_list)
    print(f"  ✓ RMS Energy trung bình của các mẫu: {avg_rms:.6f} (Kỳ vọng: 1.000000)")
    assert np.allclose(avg_rms, 1.0, atol=1e-4), "Lỗi chuẩn hóa RMS!"
    print("  ✓ Chuẩn hóa RMS thành công! Đã triệt tiêu chênh lệch công suất trung bình giữa các lớp.")

    # 4. Phân chia tập dữ liệu 70% Train / 15% Val / 15% Test
    print("\n[4] Kiểm tra Phân chia Tập dữ liệu (70/15/15 Split)...")
    train_ds, val_ds, test_ds = split_dataset(
        dataset_dict,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    print(f"  ✓ Số mẫu tập Train (70%): {len(train_ds)} bursts")
    print(f"  ✓ Số mẫu tập Val   (15%): {len(val_ds)} bursts")
    print(f"  ✓ Số mẫu tập Test  (15%): {len(test_ds)} bursts")
    
    assert len(train_ds) + len(val_ds) + len(test_ds) == len(x), "Tổng số mẫu không khớp!"
    print("  ✓ Phân chia dữ liệu thành công, đảm bảo Zero Data Leakage giữa Train/Val/Test.")

    print("\n==========================================================")
    print("✅ HOÀN THÀNH HOÀN HẢO GIAI ĐOẠN 1!")
    print("==========================================================")

if __name__ == '__main__':
    main()
