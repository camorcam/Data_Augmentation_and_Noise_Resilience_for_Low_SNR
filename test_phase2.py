"""
Test Script cho Giai đoạn 2 (Phase 2): Module Data Augmentation & Kiến trúc Mô hình 1D CNN
- Kiểm tra các phép biến đổi Data Augmentation trên miền RF (Random Phase, AWGN Noise, Time Shift, Amplitude Scaling)
- Kiểm tra RFAugmentationPipeline bảo toàn Tensor Shape [N_batch, 2, 128]
- Kiểm tra Kiến trúc Mô hình 1D CNN (Forward Pass & Logits Shape [N_batch, 8])
- Kiểm tra số lượng tham số của mô hình (Trainable Parameters)
"""
import sys
import torch  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]

# Cấu hình UTF-8 cho Windows console output
sys.stdout.reconfigure(encoding='utf-8')

# Import các module vừa tạo
from models import (  # pyrefly: ignore [missing-import]
    RandomPhaseRotation,
    AdditiveGaussianNoise,
    TimeShiftJitter,
    AmplitudeScaling,
    RFAugmentationPipeline,
    ModulationClassifier1DCNN
)
from models.network import count_parameters  # pyrefly: ignore [missing-import]
from dataset import generate_synthetic_dataset, WirelessIQDataset  # pyrefly: ignore [missing-import]

def main():
    print("==========================================================")
    print("🚀 BẮT ĐẦU KIỂM TRA GIAI ĐOẠN 2 (PHASE 2 MODEL & TRANSFORMS)")
    print("==========================================================")

    # 1. Kiểm tra Module Data Augmentation miền RF
    print("\n[1] Kiểm tra các phép biến đổi Data Augmentation miền RF...")
    dummy_burst = torch.randn(2, 128) # Shape [2, 128]
    
    phase_rot = RandomPhaseRotation(p=1.0)
    noise_inj = AdditiveGaussianNoise(p=1.0)
    time_shift = TimeShiftJitter(p=1.0)
    amp_scale = AmplitudeScaling(p=1.0)

    out_phase = phase_rot(dummy_burst)
    out_noise = noise_inj(dummy_burst)
    out_shift = time_shift(dummy_burst)
    out_scale = amp_scale(dummy_burst)

    assert out_phase.shape == (2, 128), "Lỗi shape RandomPhaseRotation!"
    assert out_noise.shape == (2, 128), "Lỗi shape AdditiveGaussianNoise!"
    assert out_shift.shape == (2, 128), "Lỗi shape TimeShiftJitter!"
    assert out_scale.shape == (2, 128), "Lỗi shape AmplitudeScaling!"

    print("  ✓ Tất cả các biến đổi đơn lẻ đều bảo toàn shape [2, 128] float32 thành công!")

    # 2. Kiểm tra RFAugmentationPipeline tổng hợp
    print("\n[2] Kiểm tra RFAugmentationPipeline tổng hợp cho Topic 4...")
    pipeline = RFAugmentationPipeline(phase_prob=1.0, noise_prob=1.0, shift_prob=1.0, scale_prob=1.0)
    batch_tensor = torch.randn(32, 2, 128) # Batch 32 mẫu
    aug_batch = pipeline(batch_tensor)

    assert aug_batch.shape == (32, 2, 128), "Lỗi shape RFAugmentationPipeline batch!"
    print("  ✓ RFAugmentationPipeline biến đổi batch [32, 2, 128] hoạt động hoàn hảo!")

    # 3. Kiểm tra Kiến trúc Mô hình 1D CNN
    print("\n[3] Kiểm tra Kiến trúc Mô hình 1D CNN (ModulationClassifier1DCNN)...")
    model = ModulationClassifier1DCNN(num_classes=8, dropout_rate=0.3)
    num_params = count_parameters(model)
    print(f"  ✓ Tổng số tham số trainable của 1D CNN: {num_params:,} parameters")

    # Forward Pass test
    logits = model(aug_batch)
    assert logits.shape == (32, 8), f"Shape Logits không đúng: {logits.shape}"
    print(f"  ✓ Forward Pass thành công! Logits shape: {logits.shape} (Kỳ vọng: [32, 8])")

    # 4. Kiểm tra Tích hợp PyTorch Dataset với RFAugmentationPipeline
    print("\n[4] Kiểm tra Tích hợp Dataset với Augmentation Pipeline...")
    dataset_dict = generate_synthetic_dataset(samples_per_class_per_snr=5, seed=42) # Dataset nhỏ
    
    # Dataset có dùng Augmentation (Topic 4)
    aug_ds = WirelessIQDataset(
        dataset_dict['x'], dataset_dict['y'], dataset_dict['snr'],
        transform=pipeline
    )
    
    sample_x, sample_y, sample_snr = aug_ds[0]
    assert sample_x.shape == (2, 128), "Lỗi shape Dataset item!"
    print(f"  ✓ Mẫu lấy ra từ Dataset có shape: {sample_x.shape}, label: {sample_y.item()} ({dataset_dict['classes'][sample_y.item()]}), SNR: {sample_snr.item()} dB")

    print("\n==========================================================")
    print("✅ HOÀN THÀNH HOÀN HẢO GIAI ĐOẠN 2!")
    print("==========================================================")

if __name__ == '__main__':
    main()
