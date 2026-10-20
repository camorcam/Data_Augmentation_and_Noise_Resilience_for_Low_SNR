"""
Script Vẽ Đồ thị & Trực quan hóa cho Giai đoạn 4 (Phase 4):
- Đồ thị Accuracy vs SNR (-10 dB đến +18 dB)
- Đồ thị Macro-F1 vs SNR
- Ma trận nhầm lẫn chuẩn hóa dòng (Row-normalized Confusion Matrices) tại Low SNR (-4 dB) và High SNR (+14 dB)
Xuất đồ thị sắc nét vào thư mục plots/
"""
import sys
import os
import json
import numpy as np  # pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import]
import seaborn as sns  # pyrefly: ignore [missing-import]

# Cấu hình UTF-8 cho Windows console output
sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình style đồ thị chuẩn học thuật (Clean Academic Style)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#E2E8F0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

def plot_accuracy_vs_snr(snr_grid, acc_baseline, acc_augmented, save_path='plots/accuracy_vs_snr.png'):
    """Vẽ đồ thị Accuracy vs SNR."""
    plt.figure(figsize=(10, 6), dpi=300)
    
    plt.plot(snr_grid, np.array(acc_baseline) * 100, 'o-', color='#DC2626', linewidth=2.5, markersize=7, label='Mô hình A (Baseline 1D CNN)')
    plt.plot(snr_grid, np.array(acc_augmented) * 100, 's-', color='#0284C7', linewidth=2.5, markersize=7, label='Mô hình B (Augmented 1D CNN - Topic 4)')
    
    # Tô màu vùng SNR thấp (-10 dB đến +2 dB)
    plt.axvspan(-10, 2, color='#F1F5F9', alpha=0.8, label='Vùng nhiễu nặng (-10 đến +2 dB)')

    plt.title('Đường cong Hiệu năng Accuracy vs. SNR Grid', fontsize=14, fontweight='bold', pad=15, color='#0F172A')
    plt.xlabel('Tỷ số Tín hiệu trên Nhiễu SNR (dB)', fontsize=12, fontweight='bold', color='#0F172A')
    plt.ylabel('Độ chính xác Classification Accuracy (%)', fontsize=12, fontweight='bold', color='#0F172A')
    plt.xticks(snr_grid, fontsize=10)
    plt.yticks(range(10, 101, 10), fontsize=10)
    plt.ylim(10, 100)
    plt.grid(True)
    plt.legend(fontsize=11, loc='lower right', frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Đã xuất đồ thị Accuracy vs SNR: {save_path}")

def plot_macro_f1_vs_snr(snr_grid, f1_baseline, f1_augmented, save_path='plots/macro_f1_vs_snr.png'):
    """Vẽ đồ thị Macro-F1 vs SNR."""
    plt.figure(figsize=(10, 6), dpi=300)
    
    plt.plot(snr_grid, f1_baseline, 'o--', color='#E11D48', linewidth=2.2, markersize=6, label='Baseline Macro-F1')
    plt.plot(snr_grid, f1_augmented, 's-', color='#2563EB', linewidth=2.5, markersize=7, label='Augmented Macro-F1 (Topic 4)')

    plt.title('Đường cong Chỉ số Macro-F1 vs. SNR Grid', fontsize=14, fontweight='bold', pad=15, color='#0F172A')
    plt.xlabel('Tỷ số Tín hiệu trên Nhiễu SNR (dB)', fontsize=12, fontweight='bold', color='#0F172A')
    plt.ylabel('Chỉ số Macro-F1 Score', fontsize=12, fontweight='bold', color='#0F172A')
    plt.xticks(snr_grid, fontsize=10)
    plt.yticks(np.arange(0.1, 1.05, 0.1), fontsize=10)
    plt.ylim(0.1, 1.02)
    plt.grid(True)
    plt.legend(fontsize=11, loc='lower right', frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Đã xuất đồ thị Macro-F1 vs SNR: {save_path}")

def plot_confusion_matrices(cm_baseline, cm_augmented, classes, snr_val, save_path):
    """Vẽ Side-by-Side Row-Normalized Confusion Matrices tại 1 điểm SNR cụ thể."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # 1. Baseline Model
    sns.heatmap(
        cm_baseline, annot=True, fmt='.2f', cmap='Reds',
        xticklabels=classes, yticklabels=classes,
        ax=axes[0], cbar=False, vmin=0.0, vmax=1.0, annot_kws={'size': 9}
    )
    axes[0].set_title(f'Mô hình A (Baseline) - SNR = {snr_val} dB', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Modulation Dự đoán (Predicted)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Modulation Thực tế (True)', fontsize=11, fontweight='bold')

    # 2. Augmented Model
    sns.heatmap(
        cm_augmented, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        ax=axes[1], cbar=False, vmin=0.0, vmax=1.0, annot_kws={'size': 9}
    )
    axes[1].set_title(f'Mô hình B (Augmented - Topic 4) - SNR = {snr_val} dB', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Modulation Dự đoán (Predicted)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Modulation Thực tế (True)', fontsize=11, fontweight='bold')

    fig.suptitle(f'Ma trận Nhầm lẫn Chuẩn hóa theo Dòng (Row-Normalized) tại SNR = {snr_val} dB', fontsize=14, fontweight='bold', y=0.98)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Đã xuất Ma trận nhầm lẫn tại SNR = {snr_val} dB: {save_path}")

def generate_all_plots(results_file='checkpoint/benchmark_results.json'):
    """Đọc file kết quả và tạo toàn bộ đồ thị."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Không tìm thấy file: {results_file}. Vui lòng chạy evaluate.py trước!")

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    classes = data['classes']
    snr_grid = data['snr_grid']
    baseline = data['baseline']
    augmented = data['augmented']

    print("📊 Đang tiến hành trực quan hóa toàn bộ đồ thị cho Giai đoạn 4...")

    # 1. Vẽ Accuracy vs SNR
    plot_accuracy_vs_snr(snr_grid, baseline['acc_per_snr'], augmented['acc_per_snr'])

    # 2. Vẽ Macro-F1 vs SNR
    plot_macro_f1_vs_snr(snr_grid, baseline['f1_per_snr'], augmented['f1_per_snr'])

    # 3. Vẽ Confusion Matrix tại Low SNR (-4 dB)
    if "-4" in baseline['cm_per_snr']:
        plot_confusion_matrices(
            np.array(baseline['cm_per_snr']["-4"]),
            np.array(augmented['cm_per_snr']["-4"]),
            classes, snr_val=-4,
            save_path='plots/confusion_matrices_low_snr.png'
        )

    # 4. Vẽ Confusion Matrix tại High SNR (+14 dB)
    if "14" in baseline['cm_per_snr']:
        plot_confusion_matrices(
            np.array(baseline['cm_per_snr']["14"]),
            np.array(augmented['cm_per_snr']["14"]),
            classes, snr_val=14,
            save_path='plots/confusion_matrices_high_snr.png'
        )

if __name__ == '__main__':
    generate_all_plots()
