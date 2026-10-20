# EE136IU Capstone Mini-Project — Topic 4: Data Augmentation & Noise Resilience for Low-SNR

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Dự án Capstone Mini-Project môn học **EE136IU: Machine Learning for Wireless Communications**.  
**Đề tài Topic 4**: *Tăng cường Dữ liệu trên Miền Vật lý & Nâng cao Khả năng Kháng nhiễu Bộ thu trong Vùng SNR Thấp (-10 dB đến +2 dB)*.

---

## 🚀 Hướng dẫn Nhanh (1-Command Quickstart)

Tái tạo toàn bộ tập dữ liệu, chạy thực nghiệm đánh giá và mở Web App Demo chỉ với **1 dòng lệnh trên Terminal**:

```powershell
streamlit run app.py
```

Ứng dụng Web tương tác sẽ khởi chạy tại: `http://localhost:8501`.

---

## 🛠️ Cài đặt Môi trường (Installation)

1. **Tải mã nguồn về máy hoặc Clone Repository**:
   ```bash
   git clone https://github.com/your-username/EE136IU_Topic4_Capstone.git
   cd EE136IU_Topic4_Capstone
   ```

2. **Cài đặt các thư viện phụ thuộc**:
   ```powershell
   py -m pip install -r requirements.txt
   ```

---

## 📁 Cấu trúc Thư mục Dự án (Modular Architecture)

```text
├── dataset/                   # Module Tiền xử lý & Sinh tín hiệu I/Q Baseband
│   ├── __init__.py
│   ├── preprocessing.py       # Chuẩn hóa RMS (E_rms = 1.0) & định dạng Tensor [N, 2, 128]
│   ├── signal_generator.py    # Sinh 8 lớp modulation & lưới 15 điểm SNR Grid (-10 dB đến +18 dB)
│   └── pytorch_dataset.py     # PyTorch Dataset wrapper & Phân chia 70/15/15 Zero-Leakage Split
├── models/                    # Module Kiến trúc Mô hình & Transforms
│   ├── __init__.py
│   ├── transforms.py          # RFAugmentationPipeline (Random Phase, AWGN, Time Shift, Scaling)
│   └── network.py             # Kiến trúc 1D CNN 3 khối (ModulationClassifier1DCNN)
├── utils/                     # Module Trực quan hóa & Đồ thị
│   ├── __init__.py
│   └── plot_benchmarks.py     # Xuất đồ thị Accuracy, Macro-F1 & Confusion Matrices
├── checkpoint/                # Thư mục lưu Model Weights & Metrics JSON
│   ├── baseline_1d_cnn.pth    # Weights Mô hình Baseline
│   ├── augmented_1d_cnn.pth   # Weights Mô hình Topic 4 (Augmented)
│   └── benchmark_results.json # Bảng số liệu benchmark 15 điểm SNR
├── plots/                     # Thư mục lưu ảnh đồ thị xuất ra
│   ├── accuracy_vs_snr.png
│   ├── macro_f1_vs_snr.png
│   ├── confusion_matrices_low_snr.png
│   └── confusion_matrices_high_snr.png
├── app.py                     # Web App Demo Tương tác Streamlit
├── train.py                   # Script Huấn luyện tự động 2 mô hình
├── evaluate.py                # Script Đánh giá thực nghiệm tập Test
├── test_phase1.py             # Unit test Giai đoạn 1 (Dataset & Preprocessing)
├── test_phase2.py             # Unit test Giai đoạn 2 (Augmentation & Model)
├── requirements.txt           # Thư viện phụ thuộc
├── slide_style_requirements.md# Quy chuẩn thiết kế Slide thuyết trình
├── phase4_failure_analysis_report.md # Báo cáo Phân tích Nguyên nhân Lỗi Vật lý
└── README.md                  # Hướng dẫn dự án
```

---

## 📊 Quy trình Chạy Dự án theo Từng Bước (Workflow Execution)

### Bước 1: Kiểm thử Tiền xử lý Tín hiệu & Tensor Shape
```powershell
py test_phase1.py
```
*Đảm bảo tensor đạt shape `[N_batch, 2, 128]` và chuẩn hóa RMS năng lượng $E_{\text{rms}} = 1.0$.*

### Bước 2: Kiểm thử Pipeline Augmentation & Forward Pass Mô hình
```powershell
py test_phase2.py
```
*Kiểm tra 4 phép biến đổi miền RF vật lý và kiến trúc 1D CNN (175,432 parameters).*

### Bước 3: Huấn luyện 2 Mô hình (Baseline vs Augmented)
```powershell
py train.py
```
*Huấn luyện song song Mô hình Baseline và Mô hình Topic 4 trên 18,000 bursts tín hiệu, lưu weights vào `checkpoint/`.*

### Bước 4: Đánh giá Benchmark & Trực quan hóa Đồ thị
```powershell
py evaluate.py
py utils/plot_benchmarks.py
```
*Đánh giá độ chính xác trên 15 điểm SNR Grid và xuất đồ thị ra thư mục `plots/`.*

### Bước 5: Mở Web App Demo Tương tác
```powershell
streamlit run app.py
```

---

## 🔬 Kết quả Thực nghiệm Nổi bật (Key Results)

| Tiêu chí Đánh giá | Mô hình Baseline 1D CNN | Mô hình Augmented 1D CNN (Topic 4) | Cải thiện (Gain) |
| :--- | :---: | :---: | :---: |
| **Validation Loss** | $0.8757$ | **$0.7670$** | **Giảm Loss -0.1087** |
| **Validation Accuracy** | $65.33\%$ | **$66.07\%$** | **+0.74%** |
| **Độ chính xác SNR = -8 dB** | $25.86\%$ | **$30.17\%$** | **+4.31% gain** |
| **Độ chính xác SNR = -2 dB** | $57.85\%$ | **$61.16\%$** | **+3.31% gain** |
| **Bất biến Pha ($\theta \in [0, 2\pi)$)** | Nhầm lẫn (Acc < 25%) | Dự đoán đúng 100% | Kháng lệch pha tuyệt đối |

---

## 📜 Giấy phép & Tác quyền (License)
Dự án được phát triển phục vụ báo cáo Midterm môn học **EE136IU** tại Trường Đại học Quốc tế - ĐHQG TP.HCM.
