# 📡 PROJECT OVERVIEW — EE136IU Capstone Midterm (Topic 4)

> **Môn học**: EE136IU — Machine Learning for Wireless Communications  
> **Đề tài**: Data Augmentation & Noise Resilience for Low-SNR  
> **Cấp độ**: Midterm Capstone Mini-Project

---

## 🎯 Mục tiêu dự án là gì?

**Bài toán thực tế:**  
Trong viễn thông không dây, thiết bị thu cần **nhận dạng loại điều chế (modulation)** của tín hiệu đến — ví dụ: BPSK, QPSK, 16QAM, 64QAM... — để giải mã đúng dữ liệu. Nhưng ở môi trường nhiễu nặng (**SNR từ -10 dB đến +2 dB**), tín hiệu bị biến dạng nặng, mô hình AI thường xuyên nhầm lẫn.

**Giải pháp đề tài đề xuất:**  
Thay vì cần thêm dữ liệu thật từ phần cứng, nhóm dùng **Physical RF Data Augmentation** — biến đổi nhân tạo dữ liệu I/Q trên miền tín hiệu vật lý — để dạy mô hình chống nhiễu tốt hơn.

**Cách kiểm chứng:**  
Train 2 mô hình 1D CNN **cùng kiến trúc, cùng hyperparameters** — 1 cái không dùng augmentation (Baseline), 1 cái dùng (Topic 4) — rồi so sánh hiệu năng.

---

## 🗂️ Cấu trúc Project & File nào làm gì?

```
Mid_Presentation/
│
├── dataset/                   ← Module sinh & xử lý dữ liệu
│   ├── signal_generator.py    ← Sinh tín hiệu I/Q cho 8 loại modulation
│   ├── preprocessing.py       ← Chuẩn hóa RMS năng lượng
│   └── pytorch_dataset.py     ← PyTorch Dataset + phân chia 70/15/15
│
├── models/                    ← Module mô hình & augmentation
│   ├── network.py             ← Kiến trúc 1D CNN (ModulationClassifier1DCNN)
│   └── transforms.py          ← RFAugmentationPipeline (4 phép biến đổi vật lý)
│
├── utils/
│   └── plot_benchmarks.py     ← Vẽ đồ thị Accuracy/F1 vs SNR, Confusion Matrix
│
├── checkpoint/                ← Kết quả sau khi train
│   ├── baseline_1d_cnn.pth    ← Weights mô hình Baseline
│   ├── augmented_1d_cnn.pth   ← Weights mô hình Topic 4
│   └── benchmark_results.json ← Số liệu benchmark 15 điểm SNR
│
├── plots/                     ← Đồ thị xuất ra
│   ├── accuracy_vs_snr.png
│   ├── macro_f1_vs_snr.png
│   ├── confusion_matrices_low_snr.png
│   └── confusion_matrices_high_snr.png
│
├── train.py       ← Chạy để train 2 mô hình
├── evaluate.py    ← Chạy để đánh giá benchmark 15 SNR points
├── app.py         ← Web Demo Streamlit (chạy để demo live)
├── test_phase1.py ← Unit test kiểm tra dataset & tensor shape
└── test_phase2.py ← Unit test kiểm tra augmentation & model forward
```

---

## 🔄 Quy trình thực hiện (Workflow)

```
Bước 1: Sinh dữ liệu I/Q          → dataset/signal_generator.py
         8 loại modulation × 15 SNR × 150 mẫu = 18,000 bursts

Bước 2: Tiền xử lý tensor         → dataset/preprocessing.py
         Chuẩn hóa RMS → tensor [N, 2, 128]
         Phân chia 70% Train / 15% Val / 15% Test (zero-leakage)

Bước 3: Augmentation pipeline      → models/transforms.py
         4 phép biến đổi vật lý áp dụng ngẫu nhiên mỗi batch

Bước 4: Train 2 mô hình           → train.py
         Model A: train trên dữ liệu gốc (Baseline)
         Model B: train trên dữ liệu đã augment (Topic 4)
         → Lưu .pth vào checkpoint/

Bước 5: Đánh giá benchmark        → evaluate.py + utils/plot_benchmarks.py
         Đo Accuracy & Macro-F1 trên 15 điểm SNR
         Xuất đồ thị & confusion matrices ra plots/

Bước 6: Demo live                  → app.py (Streamlit)
         Giao diện web chỉnh SNR, Phase Rotation realtime
         So sánh dự đoán side-by-side
```

---

## 🧠 Kỹ thuật cốt lõi — 4 phép RF Augmentation

> File: `models/transforms.py` — class `RFAugmentationPipeline`

| Phép biến đổi | Công thức | Mô phỏng hiện tượng | Xác suất |
|---|---|---|:---:|
| **Random Phase Rotation** | `s *= exp(j*θ)`, θ ~ U[0,2π) | Lệch pha sóng mang (CFO) | 50% |
| **AWGN Injection** | `s += N(0, σ²)` | Nhiễu Gaussian thêm vào | 40% |
| **Time-Shift Jitter** | Dịch vòng ngẫu nhiên | Trễ đồng bộ (timing offset) | 40% |
| **Amplitude Scaling** | `s *= α`, α ~ U[0.8,1.2] | Suy hao/tăng ích kênh | 30% |

**Cơ chế hoạt động:** Mỗi batch train, pipeline ngẫu nhiên áp dụng 0–4 phép trên → model buộc phải học đặc trưng bất biến thay vì học thuộc lòng dữ liệu sạch.

---

## 🏗️ Kiến trúc mô hình 1D CNN

> File: `models/network.py` — class `ModulationClassifier1DCNN`

```
Input: [batch, 2, 128]      ← 2 kênh I và Q, mỗi kênh 128 mẫu
   │
   ├─ Conv1D(64) + BatchNorm + ReLU + MaxPool(2)
   ├─ Conv1D(128) + BatchNorm + ReLU + MaxPool(2)
   ├─ Conv1D(256) + BatchNorm + ReLU + MaxPool(2)
   │
   ├─ AdaptiveAvgPool → Flatten
   ├─ FC(256) + ReLU + Dropout(0.4)
   └─ FC(8) → Softmax

Output: xác suất 8 lớp modulation
Tổng params: ~175,432
```

**Hyperparameters (giống nhau cho cả 2 model):**
- Optimizer: Adam, lr=0.001, weight_decay=1e-4
- Scheduler: CosineAnnealingLR, T_max=25
- Epochs: 25, Batch size: 64

---

## 📊 Kết quả chính

| Tiêu chí | 🔴 Baseline | 🔵 Augmented (Topic 4) | Gain |
|---|:---:|:---:|:---:|
| Val Loss | 0.8757 | **0.7670** | ↓ 12.4% |
| Val Accuracy | 65.33% | **66.07%** | +0.74% |
| Accuracy @ -8 dB | 25.86% | **30.17%** | **+4.31%** |
| Accuracy @ -2 dB | 57.85% | **61.16%** | **+3.31%** |
| Phase Invariance | ❌ < 25% | ✅ **100%** | Tuyệt đối |

**Kết luận:** Ở SNR cao (> +6 dB) cả 2 tương đương. Ở SNR thấp (< 0 dB) Augmented vượt trội rõ rệt, đặc biệt kháng lệch pha hoàn toàn.

---

## ▶️ Cách chạy nhanh

```powershell
# Cài dependencies
py -m pip install -r requirements.txt

# Chạy unit tests (kiểm tra pipeline)
py test_phase1.py
py test_phase2.py

# Train 2 mô hình (~5-10 phút)
py train.py

# Đánh giá benchmark
py evaluate.py
py utils/plot_benchmarks.py

# Mở Web Demo (demo live)
streamlit run app.py
# → Truy cập http://localhost:8501
```

---

## 📁 Files hỗ trợ khác

| File | Mục đích |
|---|---|
| `presentation_script.md` | Script thuyết trình 15 phút |
| `phase4_failure_analysis_report.md` | Phân tích kỹ thuật tại sao Baseline thất bại |
| `topic4_execution_plan.md` | Lộ trình thực hiện 6 giai đoạn |
| `checklist_defense.md` | Checklist kiểm tra trước khi bảo vệ |
| `demo_instructions.md` | Hướng dẫn thao tác demo |
| `slide_style_requirements.md` | Quy chuẩn thiết kế slide |
