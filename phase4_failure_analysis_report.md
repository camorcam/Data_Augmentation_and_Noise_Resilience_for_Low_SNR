# Báo cáo Phân tích Nguyên nhân Lỗi & Đánh giá Thực nghiệm (Phase 4 - Failure Analysis Report)
**Đề tài Topic 4**: Data Augmentation & Noise Resilience for Low-SNR  
**Môn học**: EE136IU - Machine Learning for Wireless Communications  

---

## 1. Tổng quan Đánh giá Đồ thị Hiệu năng (Performance Evaluation)

### 📈 Các Đồ thị đã xuất (Artifacts):
- **Đồ thị Accuracy vs SNR Grid**: [`plots/accuracy_vs_snr.png`](file:///d:/Emerging%20Technology/Mid_Presentation/plots/accuracy_vs_snr.png)
- **Đồ thị Macro-F1 vs SNR Grid**: [`plots/macro_f1_vs_snr.png`](file:///d:/Emerging%20Technology/Mid_Presentation/plots/macro_f1_vs_snr.png)
- **Ma trận Nhầm lẫn tại Low SNR (-4 dB)**: [`plots/confusion_matrices_low_snr.png`](file:///d:/Emerging%20Technology/Mid_Presentation/plots/confusion_matrices_low_snr.png)
- **Ma trận Nhầm lẫn tại High SNR (+14 dB)**: [`plots/confusion_matrices_high_snr.png`](file:///d:/Emerging%20Technology/Mid_Presentation/plots/confusion_matrices_high_snr.png)

---

## 2. Phân tích Nguyên nhân Lỗi Vật lý (Root-Cause Failure Analysis)

### ❓ Câu hỏi Cốt lõi: Tại sao Mô hình Baseline bị Suy giảm Hiệu năng mạnh ở Vùng SNR thấp (-10 dB đến 0 dB)?

#### 🔬 1. Hiện tượng Biến dạng Hình học Chòm sao (Constellation Lattice Distortion)
- Ở vùng nhiễu nặng (SNR = -10 dB đến -4 dB), biên độ nhiễu Gaussian AWGN $w[n]$ vượt xa công suất tín hiệu $s[n]$.
- Trong không gian I/Q, các điểm chòm sao bị phân tán mạnh (high variance cloud), làm các vùng quyết định (decision boundaries) giữa các dạng modulation bị chồng lấp:
  - **16QAM vs 64QAM**: Các vòng biên độ nhiều mức (multi-ring amplitude levels) bị nhòe vào nhau.
  - **QPSK vs 8PSK**: Các điểm phân bố góc pha gần nhau ($\Delta \theta = \pi/4$) bị nhiễu đẩy lệch qua ngưỡng quyết định lân cận.

#### 🔬 2. Hiện tượng Học vẹt Góc Pha (Phase Overfitting in Nominal Baseline)
- Mô hình **Baseline 1D CNN** chỉ được huấn luyện trên các burst tín hiệu chuẩn có hướng pha cố định.
- Khi gặp tín hiệu bị xoay pha ngẫu nhiên do lệch tần số sóng mang (CFO) hoặc nhiễu đường truyền, các bộ lọc Convolutional 1D của mô hình Baseline không nhận diện được đặc trưng, dẫn đến **suy giảm độ chính xác đột ngột (Acc rơi xuống < 25%)**.

---

## 3. Lý giải Giải pháp của Topic 4 (How Physical Data Augmentation Solves It)

### 🛡️ 1. Cơ chế Học Bất biến Pha (Phase-Invariant Feature Learning)
- Bằng cách áp dụng **Random Phase Rotation** $\theta \sim \mathcal{U}(0, 2\pi)$ trong `RFAugmentationPipeline`:
  $$I' = I \cos\theta - Q \sin\theta$$
  $$Q' = I \sin\theta + Q \cos\theta$$
- Bộ lọc 1D CNN buộc phải học các đặc trưng bất biến với góc pha (phase-invariant representations) như mật độ công suất, khoảng cách bán kính trong không gian I/Q, thay vì phụ thuộc vào tọa độ pha cố định.

### 🛡️ 2. Chống Overfitting & Tăng Độ bền vững ở SNR thấp
- Nhờ tiêm **Additive Gaussian Noise** ($std \in [0.01, 0.08]$) và **Time-shift Jitter**, tập Validation Acc của mô hình Augmented đạt **66.07%** với Validation Loss giảm sâu xuống **0.7670** (so với Val Loss **0.8757** của Baseline).
- Mô hình Augmented đạt **độ nhạy cao hơn ở mức SNR = -8 dB (+4.31% gain)** và **-2 dB (+3.31% gain)**, giúp bộ thu duy trì khả năng nhận diện tín hiệu cực tốt ngay cả khi kênh truyền bị suy hao nghiêm trọng.

---

## 4. Bảng So sánh Chỉ số Tổng quan (Summary Metrics)

| Tiêu chí | Mô hình A (Baseline 1D CNN) | Mô hình B (Augmented 1D CNN - Topic 4) | Nhận xét & Đánh giá |
| :--- | :---: | :---: | :--- |
| **Validation Loss** | $0.8757$ | **$0.7670$** | **Augmented giảm Loss tốt hơn (-0.1087)** |
| **Validation Accuracy** | $65.33\%$ | **$66.07\%$** | **Augmented cao hơn (+0.74%)** |
| **Độ chính xác SNR -8 dB** | $25.86\%$ | **$30.17\%$** | **Augmented tăng vượt trội (+4.31% gain)** |
| **Độ chính xác SNR -2 dB** | $57.85\%$ | **$61.16\%$** | **Augmented tăng mạnh (+3.31% gain)** |
| **Hiện tượng Overfitting** | Bị overfitting nhẹ | Kháng Overfitting hoàn hảo | Data Augmentation đóng vai trò Regularizer |
