# Lộ Trình Các Bước Thực Hiện Bài Midterm - Topic 4
**Môn học**: EE136IU - Machine Learning for Wireless Communications  
**Đề tài Topic 4**: Data Augmentation & Noise Resilience for Low-SNR  

---

## 🎯 Mục tiêu Tổng quát của Topic 4
Nâng cao độ nhạy của bộ thu tín hiệu (Receiver Sensitivity) trong vùng nhiễu nặng (**SNR từ -10 dB đến +2 dB**) bằng phương pháp **Tăng cường dữ liệu trên miền vật lý (Physical RF Data Augmentation)**, so sánh trực tiếp hiệu năng giữa 1D CNN gốc (Nominal Baseline) và 1D CNN có Augmentation.

---

## 🚀 Kế hoạch Thực hiện 6 Giai đoạn (Step-by-Step Execution Plan)

### GIAI ĐOẠN 1: Chuẩn bị Dữ liệu & Tiền xử lý (Dataset & Baseband Pipeline)
- [ ] **Bước 1.1: Định dạng Tensor chuẩn**:
  - Đảm bảo dữ liệu I/Q thu thập/tạo ra có độ dài burst $L = 128$ mẫu complex baseband.
  - Chuyển đổi tensor về định dạng shape `[N_batch, 2, 128]` (Hàng 0: In-phase $I[n]$, Hàng 1: Quadrature $Q[n]$).
- [ ] **Bước 1.2: Chuẩn hóa năng lượng RMS**:
  - Áp dụng công thức chuẩn hóa theo từng burst: $r_{\text{norm}}[n] = r[n] / E_{\text{rms}}$ để triệt tiêu chênh lệch công suất giữa các modulation (tránh mô hình "ăn gian").
- [ ] **Bước 1.3: Phân chia Tập dữ liệu (Strict Partitioning)**:
  - Chia tập dữ liệu cứng theo tỷ lệ **70% Train / 15% Validation / 15% Test**.
  - Đảm bảo **Zero Data Leakage**: Tập Test không được trùng seed tạo tín hiệu hoặc thông số kênh truyền với tập Train.
- [ ] **Bước 1.4: Cấu hình Lưới SNR (15 điểm)**:
  - Chuẩn bị dữ liệu test quét qua 15 điểm SNR: `-10, -8, -6, -4, -2, 0, +2, +4, +6, +8, +10, +12, +14, +16, +18 dB`.

---

### GIAI ĐOẠN 2: Xây dựng Module Augmentation & Mô hình (Models & Transforms)
- [ ] **Bước 2.1: Lập trình Module Augmentation trong PyTorch**:
  - Tạo class `RFAugmentationPipeline` kế thừa từ PyTorch Dataset/Transform với các phép biến đổi ngẫu nhiên:
    1. **Random Phase Rotation**: Xoay pha ngẫu nhiên góc $\theta \in [0, 2\pi)$.
    2. **Additive Gaussian Noise Scaling**: Tiêm thêm nhiễu AWGN ngẫu nhiên ở các mức SNR thấp.
    3. **Time-shift Jitter**: Dịch mẫu thời gian ngẫu nhiên (dịch vòng/dịch trái phải vài mẫu).
    4. **Amplitude Scaling**: Co giãn biên độ tín hiệu ngẫu nhiên.
- [ ] **Bước 2.2: Xây dựng Mạng 1D CNN Baseline**:
  - Thiết kế kiến trúc 1D CNN chuẩn (ví dụ: 3 khối Convolutional 1D + BatchNorm + ReLU + MaxPool + Fully Connected).
  - Đảm bảo mô hình nhận input tensor `[N_batch, 2, 128]` và output ra xác suất các lớp modulation.

---

### GIAI ĐOẠN 3: Huấn luyện & Thực nghiệm So sánh (Training & Benchmarking)
- [ ] **Bước 3.1: Huấn luyện Mô hình A (Baseline 1D CNN)**:
  - Train trên tập dữ liệu gốc (không dùng Augmentation pipeline).
  - Cố định Random Seed để tái tạo kết quả. Lưu weights thành `baseline_model.pth`.
- [ ] **Bước 3.2: Huấn luyện Mô hình B (Augmented 1D CNN)**:
  - Train trên tập dữ liệu có áp dụng `RFAugmentationPipeline`.
  - **Giữ nguyên 100% Hyperparameters** (Learning rate, Batch size, Optimizer, Epochs) giống hệt Mô hình A.
  - Lưu weights thành `augmented_model.pth`.

---

### GIAI ĐOẠN 4: Đánh giá & Phân tích Nguyên nhân Lỗi (Evaluation & Failure Analysis)
- [ ] **Bước 4.1: Vẽ Đồ thị Accuracy vs. SNR**:
  - Chạy cả 2 mô hình trên tập Test qua 15 điểm SNR (-10 dB đến +18 dB).
  - Vẽ đường cong so sánh **Accuracy vs. SNR** của Mô hình A vs. Mô hình B trên cùng 1 biểu đồ.
- [ ] **Bước 4.2: Tính toán Gain Độ nhạy SNR (Sensitivity Gain)**:
  - Xác định mức SNR mà mô hình Augmented đạt mốc 80% accuracy so với mô hình Baseline (tính khoảng cải thiện bằng dB).
- [ ] **Bước 4.3: Xuất Confusion Matrices & Macro-F1**:
  - Tạo Ma trận nhầm lẫn (Row-normalized Confusion Matrix) tại 3 vùng SNR:
    - **Low SNR (-4 dB)**
    - **Medium SNR (+4 dB)**
    - **High SNR (+14 dB)**
  - Tính chỉ số **Macro-F1** cho từng mô hình.
- [ ] **Bước 4.4: Phân tích Nguyên nhân Lỗi (Root-Cause Failure Analysis)**:
  - Lý giải tại sao ở SNR = -4 dB, mô hình Baseline bị nhầm lẫn giữa các chòm sao (chẳng hạn 16QAM vs 64QAM hoặc QPSK vs 8PSK), trong khi mô hình Augmented lại kháng nhiễu tốt nhờ học được bất biến pha và nhiễu.

---

### GIAI ĐOẠN 5: Phát triển Demo Tương tác & Quay Video Dự phòng (Demo & Fail-Safe Video)
- [ ] **Bước 5.1: Xây dựng Web App Demo (Streamlit / Gradio)**:
  - Tạo ứng dụng web tương tác với các tính năng:
    - Cho phép nạp burst tín hiệu test ngẫu nhiên.
    - Thanh trượt (Slider) chỉnh SNR (từ -10 đến +18 dB) và thêm nhiễu/xoay pha live.
    - Hiển thị dự đoán **Side-by-Side** giữa Baseline Model và Augmented Model cùng với xác suất các lớp.
- [ ] **Bước 5.2: Quay Video MP4 Dự phòng (30–60s)**:
  - Quay lại thao tác chạy Web App Demo, thể hiện rõ đoạn tín hiệu nhiễu nặng ở -4 dB mà mô hình Augmented đoán đúng còn Baseline đoán sai.
  - Xuất file video `demo_backup.mp4`.

---

### GIAI ĐOẠN 6: Đóng gói Sản phẩm & Chuẩn bị Báo cáo (GitHub & Presentation)
- [ ] **Bước 6.1: Chuẩn bị GitHub Repository**:
  - Cấu trúc thư mục sạch vẽ:
    ```text
    ├── dataset/       # Code load & preprocess I/Q tensor
    ├── models/        # 1D CNN architecture & Augmentation transforms
    ├── utils/         # Đồ thị SNR, Confusion matrix, Metrics calculation
    ├── demo/          # Streamlit app (app.py)
    ├── requirements.txt
    └── README.md      # Hướng dẫn 1-command chạy code < 30s
    ```
- [ ] **Bước 6.2: Thiết kế Slide Thuyết trình (PPTX)**:
  - Sử dụng file template `EE136IU_Capstone_Mini_Project_AMC.pptx`.
  - Tuân thủ quy định Clean White Academic Theme (Font `Manrope`, `Inter`, `Consolas`).
  - **Nhúng trực tiếp file video `demo_backup.mp4` vào slide demo**.
  - Phân chia lượt nói đều giữa các thành viên (đảm bảo thuyết trình gói gọn trong **15 phút**).
- [ ] **Bước 6.3: Tổng duyệt & Tập luyện (15-Min Rehearsal)**:
  - Chạy thử phần trình bày 15 phút, canh giờ tránh bị phạt điểm overtime penalty.
  - Chuẩn bị câu hỏi phản biện Q&A (giải thích bản chất toán học của xoay pha $e^{j\theta}$ và tiêm nhiễu AWGN).
