# 📡 Script Thuyết Trình — EE136IU Capstone Midterm (Topic 4)
> **Thời gian mục tiêu**: ~15 phút | **Đề tài**: Data Augmentation & Noise Resilience for Low-SNR

---

## 🟦 SLIDE 1 — Giới thiệu Đề tài (1 phút)

> *"Chào thầy/cô và các bạn, nhóm chúng em trình bày Topic 4: Tăng cường dữ liệu trên miền vật lý để cải thiện khả năng nhận dạng điều chế tín hiệu ở vùng nhiễu nặng."*

**Vấn đề cốt lõi:**
- Trong thực tế, bộ thu không dây phải hoạt động ở môi trường nhiễu cao — **SNR từ -10 dB đến +2 dB**.
- Ở SNR thấp, dạng chòm sao bị méo mó nghiêm trọng → mô hình AI dễ dự đoán sai.
- **Câu hỏi nghiên cứu**: Làm thế nào để train 1D CNN kháng nhiễu tốt hơn mà **không cần thêm dữ liệu thật**?

**Hướng giải quyết:** Physical RF Data Augmentation — biến đổi dữ liệu I/Q ngay trên miền tín hiệu vật lý.

---

## 🟦 SLIDE 2 — Cấu trúc Dữ liệu & Tiền xử lý (2 phút)

> *"Đầu tiên, em trình bày về dữ liệu chúng em sử dụng."*

**Dữ liệu tổng hợp (Synthetic Dataset):**
- **18,000 bursts** tín hiệu I/Q baseband được sinh tổng hợp.
- **8 lớp modulation**: BPSK, QPSK, 8PSK, 16QAM, 64QAM, AM-DSB, FM, GFSK.
- **15 điểm SNR Grid**: từ -10 dB đến +18 dB (bước 2 dB).
- Mỗi burst có **128 mẫu phức** → tensor `[2, 128]` (hàng I và hàng Q).

**Tiền xử lý quan trọng:**
- **Chuẩn hóa RMS**: r_norm[n] = r[n] / E_rms → triệt tiêu chênh lệch công suất giữa các lớp.
- **Phân chia Zero-Leakage**: 70% Train / 15% Val / 15% Test — đảm bảo tập Test không bị rò seed.

---

## 🟦 SLIDE 3 — Pipeline Augmentation (3 phút)

> *"Đây là phần cốt lõi của đề tài — 4 phép biến đổi vật lý mà chúng em áp dụng."*

**Lý do cần Augmentation trên miền RF:**
- Tín hiệu thực tế bị ảnh hưởng bởi lệch pha, nhiễu thêm vào, trễ thời gian, và suy hao biên độ.
- Nếu chỉ train với dữ liệu sạch → mô hình **không học được tính bất biến** với những biến đổi này.

**4 phép biến đổi trong `RFAugmentationPipeline`:**

| # | Tên phép biến đổi | Ký hiệu toán học | Mô phỏng hiện tượng gì |
|---|---|---|---|
| 1 | **Random Phase Rotation** | s'[n] = s[n] * exp(j*θ), θ ~ U[0, 2π) | Lệch pha sóng mang (carrier offset) |
| 2 | **AWGN Injection** | s'[n] = s[n] + w[n], w ~ N(0, σ²) | Nhiễu Gaussian trắng cộng thêm |
| 3 | **Time-Shift Jitter** | Dịch vòng ngẫu nhiên vài mẫu | Trễ đồng bộ (timing offset) |
| 4 | **Amplitude Scaling** | s'[n] = α * s[n], α ~ U[0.8, 1.2] | Suy hao/tăng ích kênh truyền |

> ✅ *Xác suất kích hoạt mỗi phép: Phase 50%, AWGN 40%, Shift 40%, Scale 30% — áp dụng ngẫu nhiên mỗi batch.*

---

## 🟦 SLIDE 4 — Kiến trúc Mô hình 1D CNN (1.5 phút)

> *"Cả 2 mô hình đều dùng chung 1 kiến trúc — chỉ khác nhau ở dữ liệu training."*

**`ModulationClassifier1DCNN`** — kiến trúc 3 khối:
```
Input [N, 2, 128]
→ Conv1D(64) + BatchNorm + ReLU + MaxPool
→ Conv1D(128) + BatchNorm + ReLU + MaxPool
→ Conv1D(256) + BatchNorm + ReLU + MaxPool
→ Flatten → FC(256) → Dropout → FC(8)
→ Output: Softmax 8 lớp modulation
```
- **Tổng số tham số**: ~175,432 parameters
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR — T_max=25 epochs
- **Điểm khác biệt duy nhất**: Model A train trên `train_ds_raw`, Model B train trên `train_ds_augmented`.

---

## 🟦 SLIDE 5 — Kết quả Thực nghiệm (3 phút)

> *"Đây là kết quả so sánh sau 25 epochs huấn luyện."*

**Bảng so sánh tổng quan:**

| Tiêu chí | 🔴 Baseline 1D CNN | 🔵 Augmented 1D CNN | Gain |
|---|:---:|:---:|:---:|
| Validation Loss | 0.8757 | **0.7670** | ↓ 0.1087 |
| Validation Accuracy | 65.33% | **66.07%** | +0.74% |
| Accuracy @ SNR = -8 dB | 25.86% | **30.17%** | **+4.31%** |
| Accuracy @ SNR = -2 dB | 57.85% | **61.16%** | **+3.31%** |
| Bất biến pha | < 25% ❌ | **100% ✅** | Tuyệt đối |

**Nhận xét chính:**
- Ở **SNR cao (> +6 dB)**: cả 2 mô hình tương đương — chòm sao rõ ràng, nhiễu ít.
- Ở **SNR thấp (< 0 dB)**: Model B vượt trội rõ rệt — nhờ đã học được bất biến với nhiễu và lệch pha.
- **Kháng pha hoàn toàn**: Khi xoay pha bất kỳ góc θ, Baseline nhầm lẫn; Augmented vẫn đúng 100%.

---

## 🟦 SLIDE 6 — Phân tích Nguyên nhân Lỗi (2 phút)

> *"Tại sao Baseline bị nhầm? Chúng em phân tích qua Confusion Matrix."*

**Nhầm lẫn đặc trưng của Baseline ở SNR = -4 dB:**
- **16QAM ↔ 64QAM**: Khi nhiễu lớn, 16 điểm chòm sao bị phân tán thành dạng giống 64QAM.
- **QPSK ↔ 8PSK**: Lệch pha ngẫu nhiên làm 4 điểm QPSK trông như 8 điểm 8PSK.
- **FM ↔ AM-DSB**: Hai lớp này vốn đã có phổ tần tương tự nhau ở SNR thấp.

**Tại sao Augmented thoát được:**
- Đã thấy hàng nghìn biến thể xoay pha trong training → học được **tính đối xứng vòng tròn** của chòm sao.
- Đã thấy nhiều mức AWGN khác nhau → học được **biên quyết định bền vững hơn** (robust decision boundary).

---

## 🟦 SLIDE 7 — Demo Live (2 phút)

> *"Bây giờ em xin phép demo trực tiếp Web App Streamlit."*

**Demo flow (live trên `http://localhost:8501`):**

1. **Chọn modulation**: 16QAM (loại khó nhận dạng nhất).
2. **Kéo SNR xuống -4 dB** → thấy chòm sao bị méo trên Constellation Plot.
3. **Bật Phase Rotation** → Baseline ❌ SAI, Augmented ✅ ĐÚNG.
4. **Kéo SNR lên +10 dB** → cả 2 cùng đúng (SNR cao thì không có khoảng cách).
5. **Thử modulation khác** (BPSK / QPSK) ở SNR = -8 dB → tiếp tục thấy gap.

> 💡 *Nhấn mạnh: 2 model cùng kiến trúc, cùng hyperparameter — chỉ khác cách tổ chức training data.*

---

## 🟦 SLIDE 8 — Kết luận & Hướng phát triển (30 giây)

> *"Tóm lại, nhóm em đã chứng minh được:"*

✅ **Physical RF Augmentation hoạt động** — cải thiện đáng kể độ chính xác ở Low-SNR mà không cần thêm dữ liệu thật.

✅ **Bất biến pha hoàn toàn** — giải quyết được điểm yếu cốt lõi của Baseline.

**Hướng phát triển tiếp theo:**
- Áp dụng thêm Channel Fading (Rayleigh/Rician) vào Augmentation pipeline.
- Thử nghiệm với Transformer/LSTM thay vì 1D CNN.
- Mở rộng lên 11 lớp modulation (thêm AM-SSB, OOK, WBFM).

> *"Nhóm em xin hết phần trình bày. Kính mời thầy/cô và các bạn đặt câu hỏi."*

---

## 💬 Q&A — Câu hỏi thường gặp

**Q: Tại sao chọn 1D CNN thay vì 2D CNN hay Transformer?**
> 1D CNN phù hợp với chuỗi thời gian I/Q (dữ liệu 1 chiều), nhẹ hơn 2D CNN và nhanh hơn Transformer để validate trong scope Midterm.

**Q: Data augmentation ở đây có gian lận không (data leakage)?**
> Không. Augmentation chỉ áp dụng trên **tập Train** mỗi epoch (on-the-fly). Tập Val và Test giữ nguyên sạch.

**Q: `exp(j*θ)` trong code thể hiện như thế nào?**
> `noisy_complex * np.exp(1j * rad)` — nhân số phức với factor xoay pha → I' = I·cosθ - Q·sinθ, Q' = I·sinθ + Q·cosθ.

**Q: 18,000 mẫu có đủ không?**
> Với 8 lớp × 15 SNR × 150 mẫu = 18,000 bursts, đủ để validate concept trong Midterm. Production thường cần 10× nhiều hơn.

**Q: Val Accuracy chỉ tăng 0.74%, có đáng kể không?**
> Overall accuracy tăng ít vì ở High-SNR cả 2 đều đã rất tốt. Điểm quan trọng là **gain ở Low-SNR** (+4.31% tại -8 dB) và **bất biến pha hoàn toàn** — đây mới là mục tiêu của Topic 4.

---

*Script được tạo cho EE136IU Capstone Midterm Presentation — Topic 4*
