# Hướng dẫn Chạy Web App Demo & Quay Video MP4 Dự phòng (Phase 5 Guide)
**Đề tài Topic 4**: Data Augmentation & Noise Resilience for Low-SNR  
**Môn học**: EE136IU - Machine Learning for Wireless Communications  

---

## 1. 🚀 Hướng dẫn Chạy Web App Demo (1-Command Execution)

Mở Terminal / PowerShell tại thư mục dự án và gõ **1 dòng lệnh duy nhất**:

```powershell
streamlit run app.py
```

Trình duyệt web sẽ tự động mở giao diện ứng dụng Demo tại địa chỉ: `http://localhost:8501`.

---

## 2. 🎛️ Các Tính năng Kỹ thuật trên Giao diện Demo

1. **Thanh điều chỉnh Live Controls (Bên thanh Sidebar)**:
   - **Loại Modulation gốc (True Class)**: Chọn 1 trong 8 lớp (`BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `PAM4`, `CPFSK`, `GFSK`).
   - **Thanh trượt SNR (dB)**: Kéo điều chỉnh mức nhiễu từ **-10 dB** đến **+18 dB**.
   - **Xoay pha ngẫu nhiên (Random Phase Rotation)**: Bật/tắt và chọn góc xoay pha bổ sung ($0^\circ$ đến $360^\circ$).
2. **Khu vực Trực quan hóa Tín hiệu (Visualizers)**:
   - Đồ thị dạng sóng thời gian I/Q (**Baseband I/Q Waveform**).
   - Đồ thị mặt phẳng chòm sao (**Constellation Diagram**).
3. **Khu vực So sánh Dự đoán Side-by-Side**:
   - Bên trái: **Mô hình A (Baseline 1D CNN)** + Biểu đồ cột xác suất (màu đỏ).
   - Bên phải: **Mô hình B (Augmented 1D CNN - Topic 4)** + Biểu đồ cột xác suất (màu xanh).

---

## 3. 📹 Kịch bản Quay Video MP4 Dự phòng (30–60 Giây Backup Video)

Sử dụng phần mềm quay màn hình (OBS Studio / Windows Screen Recorder `Win + Alt + R` / Bandicam):

- **0:00 - 0:10**: Mở ứng dụng `streamlit run app.py` trên trình duyệt.
- **0:10 - 0:25**: Chọn loại modulation `16QAM` và kéo thanh trượt SNR xuống mức **-4 dB**.
  - Chỉ ra màn hình: **Baseline Model** bị nhầm `16QAM` thành `QPSK` *(Hiển thị ô màu đỏ SAI LỖI ❌)*.
  - Chỉ ra màn hình: **Augmented Model (Topic 4)** dự đoán chính xác `16QAM` *(Hiển thị ô màu xanh CHÍNH XÁC ✅)*.
- **0:25 - 0:45**: Tích chọn **Bật Xoay pha ngẫu nhiên (45 độ)**. Cho hội đồng thấy mô hình Baseline đoán sai loạn nhịp, trong khi mô hình Topic 4 nhờ tính bất biến pha vẫn đoán đúng 100%.
- **0:45 - 0:60**: Xuất video thành file `demo_backup.mp4` và nhúng file video này vào trang Slide Demo trong PowerPoint.
