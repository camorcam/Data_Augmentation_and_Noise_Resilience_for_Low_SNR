# Checklist Kiểm tra Tổng duyệt trước Buổi Bảo vệ Midterm (Pre-Defense Verification Checklist)
**Môn học**: EE136IU - Machine Learning for Wireless Communications  
**Đề tài Topic 4**: Data Augmentation & Noise Resilience for Low-SNR  

---

## 1. Kiểm tra Kỹ thuật & Mô hình hóa (Technical & Modeling Verification)
- [x] **Phạm vi Đề tài (Topic Scope)**: Đã hoàn thành 100% nhiệm vụ Topic 4 (RFAugmentationPipeline với Phase Rotation, AWGN Noise, Time Shift Jitter, Amplitude Scaling).
- [x] **Định dạng Baseband Tensor**: Tensor I/Q đạt chuẩn shape `[N_batch, 2, 128]` với chuẩn hóa RMS $E_{\text{rms}} = 1.0$.
- [x] **Đánh giá Lưới SNR Grid**: Đã benchmark cả 2 mô hình qua đủ 15 điểm SNR từ `-10 dB` đến `+18 dB` (bước 2 dB).
- [x] **Ma trận Nhầm lẫn & Macro-F1**: Đã tính toán và vẽ Confusion Matrices chuẩn hóa theo dòng tại Low SNR (-4 dB) và High SNR (+14 dB).
- [x] **Phân tích Nguyên nhân Lỗi (Failure Analysis)**: Có báo cáo giải thích bản chất vật lý (sự biến dạng chòm sao I/Q và bất biến pha) tại file `phase4_failure_analysis_report.md`.

---

## 2. Kiểm tra Vận hành & Slide Thuyết trình (Operational & Presentation Verification)
- [x] **Ứng dụng Web Demo (Streamlit)**: App demo tương tác đã sẵn sàng, chạy bằng 1 dòng lệnh `streamlit run app.py`.
- [x] **Video Dự phòng (Fail-Safe Video Backup)**: Đã có kịch bản quay video 30–60s nhúng vào slide để chống sự cố đơ máy khi demo live.
- [x] **Chuẩn thiết kế Slide (Template Compliance)**: Slide thuyết trình làm 100% trên mẫu chuẩn `EE136IU_Capstone_Mini_Project_AMC.pptx` với màu Clean White Academic.
- [x] **Khung thời gian (15-Minute Limit)**: Đã phân chia thời lượng phát biểu bằng nhau giữa các thành viên, khớp chính xác 15 phút Slide + 10 phút Demo + 5 phút Q&A.
- [x] **GitHub Reproducibility**: Repository có cấu trúc mô-đun sạch vẽ, cố định seed ngẫu nhiên (`seed=42`) và file `README.md` tái tạo kết quả < 30 giây.
