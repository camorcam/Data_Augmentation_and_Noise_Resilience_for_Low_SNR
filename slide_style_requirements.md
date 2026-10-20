# Quy định & Hướng dẫn Thiết kế Slide Thuyết trình (Slide Style Requirements)
**Môn học**: EE136IU - Machine Learning for Wireless Communications  
**Dự án**: Capstone Mini-Project AMC  

---

## 1. Mẫu Slide Cơ sở (Base Template)
- **File Template bắt buộc**: `EE136IU_Capstone_Mini_Project_AMC.pptx`
- Tất cả các slide thuyết trình của nhóm phải được xây dựng dựa trên file mẫu chuẩn này, không tự ý đổi layout gốc hoặc sử dụng template bên ngoài.

---

## 2. Quy chuẩn Màu sắc (Color Palette & Theme)
Slide sử dụng phong cách **Clean White Academic Theme** chuẩn học thuật chuyên nghiệp:

| Thành phần | Mã màu Hex | Mô tả & Công dụng |
| :--- | :--- | :--- |
| **Nền Canvas (Background)** | `#FFFFFF` | Nền trắng tinh 100% (Solid White Canvas) |
| **Chữ chính (Body Text)** | `#0F172A` | Chữ xám đen đậm (Dark Charcoal) tạo độ tương phản cao |
| **Thẻ chứa (Container Cards)**| `#F8FAFC` | Nền thẻ màu ghi sáng (Light Slate) để phân khu nội dung |
| **Viền thẻ (Card Border)** | `#E2E8F0` | Viền xám mảnh (Subtle Border) để tách biệt các khái niệm |
| **Điểm nhấn chính (Primary)** | `#0284C7` | Xanh Tech Blue cho từ khóa, điểm nhấn |
| **Điểm nhấn phụ (Secondary)** | `#2563EB` | Xanh Deep Blue cho badge, mốc tiến trình, điểm quan trọng |

---

## 3. Quy chuẩn Font chữ (Typography Hierarchy)

- **Tiêu đề Slide & Tiêu đề Thẻ (Slide Titles & Card Headers)**:
  - Font: **Manrope**
  - Định dạng: Bold / Semi-bold, phân cấp rõ ràng với nội dung bên dưới.
- **Nội dung văn bản chính (Body Text & Bullet Points)**:
  - Font: **Inter**
  - Định dạng: Regular / Medium, kích thước chữ vừa phải, dễ đọc từ xa.
- **Công thức Toán, Tensors & Code Snippets**:
  - Font: **Consolas**
  - Định dạng: Monospace, dùng cho các biểu thức toán học ($r[n] = hs[n] + w[n]$), kích thước tensor (`[N_batch, 2, 128]`) và đoạn mã nguồn.

---

## 4. Nguyên tắc Bố cục & Trực quan hóa (Visual Structure Rules)

1. **Tuyệt đối Không dùng Khối chữ Dày đặc (No Walls of Text)**:
   - Cấm chép nguyên đoạn văn dài vào slide.
   - Tất cả khái niệm phức tạp phải được chia nhỏ thành các **Thẻ chứa (Container Cards)**, bảng so sánh (Comparison tables), sơ đồ khối (Block diagrams) hoặc danh sách ngắn gọn (Bullet points).
2. **Trực quan hóa Dữ liệu (Data Visualization Standard)**:
   - Các biểu đồ đường cong hiệu năng (**Accuracy vs. SNR**) phải có độ tương phản cao, đường nét rõ ràng và có gán nhãn (labels/legends) đầy đủ.
   - Ma trận nhầm lẫn (**Confusion Matrix**) phải được chuẩn hóa theo dòng (row-normalized).
   - Đồ thị quá trình huấn luyện (**Loss Curves**) và biểu đồ Macro-F1.
3. **Thành phần Bắt buộc trên Footer (Global Footers)**:
   - Mỗi slide (trừ slide tiêu đề) bắt buộc phải giữ lại:
     - Góc dưới bên trái: Breadcrumb môn học (`EE136: Machine Learning for Wireless Communications`).
     - Góc dưới bên phải: Số trang slide chuẩn (`Slide X of Y`).

---

## 5. Yêu cầu Nhúng Video Demo Dự phòng (Fail-Safe Video Embedding)

- Slide trình diễn Demo bắt buộc phải **nhúng trực tiếp 1 file video MP4** (thời lượng 30–60 giây).
- Video quay màn hình thao tác chạy thực tế ứng dụng Demo (Streamlit / Gradio / CLI).
- **Mục đích**: Bảo vệ nhóm khỏi rủi ro trừ điểm khi trình diễn live gặp sự cố (lỗi terminal, crash GPU, mất kết nối hoặc đơ máy).

---

## 6. Checklist Kiểm tra Slide trước buổi Bảo vệ (Pre-Defense Slide Checklist)

- [ ] Slide xây dựng 100% trên template `EE136IU_Capstone_Mini_Project_AMC.pptx`.
- [ ] Nền trắng tinh `#FFFFFF`, không dùng background màu tối hoặc gradient sặc sỡ.
- [ ] Đúng quy chuẩn font chữ: `Manrope` (Title), `Inter` (Body), `Consolas` (Math/Code).
- [ ] Không có slide nào chứa khối văn bản (wall of text) quá 4 dòng liên tục.
- [ ] Đồ thị SNR, Confusion Matrix và Loss Curve sắc nét, tương phản cao, gán nhãn đầy đủ.
- [ ] Đã nhúng sẵn video MP4 demo 30–60s vào slide tương ứng.
- [ ] Đã phân chia rõ ràng phần phát biểu của từng thành viên trong nhóm trên slide.
