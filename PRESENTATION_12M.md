# Ý chính thuyết trình — 12 phút

## Slide 1 — Giới thiệu đề tài

- Data augmentation cho AMC ở SNR thấp.
- Mục tiêu: kiểm tra augmentation có thực sự cải thiện kết quả hay không.

## Slide 2 — Bài toán AMC

- Đầu vào: burst I/Q `[2, 128]`.
- Phân loại 8 modulation: PSK, QAM, PAM và FSK.
- Vùng khó: `−10 đến +2 dB`, noise che lấp cấu trúc tín hiệu.

## Slide 3 — Câu hỏi nghiên cứu

- Augmentation có tăng low-SNR accuracy không?
- Transform nào tốt nhất: phase, noise, shift, scale hay full?
- Kết quả có ổn định qua 5 training seed không?
- Chọn model trên validation; test chỉ dùng ở bước cuối.

## Slide 4 — Dataset và signal model

- `21,480` burst, 8 classes, 15 mức SNR từ `−10 đến +18 dB`.
- Frozen split: `12,600 train / 2,880 val / 6,000 test`.
- Pipeline: shaping → random phase → AWGN → I/Q → RMS normalization.
- Synthetic benchmark, chưa đại diện đầy đủ kênh RF thực.

## Slide 5 — Thiết kế thực nghiệm

- 5 paired seed: `42–46`.
- Trong cùng seed: cùng initial weights, shuffle, CNN và hyperparameters.
- Validation chọn transform và checkpoint; không dùng test để model selection.
- Báo cáo mean ± sample SD qua 5 seed.

## Slide 6 — Các augmentation

- Phase rotation: góc hằng ngẫu nhiên cho toàn burst.
- Added AWGN: giảm expected SNR từ `0–6 dB`.
- Circular shift: tối đa `±8 samples`.
- Amplitude scaling: gain `0.85–1.15`.
- Constant phase offset không phải CFO.

## Slide 7 — CNN và training

- CNN 1D, `175,432` trainable parameters.
- Adam, learning rate `10⁻³`, 25 epoch, batch size 64.
- Cùng training recipe cho mọi cấu hình.
- Checkpoint theo validation accuracy trong vùng `−10 đến +2 dB`.

## Slide 8 — Validation ablation

- `phase_only`: **50.73%**, cao nhất.
- `noise_only`: 50.57% — chỉ thấp hơn `0.16 pp`.
- Khóa `phase_only` trước khi mở final test.
- Không kết luận phase rotation luôn tốt hơn noise augmentation.

## Slide 9 — Final test

- Global accuracy: `66.60% → 69.56%` — **+2.96 pp**.
- Macro-F1: `65.75% → 68.96%` — **+3.20 pp**.
- Low-SNR accuracy: `46.44% → 51.16%` — **+4.72 pp**.
- Gain lớn nhất ở vùng thấp: `−8 dB: +5.40 pp`, `−4 dB: +5.10 pp`.
- Global-accuracy gain dương ở cả 5 seed.

## Slide 10 — Phân tích confusion matrix

- `−4 dB`: CPFSK `46→70%`, GFSK `56→76%`.
- `+4 dB`: BPSK, PAM4, CPFSK và GFSK đạt `100%` với phase-only.
- Trade-off: QPSK `58→72%`, nhưng 8PSK `44→34%`, 64QAM `52→42%`.
- Hiệu năng tổng thể tăng, nhưng không phải mọi class đều tăng.

## Slide 11 — Interactive demo

- Chọn checkpoint seed 42 và cấu hình `phase_only`.
- Sinh burst `16QAM`, SNR `−4 dB`.
- Thay phase offset `0° → 90°`.
- So sánh dự đoán baseline và phase-only.
- Demo chỉ minh họa; kết luận dựa trên 6,000 test burst và 5 seed.

## Slide 12 — Kết luận và giới hạn

- Phase-only được chọn bằng validation và tăng low-SNR accuracy **4.72 pp**.
- Kết quả dựa trên leakage-free paired protocol và frozen synthetic benchmark.
- Chưa chứng minh khả năng chống CFO, fading hoặc multipath.
- Chưa chứng minh generalization sang dữ liệu RF thực.
- Hướng tiếp theo: realistic channel, nhiều dataset seed và over-the-air data.
