# Project overview — EE136IU Topic 4

## Research question

This project tests whether physically motivated augmentation improves automatic
modulation classification for synthetic low-SNR I/Q bursts. It compares the
same 1D CNN trained with no augmentation against controlled augmentation
variants. The result is evidence for this generator and protocol, not a claim
of real-world RF generalization.

## Reproducible protocol

- Eight classes: BPSK, QPSK, 8PSK, 16QAM, 64QAM, PAM4, CPFSK and GFSK.
- Fifteen SNR points from -10 to +18 dB; each input is `[2,128]` and normalized
  to unit per-burst RMS before training-time augmentation.
- One persisted dataset: 12,600 train / 2,880 validation / 6,000 test bursts.
  Independent RNG streams, checksums and exact-burst hash audits found zero
  duplicates among 21,480 bursts.
- Five paired training seeds (42--46), identical initial weights and shuffle
  order per configuration, 25 epochs, Adam and cosine learning-rate schedule.
- Checkpoints are selected by mean validation accuracy from -10 to +2 dB.
  The test set is not used to select augmentation.

## Ablation selected on validation

| Configuration | Mean low-SNR validation accuracy | SD across seeds |
|---|---:|---:|
| Baseline | 47.80% | 0.42 pp |
| Phase only | **50.73%** | 1.40 pp |
| Noise only | 50.57% | 0.24 pp |
| Shift only | 48.02% | 0.99 pp |
| Scale only | 47.37% | 0.88 pp |
| All four transforms | 49.42% | 1.47 pp |

`phase_only` was frozen as the augmented candidate because it had the highest
mean low-SNR validation accuracy. Its margin over `noise_only` is only 0.16
percentage points, so the ablation does not establish that phase rotation is
intrinsically superior.

## Final frozen-test result

| Metric | Baseline mean +/- SD | Phase-only mean +/- SD | Mean paired gain |
|---|---:|---:|---:|
| Global accuracy | 66.60 +/- 1.55% | **69.56 +/- 1.24%** | **+2.96 pp** |
| Global Macro-F1 | 65.75 +/- 1.66% | **68.96 +/- 1.30%** | **+3.20 pp** |
| Accuracy, -10 to +2 dB | 46.44 +/- 0.99% | **51.16 +/- 0.55%** | **+4.72 pp** |

The paired global-accuracy gain was positive for all five training seeds. The
mean SNR80 gain was 1.37 dB, but its across-seed SD was 2.18 dB and one seed had
a -1.60 dB gain; therefore sensitivity improvement is not consistent enough to
claim as guaranteed.

## Corrections to the old overview

- Original bursts already use random initial phase. Phase augmentation adds
  another constant phase offset; it is not carrier-frequency offset (CFO).
- The actual classifier head is `256 -> 128 -> 8`, dropout is 0.3, and the third
  convolution is followed directly by adaptive average pooling. The network
  outputs logits; softmax is applied only for demo display.
- GFSK now uses Gaussian pulse shaping with BT=0.3 instead of a three-tap moving
  average.
- Added-noise augmentation controls an expected SNR reduction in dB rather than
  an unexplained absolute standard deviation.
- Full augmentation was not the best validation configuration. The defensible
  conclusion is that the validation-selected phase-only model performed better
  than baseline on this frozen synthetic test set.

## Symbols and abbreviations

| Symbol / abbreviation | Meaning in this project |
|---|---|
| (N) | Number of I/Q bursts in a dataset or batch, depending on context. |
| (L) | Number of time samples in one burst; here (L=128). |
| `[N, 2, 128]` | Input tensor shape: (N) bursts, two I/Q channels and 128 samples per channel. |
| (I[n]), (Q[n]) | In-phase and quadrature samples at time index (n). |
| (s[n]=I[n]+jQ[n]) | Complex baseband representation of one I/Q sample. |
| (j) | Imaginary unit, (j^2=-1). |
| (n) | Discrete-time sample index. |
| RMS | Root-mean-square magnitude of a burst: ​\(\sqrt{\operatorname{mean}(I^2+Q^2)}\). Inputs are divided by this value. |
| SNR | Signal-to-noise power ratio. |
| SNR (dB) | (10\log_{10}(P_s/P_n)), where (P_s) and (P_n) are signal and noise power. |
| (P_s), (P_n) | Mean signal power and mean complex-noise power. |
| \(\rho\) | Linear SNR, \(\rho=10^{\mathrm{SNR}_{\mathrm{dB}}/10}\). |
| \(\sigma^2\) | Noise variance. For circular complex AWGN, each I/Q component receives half of the total complex-noise power. |
| AWGN | Additive white Gaussian noise. |
| \(\theta\) | Constant phase-rotation angle; phase augmentation applies \(s'[n]=s[n]e^{j\theta}\). |
| \(\alpha\) | Amplitude-scaling factor; scaling applies \(s'[n]=\alpha s[n]\). |
| \(k\) | Integer circular time-shift, in samples. |
| (p) | Probability that an augmentation transform is applied to a training sample. |
| \(\Delta\mathrm{SNR}\) | Random SNR reduction used by noise augmentation; configured in the range 0–6 dB. |
| BT | Gaussian-filter bandwidth–symbol-period product; GFSK uses BT = 0.3. |
| sps | Samples per symbol; GFSK/CPFSK use 4 sps and PSK/QAM/PAM use 8 sps in this generator. |
| (h) | FSK modulation index; CPFSK and GFSK use (h=0.5). |
| CNN | Convolutional neural network. |
| Logit | Raw, unnormalized output score produced by the classifier for one class. |
| Accuracy | Fraction of correctly classified bursts. |
| Macro-F1 | Arithmetic mean of the per-class F1 scores, with all eight classes included equally. |
| SD | Sample standard deviation across the five training seeds. |
| pp | Percentage points; for example, 69% − 66% = 3 pp. |
| SNR80 | First upward crossing of 80% accuracy versus SNR, linearly interpolated between adjacent grid points without smoothing or extrapolation. |
| CFO | Carrier-frequency offset. A constant phase rotation is not CFO; CFO creates a phase ramp over time. |
| RNG | Random-number generator. Separate RNG streams are used to control partitioning, shuffling, initialization and augmentation. |

## Artifacts

- Dataset provenance: `data/frozen_v2/manifest.json`
- Per-run checkpoints/config/history: `runs/ablation/seed_*/`
- Test metrics and predictions: `results/final_phase/`
- Figures: `results/final_phase/plots/`

---

# Tổng quan dự án — EE136IU Topic 4

## Câu hỏi nghiên cứu

Dự án kiểm tra liệu các phép tăng cường dữ liệu dựa trên hiện tượng vật lý có
giúp cải thiện bài toán nhận dạng điều chế tự động trên các burst I/Q tổng hợp ở
SNR thấp hay không. Thí nghiệm so sánh cùng một kiến trúc CNN 1D khi huấn luyện
không augmentation với các biến thể augmentation được kiểm soát. Kết quả chỉ là
bằng chứng trong phạm vi bộ sinh dữ liệu và quy trình thực nghiệm này, không phải
kết luận về khả năng tổng quát hóa trên tín hiệu RF thực tế.

## Quy trình thực nghiệm có thể tái lập

- Tám lớp điều chế: BPSK, QPSK, 8PSK, 16QAM, 64QAM, PAM4, CPFSK và GFSK.
- Mười lăm mức SNR từ -10 đến +18 dB. Mỗi đầu vào có kích thước `[2,128]` và
  được chuẩn hóa RMS riêng theo từng burst trước augmentation trong lúc train.
- Một bộ dữ liệu cố định gồm 12.600 burst train, 2.880 burst validation và 6.000
  burst test. Các split dùng luồng RNG độc lập. Kiểm tra checksum và hash chính
  xác nhận không có burst trùng trong tổng số 21.480 burst.
- Năm seed huấn luyện ghép cặp: 42–46. Trong mỗi seed, các cấu hình có cùng trọng
  số khởi tạo và thứ tự shuffle. Mỗi model được train 25 epoch bằng Adam và
  cosine learning-rate schedule.
- Checkpoint được chọn theo accuracy validation trung bình tại các mức SNR từ
  -10 đến +2 dB. Test set không được dùng để lựa chọn augmentation.

## Ablation và lựa chọn trên validation

| Cấu hình | Accuracy validation low-SNR trung bình | SD giữa các seed |
|---|---:|---:|
| Baseline | 47,80% | 0,42 điểm % |
| Chỉ xoay pha | **50,73%** | 1,40 điểm % |
| Chỉ thêm nhiễu | 50,57% | 0,24 điểm % |
| Chỉ dịch thời gian | 48,02% | 0,99 điểm % |
| Chỉ scale biên độ | 47,37% | 0,88 điểm % |
| Kết hợp cả bốn transform | 49,42% | 1,47 điểm % |

`phase_only` được khóa làm ứng viên augmented vì đạt accuracy low-SNR trung bình
cao nhất trên validation. Tuy nhiên, nó chỉ hơn `noise_only` 0,16 điểm phần trăm,
do đó thí nghiệm chưa chứng minh phép xoay pha về bản chất tốt hơn phép thêm
nhiễu.

## Kết quả cuối trên frozen test set

| Metric | Baseline, trung bình ± SD | Phase-only, trung bình ± SD | Gain ghép cặp trung bình |
|---|---:|---:|---:|
| Accuracy toàn cục | 66,60 ± 1,55% | **69,56 ± 1,24%** | **+2,96 điểm %** |
| Macro-F1 toàn cục | 65,75 ± 1,66% | **68,96 ± 1,30%** | **+3,20 điểm %** |
| Accuracy từ -10 đến +2 dB | 46,44 ± 0,99% | **51,16 ± 0,55%** | **+4,72 điểm %** |

Gain accuracy toàn cục theo cặp là dương ở cả năm seed huấn luyện. Gain SNR80
trung bình là 1,37 dB, nhưng SD giữa các seed lên đến 2,18 dB và một seed có gain
-1,60 dB. Vì vậy, chưa thể khẳng định augmentation luôn cải thiện độ nhạy SNR.

## Những điểm đã sửa so với overview cũ

- Dữ liệu gốc vốn đã có pha ban đầu ngẫu nhiên. Phase augmentation chỉ thêm một
  độ lệch pha hằng; đây không phải carrier-frequency offset (CFO).
- Classifier head thực tế là `256 → 128 → 8`, dropout là 0,3 và convolution thứ
  ba được nối trực tiếp với adaptive average pooling. Mạng trả về logits;
  softmax chỉ được áp dụng khi hiển thị demo.
- GFSK hiện dùng Gaussian pulse shaping với BT=0,3 thay cho moving average ba
  tap.
- Noise augmentation kiểm soát mức giảm SNR kỳ vọng theo dB thay vì dùng một độ
  lệch chuẩn tuyệt đối khó diễn giải.
- Cấu hình kết hợp cả bốn transform không đứng đầu validation. Kết luận có thể
  bảo vệ là: model phase-only được chọn bằng validation hoạt động tốt hơn
  baseline trên frozen synthetic test set này.

## Ký hiệu và từ viết tắt

| Ký hiệu / từ viết tắt | Ý nghĩa trong dự án |
|---|---|
| (N) | Số burst I/Q trong dataset hoặc batch, tùy ngữ cảnh. |
| (L) | Số mẫu thời gian trong một burst; trong dự án này (L=128). |
| `[N, 2, 128]` | Kích thước tensor đầu vào: (N) burst, hai kênh I/Q và 128 mẫu trên mỗi kênh. |
| (I[n]), (Q[n]) | Thành phần đồng pha và vuông pha tại chỉ số thời gian (n). |
| (s[n]=I[n]+jQ[n]) | Biểu diễn baseband phức của một mẫu I/Q. |
| (j) | Đơn vị ảo, (j^2=-1). |
| (n) | Chỉ số thời gian rời rạc của mẫu. |
| RMS | Căn trung bình bình phương biên độ burst: ​\(\sqrt{\operatorname{mean}(I^2+Q^2)}\). Đầu vào được chia cho giá trị này. |
| SNR | Tỷ số công suất tín hiệu trên công suất nhiễu. |
| SNR (dB) | (10\log_{10}(P_s/P_n)), trong đó (P_s) và (P_n) là công suất tín hiệu và nhiễu. |
| (P_s), (P_n) | Công suất tín hiệu trung bình và công suất nhiễu phức trung bình. |
| \(\rho\) | SNR ở thang tuyến tính, \(\rho=10^{\mathrm{SNR}_{\mathrm{dB}}/10}\). |
| \(\sigma^2\) | Phương sai nhiễu. Với AWGN phức tròn, mỗi thành phần I/Q nhận một nửa tổng công suất nhiễu phức. |
| AWGN | Nhiễu Gaussian trắng cộng tính. |
| \(\theta\) | Góc xoay pha hằng; phase augmentation áp dụng \(s'[n]=s[n]e^{j\theta}\). |
| \(\alpha\) | Hệ số scale biên độ; phép biến đổi áp dụng \(s'[n]=\alpha s[n]\). |
| \(k\) | Số mẫu dịch vòng theo thời gian, có giá trị nguyên. |
| (p) | Xác suất một phép augmentation được áp dụng lên mẫu train. |
| \(\Delta\mathrm{SNR}\) | Mức giảm SNR ngẫu nhiên của noise augmentation; được cấu hình trong khoảng 0–6 dB. |
| BT | Tích băng thông bộ lọc Gaussian và chu kỳ symbol; GFSK dùng BT = 0,3. |
| sps | Số mẫu trên mỗi symbol; bộ sinh dùng 4 sps cho GFSK/CPFSK và 8 sps cho PSK/QAM/PAM. |
| (h) | Chỉ số điều chế FSK; CPFSK và GFSK dùng (h=0,5). |
| CNN | Mạng nơ-ron tích chập. |
| Logit | Điểm số thô, chưa chuẩn hóa mà classifier tạo ra cho từng lớp. |
| Accuracy | Tỷ lệ burst được phân loại đúng. |
| Macro-F1 | Trung bình cộng F1 của từng lớp, trong đó cả tám lớp có trọng số bằng nhau. |
| SD | Độ lệch chuẩn mẫu giữa năm seed huấn luyện. |
| Điểm % (pp) | Chênh lệch tuyệt đối giữa hai tỷ lệ phần trăm; ví dụ 69% − 66% = 3 điểm %. |
| SNR80 | Điểm cắt hướng lên đầu tiên tại accuracy 80% theo SNR, nội suy tuyến tính giữa hai mức SNR kề nhau, không làm mượt hay ngoại suy. |
| CFO | Độ lệch tần số sóng mang. Xoay pha hằng không phải CFO; CFO tạo ra pha thay đổi tuyến tính theo thời gian. |
| RNG | Bộ sinh số ngẫu nhiên. Dự án dùng các luồng RNG riêng cho chia dữ liệu, shuffle, khởi tạo và augmentation. |

## Giới hạn khi diễn giải kết quả

- SD được tính qua năm seed huấn luyện trên cùng một test set tổng hợp; đây không
  phải confidence interval và không đo bất định giữa nhiều dataset độc lập.
- Bộ sinh PSK/QAM/PAM dùng xung chữ nhật đơn giản. Tốc độ symbol khác nhau giữa
  các họ điều chế có thể trở thành đặc trưng tắt để model khai thác.
- Circular time shift chỉ là xấp xỉ của sai lệch timing; amplitude scaling giữ
  nguyên SNR vì đồng thời scale cả tín hiệu và nhiễu.
- Kết quả không chứng minh khả năng chống CFO, fading, multipath, hardware
  impairment hay khả năng tổng quát hóa sang dữ liệu thu ngoài thực tế.
- Softmax confidence trong demo chưa được hiệu chuẩn thành xác suất dự đoán đúng.

## Artifact

- Nguồn gốc dataset: `data/frozen_v2/manifest.json`
- Checkpoint, cấu hình và lịch sử từng run: `runs/ablation/seed_*/`
- Metric và prediction trên test: `results/final_phase/`
- Biểu đồ: `results/final_phase/plots/`
