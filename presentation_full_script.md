# Presentation Script — Natural English / Tiếng Việt

The wording below is intentionally conversational. It focuses on the problem, the fairness of the experiment, and what the results mean. Slides 13–16 are backup slides and should only be discussed when a question leads to them.

## Slide 1 — Introduction

### English

Good morning/afternoon. We are the team behind this project on automatic modulation classification. Our main question is simple: when the received signal is very noisy, can data augmentation help the model make better decisions? We designed a controlled experiment to answer that question as fairly as possible.

### Tiếng Việt

Em xin chào thầy/cô và các bạn. Nhóm em thực hiện đề tài về nhận dạng điều chế tự động. Câu hỏi chính của nhóm khá đơn giản: khi tín hiệu thu được có rất nhiều nhiễu, data augmentation có giúp mô hình dự đoán tốt hơn không? Nhóm đã thiết kế một thí nghiệm có kiểm soát để trả lời câu hỏi này một cách công bằng nhất có thể.

## Slide 2 — The Problem

### English

The model receives a short wireless signal and has to identify its modulation type. When the signal is clean, the patterns are easier to recognize. At low SNR, noise hides those patterns, so different modulation types can look very similar. This difficult low-SNR region is where we want augmentation to help.

### Tiếng Việt

Mô hình nhận một đoạn tín hiệu vô tuyến ngắn và phải xác định loại điều chế. Khi tín hiệu sạch, các đặc trưng khá dễ nhận ra. Nhưng ở SNR thấp, nhiễu che mất những đặc trưng đó, khiến nhiều loại điều chế trông rất giống nhau. Đây chính là vùng khó mà nhóm muốn augmentation cải thiện.

## Slide 3 — What We Want to Find Out

### English

We focus on three things: whether augmentation helps, which type works best in our setup, and whether the improvement is consistent across repeated training runs. Most importantly, we choose the method using validation data. The test data is kept separate until the final comparison, so the reported result is not selected to look good on the test set.

### Tiếng Việt

Nhóm tập trung vào ba điều: augmentation có giúp hay không, loại nào phù hợp nhất trong thiết lập này, và mức cải thiện có lặp lại qua nhiều lần train hay không. Quan trọng nhất, nhóm chỉ dùng validation để chọn phương pháp. Test set được giữ riêng đến bước đánh giá cuối, nên kết quả không bị chọn theo hướng có lợi trên test set.

## Slide 4 — Dataset

### English

We use a synthetic dataset covering eight modulation classes and a wide range of signal quality. The data is split once into training, validation, and test sets, and those splits never change. The signals include phase variation and noise, but this is still a simplified environment. It does not represent every effect found in a real radio channel.

### Tiếng Việt

Nhóm dùng một dataset synthetic gồm tám lớp điều chế và nhiều mức chất lượng tín hiệu. Dữ liệu được chia một lần thành train, validation và test, sau đó giữ cố định trong toàn bộ thí nghiệm. Tín hiệu có biến đổi phase và noise, nhưng đây vẫn là môi trường đơn giản hóa, chưa bao quát mọi ảnh hưởng của kênh vô tuyến thực tế.

## Slide 5 — A Fair Comparison

### English

This is the key part of our method. For each repeated run, the baseline and augmented model start under the same conditions. They use the same network, the same training settings, and matching randomness. The meaningful difference is whether augmentation is applied. This paired design helps us attribute any consistent change to augmentation rather than to a lucky training run.

### Tiếng Việt

Đây là phần quan trọng nhất trong phương pháp. Ở mỗi lần lặp, baseline và mô hình có augmentation bắt đầu trong cùng điều kiện. Hai bên dùng cùng network, cùng thiết lập train và randomness tương ứng. Khác biệt có ý nghĩa là có áp dụng augmentation hay không. Thiết kế paired này giúp nhóm phân biệt hiệu quả thật của augmentation với một lần train may mắn.

## Slide 6 — Augmentation Methods

### English

We test four simple changes that may happen to a received signal: a phase rotation, extra noise, a small time shift, and a change in amplitude. We first test them separately, then test their combination. One important limitation is that our phase rotation is constant across a burst; it should not be interpreted as full carrier-frequency-offset simulation.

### Tiếng Việt

Nhóm thử bốn thay đổi đơn giản có thể xuất hiện ở tín hiệu thu: xoay phase, thêm noise, dịch nhẹ theo thời gian và thay đổi biên độ. Nhóm đánh giá từng phép biến đổi riêng trước, sau đó mới thử kết hợp. Một giới hạn cần lưu ý là phase rotation ở đây không đổi trong cả burst, nên không thể xem là mô phỏng đầy đủ CFO.

## Slide 7 — Model and Training

### English

All experiments use the same compact convolutional model and the same training recipe. We do not change the model to favor one setup. The only intended change is the training data seen by the model. We also select the best checkpoint based on performance in the low-SNR region, because that is the region our project is trying to improve.

### Tiếng Việt

Tất cả thí nghiệm dùng cùng một mô hình convolutional gọn nhẹ và cùng quy trình train. Nhóm không thay đổi mô hình để ưu tiên một cấu hình nào. Khác biệt chủ đích chỉ nằm ở dữ liệu mà mô hình nhìn thấy khi train. Checkpoint cũng được chọn theo kết quả ở vùng SNR thấp, vì đây là vùng mục tiêu của đề tài.

## Slide 8 — Choosing the Best Augmentation

### English

On validation data, phase-only gives the highest score, with noise-only very close behind. We therefore select phase-only for the final test. The gap between these two methods is small, so the safe conclusion is not that phase rotation is always best. It is simply the method selected by our validation results in this experiment.

### Tiếng Việt

Trên validation set, phase-only cho kết quả cao nhất, còn noise-only đứng rất gần phía sau. Vì vậy nhóm chọn phase-only cho bước test cuối. Khoảng cách giữa hai phương pháp khá nhỏ, nên nhóm không kết luận rằng phase rotation luôn tốt nhất. Đây chỉ là phương pháp được validation lựa chọn trong phạm vi thí nghiệm này.

## Slide 9 — Main Result

### English

This is our main result. Phase-only improves overall accuracy and Macro-F1, but the largest gain appears in the low-SNR region: 4.72 percentage points. The curves also show that the benefit is strongest where the signals are most difficult. This supports our original idea that augmentation is especially useful when noise hides the signal structure.

### Tiếng Việt

Đây là kết quả chính của đề tài. Phase-only cải thiện cả accuracy tổng thể và Macro-F1, nhưng mức tăng lớn nhất nằm ở vùng SNR thấp: 4,72 điểm phần trăm. Đường biểu diễn cũng cho thấy lợi ích rõ nhất ở những mức tín hiệu khó. Kết quả này ủng hộ giả thuyết ban đầu rằng augmentation hữu ích nhất khi nhiễu che lấp cấu trúc tín hiệu.

## Slide 10 — What Changed Between Classes

### English

The improvement is not shared equally by every class. Some FSK classes become much easier to recognize, while a few other classes improve less or even decrease. This tells us that augmentation changes the model's decision boundaries rather than solving every confusion. The overall result is positive, but the class-level trade-offs still matter.

### Tiếng Việt

Mức cải thiện không được chia đều cho mọi lớp. Một số lớp FSK được nhận dạng tốt hơn rõ rệt, trong khi vài lớp khác tăng ít hoặc thậm chí giảm. Điều này cho thấy augmentation làm thay đổi ranh giới quyết định của mô hình chứ không giải quyết toàn bộ nhầm lẫn. Kết quả tổng thể là tích cực, nhưng trade-off giữa các lớp vẫn cần được lưu ý.

## Slide 11 — Live Demo

### English

The demo gives a visual example of how the baseline and augmented model respond to the same noisy signal. It is useful for intuition, but it is only one example. Our conclusion comes from the full test set and repeated experiments, not from a single successful demo case.

### Tiếng Việt

Phần demo giúp minh họa trực quan cách baseline và mô hình có augmentation phản ứng với cùng một tín hiệu nhiễu. Demo giúp người xem dễ hình dung hơn, nhưng đây chỉ là một ví dụ. Kết luận của nhóm dựa trên toàn bộ test set và nhiều lần thử nghiệm, không dựa trên một trường hợp demo thuận lợi.

## Slide 12 — Conclusion

### English

Within this experiment, phase augmentation produces a clear improvement at low SNR. The evaluation is paired, the selection is made on validation data, and the gain appears across repeated runs. However, this does not yet prove robustness in real radio environments. The next step is to test richer channel effects and real over-the-air data.

### Tiếng Việt

Trong phạm vi thí nghiệm này, phase augmentation tạo ra mức cải thiện rõ ràng ở SNR thấp. Phép so sánh được thiết kế paired, cấu hình được chọn trên validation và mức tăng xuất hiện qua nhiều lần train. Tuy nhiên, kết quả này chưa chứng minh độ bền trong môi trường vô tuyến thực. Bước tiếp theo là thử với kênh phức tạp hơn và dữ liệu thu over-the-air.

## Slide 13 — Appendix: Training Seeds

### English

If asked: the five seeds represent repeated training under matched conditions. They measure training variability on the same fixed dataset; they are not five different datasets.

### Tiếng Việt

Nếu được hỏi: năm seed là năm lần train lặp lại trong điều kiện tương ứng. Chúng đo độ biến thiên của quá trình train trên cùng một dataset cố định, không phải năm dataset khác nhau.

## Slide 14 — Appendix: Phase Offset and CFO

### English

If asked: our augmentation applies one fixed phase rotation to the whole burst. CFO changes phase continuously over time, so CFO robustness is outside the scope of this experiment.

### Tiếng Việt

Nếu được hỏi: augmentation của nhóm xoay toàn bộ burst bằng một góc cố định. CFO làm phase thay đổi liên tục theo thời gian, nên khả năng chống CFO nằm ngoài phạm vi thí nghiệm này.

## Slide 15 — Appendix: SNR80

### English

If asked: SNR80 gives a secondary view of performance, but it varies strongly between runs. We therefore base the main conclusion on accuracy and Macro-F1, which are more stable here.

### Tiếng Việt

Nếu được hỏi: SNR80 cung cấp thêm một góc nhìn, nhưng chỉ số này biến thiên khá mạnh giữa các lần train. Vì vậy kết luận chính dựa trên accuracy và Macro-F1, là các chỉ số ổn định hơn trong thí nghiệm này.

## Slide 16 — Appendix: Reproducibility

### English

If asked: the dataset split, training configurations, checkpoints, predictions, and final plots are all stored. This allows each reported result to be traced back to an experiment artifact.

### Tiếng Việt

Nếu được hỏi: dataset split, cấu hình train, checkpoint, prediction và các biểu đồ cuối đều được lưu lại. Nhờ đó, mỗi kết quả báo cáo đều có thể truy ngược về artifact của thí nghiệm.

## Slide 17 — Recorded Demo

### English

This is the recorded version of our demo, included as a backup in case the live application has a technical problem.

### Tiếng Việt

Đây là video quay sẵn của phần demo, được dùng làm phương án dự phòng nếu ứng dụng live gặp sự cố kỹ thuật.

## Slide 18 — Thank You

### English

Thank you for listening. We are ready for your questions.

### Tiếng Việt

Nhóm em xin cảm ơn thầy/cô và các bạn đã lắng nghe. Nhóm em xin sẵn sàng trả lời câu hỏi.
