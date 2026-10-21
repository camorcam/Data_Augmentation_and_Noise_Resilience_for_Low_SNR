# Verification of the revised implementation

Verified locally on CPU with Python 3.12.3, PyTorch 2.14.1 and NumPy 2.5.4.

- All six automated pipeline tests pass.
- The frozen dataset contains 12,600 train, 2,880 validation and 6,000 test
  bursts. All 21,480 exact burst hashes are unique and all file checksums pass.
- Six configurations were trained for 25 epochs on each of seeds 42--46.
- Augmentation was selected using validation only. `phase_only` had the highest
  mean low-SNR validation accuracy (50.73%).
- The frozen baseline/phase-only pair was then evaluated on test once and all
  predictions, raw counts, normalized confusion matrices and summaries were
  saved in `results/final_phase/`.
- Twenty-two figures were generated: two SNR curves, ten loss plots and ten
  confusion-matrix plots.

Test means across five training seeds:

| Metric | Baseline | Phase only | Paired difference |
|---|---:|---:|---:|
| Accuracy | 66.60% | 69.56% | +2.96 percentage points |
| Macro-F1 | 65.75% | 68.96% | +3.20 percentage points |
| Mean accuracy, -10 to +2 dB | 46.44% | 51.16% | +4.72 percentage points |
| SNR80 | -- | -- | +1.37 dB mean gain |

The values after `+/-` in the JSON are sample SD across training seeds on one
shared synthetic test set, not confidence intervals and not uncertainty across
independently generated datasets. SNR80 gain is especially variable (sample SD
2.18 dB; one of five seeds is negative), so it must not be described as a
guaranteed sensitivity improvement.
