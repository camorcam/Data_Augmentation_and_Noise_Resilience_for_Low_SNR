# Low-SNR Automatic Modulation Classification

EE136IU project evaluating RF-inspired data augmentation for low-SNR Automatic
Modulation Classification. The final experiment compares a baseline 1D CNN
with validation-selected `phase_only` augmentation across five paired seeds.

## Structure

```text
.
├── dataset/                 # Signal generation and frozen dataset I/O
├── models/                  # 1D CNN and augmentation transforms
├── data/frozen_v2/          # Frozen train/validation/test data
├── runs/ablation/           # Checkpoints and training histories
├── results/final_phase/     # Final metrics, predictions and plots
├── tests/                   # Pipeline tests
├── train.py                 # Train experiment configurations
├── summarize_validation.py  # Select configuration on validation
├── evaluate.py              # Evaluate the frozen test set
├── plot_results.py          # Generate result figures
├── app.py                   # Streamlit demo
├── PRESENTATION_12M.md      # Presentation key points
└── PROJECT_OVERVIEW.md      # Detailed methodology and results
```

## Setup

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For CUDA, install the appropriate PyTorch build for your system.

## Run the existing demo

The frozen data, trained checkpoints and final results are already included.

```powershell
python -m streamlit run app.py
```

Defaults: `runs/ablation`, seed `42`, configuration `phase_only`.

## Reproduce the experiment

Run these steps in order. Skip step 1 when reusing `data/frozen_v2/`.

### 1. Generate data

```powershell
python -m dataset.storage --out data/frozen_v2
```

### 2. Train all candidates

```powershell
python train.py `
  --data data/frozen_v2 `
  --out runs/ablation `
  --configs baseline phase_only noise_only shift_only scale_only augmented_full `
  --seeds 42 43 44 45 46
```

### 3. Compare validation results

```powershell
python summarize_validation.py --runs runs/ablation --seeds 42 43 44 45 46
```

The completed experiment selects `phase_only`. Configuration selection must not
use test results.

### 4. Evaluate the selected pair

```powershell
python evaluate.py `
  --data data/frozen_v2 `
  --runs runs/ablation `
  --augmented-config phase_only `
  --out results/final_phase `
  --seeds 42 43 44 45 46
```

### 5. Generate plots

```powershell
python plot_results.py --results results/final_phase --runs runs/ablation
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

Quick CPU smoke run:

```powershell
python -m dataset.storage --out data/smoke --train-per-cell 2 --val-per-cell 2 --test-per-cell 2
python train.py --data data/smoke --out runs/smoke --seeds 42 43 --epochs 1 --device cpu
python evaluate.py --data data/smoke --runs runs/smoke --out results/smoke --seeds 42 43 --device cpu
python plot_results.py --runs runs/smoke --results results/smoke
```

The smoke run checks integration only; its metrics are not research results.

## Outputs

- `runs/ablation/seed_*/<config>/`: checkpoints and training/validation history
- `results/final_phase/benchmark_results.json`: final metrics
- `results/final_phase/predictions.npz`: test predictions
- `results/final_phase/plots/`: accuracy, Macro-F1, loss and confusion plots

Results apply to this frozen synthetic benchmark. Constant phase rotation is
not CFO, and real-world RF generalization is not established.
