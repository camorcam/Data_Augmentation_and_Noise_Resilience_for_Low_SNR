"""
Ứng dụng Web Demo Tương tác (Streamlit Web UI) cho EE136IU Capstone Mini-Project (Topic 4).
Tính năng:
- So sánh trực tiếp Side-by-Side giữa Baseline 1D CNN và Augmented 1D CNN (Topic 4).
- Trực quan hóa Tín hiệu Baseband I/Q (Time-series) và Hình học Chòm sao (Constellation Scatter Plot).
- Điều chỉnh Live các tham số: Modulation Type, SNR (-10 dB đến +18 dB), Phase Rotation ngẫu nhiên.
Chạy ứng dụng:
    streamlit run app.py
"""

import sys
from pathlib import Path
import json
import os
import torch  # pyrefly: ignore [missing-import]
import torch.nn.functional as F  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]
import pandas as pd  # pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import]
import streamlit as st  # pyrefly: ignore [missing-import]

# Cấu hình UTF-8 cho Windows console output
sys.stdout.reconfigure(encoding="utf-8")

# Import các module từ gói dataset và models
from dataset import (  # pyrefly: ignore [missing-import]
    normalize_rms,
    MODULATION_CLASSES,
    SNR_GRID,
)
from dataset.signal_generator import (
    _generate_modulation_burst,
    add_awgn_noise,
)  # pyrefly: ignore [missing-import]
from models import ModulationClassifier1DCNN  # pyrefly: ignore [missing-import]

# Cấu hình Trang Streamlit
st.set_page_config(
    page_title="EE136IU Capstone Demo - Topic 4",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho giao diện Clean White Academic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: inherit !important;
        border-bottom: 2px solid #0284C7;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 18px;
        font-weight: 600;
        color: #0284C7;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .correct-badge {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .wrong-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 6px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models(run_root_str, seed, aug_config):
    """Tải 2 mô hình đã huấn luyện vào bộ nhớ cache."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_root = Path(run_root_str)
    path_baseline = run_root / f"seed_{seed}" / "baseline" / "best.pth"
    path_augmented = run_root / f"seed_{seed}" / aug_config / "best.pth"

    if not os.path.exists(path_baseline) or not os.path.exists(path_augmented):
        st.error(
            "❌ Chưa tìm thấy checkpoint mô hình! Vui lòng chạy `py train.py` trước."
        )
        st.stop()

    model_base = ModulationClassifier1DCNN(num_classes=len(MODULATION_CLASSES)).to(
        device
    )
    ckpt_base = torch.load(path_baseline, map_location=device, weights_only=False)
    model_base.load_state_dict(ckpt_base["model_state_dict"])
    model_base.eval()

    model_aug = ModulationClassifier1DCNN(num_classes=len(MODULATION_CLASSES)).to(
        device
    )
    ckpt_aug = torch.load(path_augmented, map_location=device, weights_only=False)
    if ckpt_base["config"]["manifest_sha256"] != ckpt_aug["config"]["manifest_sha256"]:
        raise ValueError("Checkpoints use different datasets")
    model_aug.load_state_dict(ckpt_aug["model_state_dict"])
    model_aug.eval()

    return model_base, model_aug, device


def discover_trained_runs(run_root):
    """Liệt kê seed và cấu hình có checkpoint hợp lệ để demo."""
    available = {}
    if not run_root.exists():
        return available
    for seed_dir in run_root.glob("seed_*"):
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        if not (seed_dir / "baseline" / "best.pth").is_file():
            continue
        configs = sorted(
            child.name
            for child in seed_dir.iterdir()
            if child.is_dir()
            and child.name != "baseline"
            and (child / "best.pth").is_file()
        )
        if configs:
            available[seed] = configs
    return dict(sorted(available.items()))


def plot_signal_waveform(iq_burst):
    """Vẽ dạng sóng I/Q thời gian."""
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    ax.plot(iq_burst[0, :], label="In-phase (I)", color="#0284C7", linewidth=1.5)
    ax.plot(
        iq_burst[1, :],
        label="Quadrature (Q)",
        color="#E11D48",
        linewidth=1.5,
        linestyle="--",
    )
    ax.set_title(
        "Baseband I/Q Waveform (L = 128 samples)",
        fontsize=11,
        fontweight="bold",
        color="#0F172A",
    )
    ax.set_xlabel("Sample Index n", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def plot_constellation(iq_burst, title="Constellation Diagram"):
    """Vẽ sơ đồ chòm sao (Constellation Diagram) trên mặt phẳng I/Q."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    ax.scatter(
        iq_burst[0, :],
        iq_burst[1, :],
        color="#2563EB",
        alpha=0.75,
        edgecolors="k",
        linewidth=0.5,
        s=25,
    )
    ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="#94A3B8", linewidth=0.8, linestyle=":")
    ax.set_title(title, fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_xlabel("In-Phase (I)", fontsize=9)
    ax.set_ylabel("Quadrature (Q)", fontsize=9)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def main():
    st.markdown(
        '<div class="main-header">📡 EE136IU Capstone Mini-Project Demo — Topic 4</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Đề tài**: *Data Augmentation & Noise Resilience for Low-SNR Regime (-10 dB to +2 dB)*"
    )

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("🧠 Chọn mô hình đã huấn luyện")
    run_root = Path(os.environ.get("AMC_RUN_ROOT", "runs/ablation"))
    available = discover_trained_runs(run_root)
    if not available:
        st.error(
            f"❌ Không tìm thấy checkpoint hoàn chỉnh trong `{run_root}`. "
            "Hãy chạy ablation theo README trước."
        )
        st.stop()

    default_seed = int(os.environ.get("AMC_SEED", "42"))
    seeds = list(available)
    selected_training_seed = st.sidebar.selectbox(
        "Training seed:",
        seeds,
        index=seeds.index(default_seed) if default_seed in seeds else 0,
        help="Seed điều khiển khởi tạo trọng số, dropout và thứ tự shuffle khi train.",
    )

    config_labels = {
        "phase_only": "Phase only — xoay pha",
        "noise_only": "Noise only — thêm AWGN",
        "shift_only": "Shift only — dịch thời gian",
        "scale_only": "Scale only — scale biên độ",
        "augmented_full": "Full — kết hợp 4 transform",
    }
    configs = available[selected_training_seed]
    preferred = os.environ.get("AMC_AUG_CONFIG", "phase_only")
    selected_aug_config = st.sidebar.selectbox(
        "Cấu hình augmentation:",
        configs,
        index=configs.index(preferred) if preferred in configs else 0,
        format_func=lambda name: config_labels.get(name, name),
    )

    with st.sidebar.expander("Seed 42, 43, 44, 45, 46 nghĩa là gì?"):
        st.markdown(
            """
            **Training seed** là điểm khởi đầu cố định của bộ sinh số ngẫu nhiên.
            Mỗi số tạo ra một lần train khác nhau về trọng số ban đầu, dropout và
            thứ tự batch, nhưng giữ nguyên dataset và hyperparameter.

            Các số 42–46 chỉ là năm mã seed tùy chọn, **không phải** SNR, epoch,
            độ chính xác hay phiên bản model. Chạy nhiều seed giúp kiểm tra kết
            quả có ổn định hay chỉ may mắn ở một lần train. Kết luận chính dùng
            trung bình và độ lệch chuẩn của cả năm seed, không chọn seed đẹp nhất.
            """
        )

    model_base, model_aug, device = load_models(
        str(run_root), selected_training_seed, selected_aug_config
    )

    st.sidebar.divider()
    st.sidebar.header("🎛️ Điều khiển tín hiệu demo")

    selected_mod = st.sidebar.selectbox(
        "Loại Modulation gốc (True Class):",
        MODULATION_CLASSES,
        index=3,  # Mặc định 16QAM
    )

    selected_snr = st.sidebar.slider(
        "Tỷ số Tín hiệu / Nhiễu SNR (dB):",
        min_value=-10,
        max_value=18,
        value=-4,  # Mặc định -4 dB (Low SNR)
        step=2,
    )

    apply_phase_rotation = st.sidebar.checkbox(
        "Bật xoay pha bổ sung (constant phase offset)", value=True
    )
    phase_angle_deg = st.sidebar.slider(
        "Góc xoay pha bổ sung (Độ):", 0, 360, 45, disabled=not apply_phase_rotation
    )

    st.sidebar.caption(
        "Tín hiệu gốc đã có pha ban đầu ngẫu nhiên. Slider thêm một phase offset, không phải CFO."
    )
    random_seed = st.sidebar.number_input(
        "Seed sinh burst demo:",
        min_value=1,
        max_value=9999,
        value=42,
        help="Chỉ tái tạo burst đang hiển thị; không đổi model hoặc training seed.",
    )

    # --- GENERATE SIGNAL BURST ---
    rng = np.random.default_rng(random_seed)
    raw_complex = _generate_modulation_burst(selected_mod, num_samples=128, rng=rng)
    noisy_complex = add_awgn_noise(raw_complex, snr_db=selected_snr, rng=rng)

    if apply_phase_rotation:
        rad = np.deg2rad(phase_angle_deg)
        noisy_complex = noisy_complex * np.exp(1j * rad)

    # Tensor [2, 128] chuẩn hóa RMS
    iq_raw = np.stack([np.real(noisy_complex), np.imag(noisy_complex)], axis=0)
    iq_norm = normalize_rms(iq_raw)  # shape [2, 128]

    # Model Inference
    input_tensor = (
        torch.from_numpy(iq_norm).float().unsqueeze(0).to(device)
    )  # shape [1, 2, 128]

    with torch.no_grad():
        logits_base = model_base(input_tensor)
        probs_base = F.softmax(logits_base, dim=1).cpu().numpy()[0]
        pred_base_idx = int(np.argmax(probs_base))

        logits_aug = model_aug(input_tensor)
        probs_aug = F.softmax(logits_aug, dim=1).cpu().numpy()[0]
        pred_aug_idx = int(np.argmax(probs_aug))

    true_mod_idx = MODULATION_CLASSES.index(selected_mod)

    # --- LAYOUT TOP: SIGNAL VISUALIZATION ---
    col_vis1, col_vis2 = st.columns([1.5, 1])

    with col_vis1:
        st.subheader("📊 Trực quan hóa Dạng sóng Baseband I/Q")
        st.pyplot(plot_signal_waveform(iq_norm))

    with col_vis2:
        st.subheader("🎯 Mặt phẳng Chòm sao (Constellation)")
        st.pyplot(
            plot_constellation(iq_norm, title=f"Constellation @ SNR={selected_snr} dB")
        )

    st.markdown("---")

    # --- LAYOUT BOTTOM: SIDE-BY-SIDE PREDICTION COMPARISON ---
    st.subheader("⚡ So sánh Trực tiếp Dự đoán Side-by-Side (Model Comparison)")

    col_model_a, col_model_b = st.columns(2)

    # --- MODEL A (BASELINE) ---
    with col_model_a:
        st.markdown(
            '<div class="sub-header">🔴 Mô hình A: Baseline 1D CNN</div>',
            unsafe_allow_html=True,
        )
        st.write("*Không sử dụng Data Augmentation khi huấn luyện*")
        st.caption(f"Training seed: {selected_training_seed}")

        pred_base_name = MODULATION_CLASSES[pred_base_idx]
        conf_base = probs_base[pred_base_idx] * 100.0

        if pred_base_idx == true_mod_idx:
            st.markdown(
                f"Dự đoán: **{pred_base_name}** <span class='correct-badge'>CHÍNH XÁC ✅</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"Dự đoán: **{pred_base_name}** <span class='wrong-badge'>SAI LỖI ❌</span> (Thực tế: **{selected_mod}**)",
                unsafe_allow_html=True,
            )

        st.write(f"Độ tin cậy (Confidence): **{conf_base:.2f}%**")

        # Probability distribution bar chart
        df_base = pd.DataFrame(
            {"Modulation": MODULATION_CLASSES, "Probability": probs_base}
        )
        st.bar_chart(df_base.set_index("Modulation"), color="#DC2626")

    # --- MODEL B (AUGMENTED - TOPIC 4) ---
    with col_model_b:
        st.markdown(
            '<div class="sub-header">🔵 Mô hình B: Augmented 1D CNN (Topic 4)</div>',
            unsafe_allow_html=True,
        )
        st.write("Cấu hình: " + config_labels.get(selected_aug_config, selected_aug_config))
        st.caption(f"Training seed: {selected_training_seed}")

        pred_aug_name = MODULATION_CLASSES[pred_aug_idx]
        conf_aug = probs_aug[pred_aug_idx] * 100.0

        if pred_aug_idx == true_mod_idx:
            st.markdown(
                f"Dự đoán: **{pred_aug_name}** <span class='correct-badge'>CHÍNH XÁC ✅</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"Dự đoán: **{pred_aug_name}** <span class='wrong-badge'>SAI LỖI ❌</span> (Thực tế: **{selected_mod}**)",
                unsafe_allow_html=True,
            )

        st.write(f"Độ tin cậy (Confidence): **{conf_aug:.2f}%**")

        # Probability distribution bar chart
        df_aug = pd.DataFrame(
            {"Modulation": MODULATION_CLASSES, "Probability": probs_aug}
        )
        st.bar_chart(df_aug.set_index("Modulation"), color="#0284C7")

    # --- DEMO SUMMARY FOOTER ---
    st.info(
        "Một burst chỉ minh họa dự đoán, không xác định mô hình nào tốt hơn. "
        "So sánh tổng thể bằng benchmark nhiều seed trên test độc lập. "
        "Softmax chưa được hiệu chuẩn thành xác suất đúng."
    )


if __name__ == "__main__":
    main()
