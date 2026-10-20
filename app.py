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
import os
import torch  # pyrefly: ignore [missing-import]
import torch.nn.functional as F  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]
import pandas as pd  # pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import]
import streamlit as st  # pyrefly: ignore [missing-import]

# Cấu hình UTF-8 cho Windows console output
sys.stdout.reconfigure(encoding='utf-8')

# Import các module từ gói dataset và models
from dataset import (  # pyrefly: ignore [missing-import]
    normalize_rms,
    MODULATION_CLASSES,
    SNR_GRID
)
from dataset.signal_generator import _generate_modulation_burst, add_awgn_noise  # pyrefly: ignore [missing-import]
from models import ModulationClassifier1DCNN  # pyrefly: ignore [missing-import]

# Cấu hình Trang Streamlit
st.set_page_config(
    page_title="EE136IU Capstone Demo - Topic 4",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện Clean White Academic
st.markdown("""
    <style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
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
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Tải 2 mô hình đã huấn luyện vào bộ nhớ cache."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    path_baseline = 'checkpoint/baseline_1d_cnn.pth'
    path_augmented = 'checkpoint/augmented_1d_cnn.pth'

    if not os.path.exists(path_baseline) or not os.path.exists(path_augmented):
        st.error("❌ Chưa tìm thấy checkpoint mô hình! Vui lòng chạy `py train.py` trước.")
        st.stop()

    model_base = ModulationClassifier1DCNN(num_classes=len(MODULATION_CLASSES)).to(device)
    ckpt_base = torch.load(path_baseline, map_location=device)
    model_base.load_state_dict(ckpt_base['model_state_dict'])
    model_base.eval()

    model_aug = ModulationClassifier1DCNN(num_classes=len(MODULATION_CLASSES)).to(device)
    ckpt_aug = torch.load(path_augmented, map_location=device)
    model_aug.load_state_dict(ckpt_aug['model_state_dict'])
    model_aug.eval()

    return model_base, model_aug, device

def plot_signal_waveform(iq_burst):
    """Vẽ dạng sóng I/Q thời gian."""
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    ax.plot(iq_burst[0, :], label='In-phase (I)', color='#0284C7', linewidth=1.5)
    ax.plot(iq_burst[1, :], label='Quadrature (Q)', color='#E11D48', linewidth=1.5, linestyle='--')
    ax.set_title("Baseband I/Q Waveform (L = 128 samples)", fontsize=11, fontweight='bold', color='#0F172A')
    ax.set_xlabel("Sample Index n", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    return fig

def plot_constellation(iq_burst, title="Constellation Diagram"):
    """Vẽ sơ đồ chòm sao (Constellation Diagram) trên mặt phẳng I/Q."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    ax.scatter(iq_burst[0, :], iq_burst[1, :], color='#2563EB', alpha=0.75, edgecolors='k', linewidth=0.5, s=25)
    ax.axhline(0, color='#94A3B8', linewidth=0.8, linestyle=':')
    ax.axvline(0, color='#94A3B8', linewidth=0.8, linestyle=':')
    ax.set_title(title, fontsize=11, fontweight='bold', color='#0F172A')
    ax.set_xlabel("In-Phase (I)", fontsize=9)
    ax.set_ylabel("Quadrature (Q)", fontsize=9)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig

def main():
    st.markdown('<div class="main-header">📡 EE136IU Capstone Mini-Project Demo — Topic 4</div>', unsafe_allow_html=True)
    st.markdown("**Đề tài**: *Data Augmentation & Noise Resilience for Low-SNR Regime (-10 dB to +2 dB)*")

    model_base, model_aug, device = load_models()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("🎛️ Bảng Điều khiển Tín hiệu (Live Controls)")
    
    selected_mod = st.sidebar.selectbox(
        "Loại Modulation gốc (True Class):",
        MODULATION_CLASSES,
        index=3 # Mặc định 16QAM
    )

    selected_snr = st.sidebar.slider(
        "Tỷ số Tín hiệu / Nhiễu SNR (dB):",
        min_value=-10,
        max_value=18,
        value=-4, # Mặc định -4 dB (Low SNR)
        step=2
    )

    apply_phase_rotation = st.sidebar.checkbox("Bật Xoay pha ngẫu nhiên (Random Phase Rotation)", value=True)
    phase_angle_deg = st.sidebar.slider("Góc xoay pha bổ sung (Độ):", 0, 360, 45, disabled=not apply_phase_rotation)

    random_seed = st.sidebar.number_input("Random Seed tạo tín hiệu:", min_value=1, max_value=9999, value=42)

    # --- GENERATE SIGNAL BURST ---
    rng = np.random.default_rng(random_seed)
    raw_complex = _generate_modulation_burst(selected_mod, num_samples=128, rng=rng)
    noisy_complex = add_awgn_noise(raw_complex, snr_db=selected_snr, rng=rng)

    if apply_phase_rotation:
        rad = np.deg2rad(phase_angle_deg)
        noisy_complex = noisy_complex * np.exp(1j * rad)

    # Tensor [2, 128] chuẩn hóa RMS
    iq_raw = np.stack([np.real(noisy_complex), np.imag(noisy_complex)], axis=0)
    iq_norm = normalize_rms(iq_raw) # shape [2, 128]

    # Model Inference
    input_tensor = torch.from_numpy(iq_norm).float().unsqueeze(0).to(device) # shape [1, 2, 128]

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
        st.pyplot(plot_constellation(iq_norm, title=f"Constellation @ SNR={selected_snr} dB"))

    st.markdown("---")

    # --- LAYOUT BOTTOM: SIDE-BY-SIDE PREDICTION COMPARISON ---
    st.subheader("⚡ So sánh Trực tiếp Dự đoán Side-by-Side (Model Comparison)")

    col_model_a, col_model_b = st.columns(2)

    # --- MODEL A (BASELINE) ---
    with col_model_a:
        st.markdown('<div class="sub-header">🔴 Mô hình A: Baseline 1D CNN</div>', unsafe_allow_html=True)
        st.write("*Không sử dụng Data Augmentation khi huấn luyện*")

        pred_base_name = MODULATION_CLASSES[pred_base_idx]
        conf_base = probs_base[pred_base_idx] * 100.0

        if pred_base_idx == true_mod_idx:
            st.markdown(f"Dự đoán: **{pred_base_name}** <span class='correct-badge'>CHÍNH XÁC ✅</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"Dự đoán: **{pred_base_name}** <span class='wrong-badge'>SAI LỖI ❌</span> (Thực tế: **{selected_mod}**)", unsafe_allow_html=True)
        
        st.write(f"Độ tin cậy (Confidence): **{conf_base:.2f}%**")

        # Probability distribution bar chart
        df_base = pd.DataFrame({'Modulation': MODULATION_CLASSES, 'Probability': probs_base})
        st.bar_chart(df_base.set_index('Modulation'), color='#DC2626')

    # --- MODEL B (AUGMENTED - TOPIC 4) ---
    with col_model_b:
        st.markdown('<div class="sub-header">🔵 Mô hình B: Augmented 1D CNN (Topic 4)</div>', unsafe_allow_html=True)
        st.write("*Tích hợp Physical RF Data Augmentation (Phase, AWGN, Jitter, Scaling)*")

        pred_aug_name = MODULATION_CLASSES[pred_aug_idx]
        conf_aug = probs_aug[pred_aug_idx] * 100.0

        if pred_aug_idx == true_mod_idx:
            st.markdown(f"Dự đoán: **{pred_aug_name}** <span class='correct-badge'>CHÍNH XÁC ✅</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"Dự đoán: **{pred_aug_name}** <span class='wrong-badge'>SAI LỖI ❌</span> (Thực tế: **{selected_mod}**)", unsafe_allow_html=True)

        st.write(f"Độ tin cậy (Confidence): **{conf_aug:.2f}%**")

        # Probability distribution bar chart
        df_aug = pd.DataFrame({'Modulation': MODULATION_CLASSES, 'Probability': probs_aug})
        st.bar_chart(df_aug.set_index('Modulation'), color='#0284C7')

    # --- DEMO SUMMARY FOOTER ---
    st.info("💡 **Kết luận Trình diễn**: Ở các vùng nhiễu thấp (SNR = -4 dB) hoặc khi bị xoay pha ngẫu nhiên, mô hình Baseline bị lệch chòm sao và dự đoán sai. Ngược lại, **mô hình Topic 4 (Augmented)** nhờ học được tính bất biến pha vẫn đưa ra kết quả dự đoán đúng với độ tin cậy vượt trội!")

if __name__ == '__main__':
    main()
