import numpy as np  # pyrefly: ignore [missing-import]
from typing import Tuple, List, Dict
from .preprocessing import normalize_rms

# Danh sách 8 lớp Modulation chuẩn của khóa học EE136IU
MODULATION_CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "CPFSK", "GFSK"]

# Lưới 15 điểm SNR (-10 dB đến +18 dB, bước 2 dB)
SNR_GRID = list(range(-10, 20, 2))  # [-10, -8, -6, ..., +16, +18]


def _generate_modulation_burst(
    mod_type: str, num_samples: int = 128, rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate simplified synthetic baseband. PSK/QAM/PAM use rectangular pulses; FSK has continuous phase.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(num_samples)

    if mod_type == "BPSK":
        symbols = rng.choice([-1.0, 1.0], size=num_samples // 8)
        symbols_upsampled = np.repeat(symbols, 8)
        complex_signal = symbols_upsampled + 0.0j

    elif mod_type == "QPSK":
        constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        symbols = rng.choice(constellation, size=num_samples // 8)
        complex_signal = np.repeat(symbols, 8)

    elif mod_type == "8PSK":
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        constellation = np.exp(1j * phases)
        symbols = rng.choice(constellation, size=num_samples // 8)
        complex_signal = np.repeat(symbols, 8)

    elif mod_type == "16QAM":
        grid = np.array([-3, -1, 1, 3])
        i_sym = rng.choice(grid, size=num_samples // 8)
        q_sym = rng.choice(grid, size=num_samples // 8)
        constellation = (i_sym + 1j * q_sym) / np.sqrt(10)
        complex_signal = np.repeat(constellation, 8)

    elif mod_type == "64QAM":
        grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        i_sym = rng.choice(grid, size=num_samples // 8)
        q_sym = rng.choice(grid, size=num_samples // 8)
        constellation = (i_sym + 1j * q_sym) / np.sqrt(42)
        complex_signal = np.repeat(constellation, 8)

    elif mod_type == "PAM4":
        levels = np.array([-3, -1, 1, 3]) / np.sqrt(5)
        symbols = rng.choice(levels, size=num_samples // 8)
        complex_signal = np.repeat(symbols, 8) + 0.0j

    elif mod_type == "CPFSK":
        bits = rng.choice([-1, 1], size=num_samples // 4)
        h = 0.5  # Modulation index
        phase = np.cumsum(np.repeat(bits, 4)) * (np.pi * h / 4)
        complex_signal = np.exp(1j * phase)

    elif mod_type == "GFSK":
        bits = rng.choice([-1, 1], size=num_samples // 4)
        # Gaussian frequency pulse, BT=0.3, span=4 symbols, 4 samples/symbol.
        bt, sps, span = 0.3, 4, 4
        tau = np.arange(-span * sps, span * sps + 1) / sps
        kernel = np.exp(-2 * (np.pi * bt * tau) ** 2 / np.log(2))
        kernel /= kernel.sum()
        freq = np.repeat(bits, sps)
        padded = np.pad(freq, (len(kernel) // 2,), mode="edge")
        smooth_bits = np.convolve(padded, kernel, mode="valid")
        phase = np.cumsum(smooth_bits) * (np.pi * 0.5 / 4)
        complex_signal = np.exp(1j * phase)

    else:
        raise ValueError(f"Không hỗ trợ loại modulation: {mod_type}")

    # Xoay pha ngẫu nhiên ban đầu để mô phỏng pha kênh truyền tự nhiên
    random_initial_phase = rng.uniform(0, 2 * np.pi)
    complex_signal = complex_signal * np.exp(1j * random_initial_phase)

    return complex_signal


def add_awgn_noise(
    complex_signal: np.ndarray, snr_db: float, rng: np.random.Generator = None
) -> np.ndarray:
    """
    Thêm nhiễu AWGN theo đúng chỉ số SNR (dB) chỉ định.
    """
    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(np.abs(complex_signal) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / (snr_linear + 1e-12)

    # Nhiễu phức Gaussian (mỗi kênh I và Q chịu noise_power / 2)
    noise_i = rng.normal(0, np.sqrt(noise_power / 2.0), size=complex_signal.shape)
    noise_q = rng.normal(0, np.sqrt(noise_power / 2.0), size=complex_signal.shape)
    noise = noise_i + 1j * noise_q

    return complex_signal + noise


def generate_synthetic_dataset(
    samples_per_class_per_snr: int = 100,
    snr_grid: List[int] = SNR_GRID,
    mod_classes: List[str] = MODULATION_CLASSES,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Sinh tập dữ liệu I/Q đạt chuẩn EE136IU:
    - Shape [N_total, 2, 128]
    - Đã RMS normalize
    - Nhãn class và SNR tương ứng
    """
    rng = np.random.default_rng(seed)

    iq_list = []
    labels_list = []
    snr_list = []

    for mod_idx, mod_name in enumerate(mod_classes):
        for snr in snr_grid:
            for _ in range(samples_per_class_per_snr):
                # 1. Sinh tín hiệu baseband
                clean_signal = _generate_modulation_burst(
                    mod_name, num_samples=128, rng=rng
                )
                # 2. Thêm nhiễu AWGN ở mức SNR chỉ định
                noisy_signal = add_awgn_noise(clean_signal, snr_db=snr, rng=rng)

                # 3. Chuyển thành tensor [2, 128]
                i_channel = np.real(noisy_signal)
                q_channel = np.imag(noisy_signal)
                raw_burst = np.stack([i_channel, q_channel], axis=0)  # shape [2, 128]

                # 4. Chuẩn hóa RMS
                norm_burst = normalize_rms(raw_burst)

                iq_list.append(norm_burst)
                labels_list.append(mod_idx)
                snr_list.append(snr)

    x_data = np.array(iq_list, dtype=np.float32)  # Shape [N_total, 2, 128]
    y_data = np.array(labels_list, dtype=np.int64)  # Shape [N_total]
    snr_data = np.array(snr_list, dtype=np.float32)  # Shape [N_total]

    return {
        "x": x_data,
        "y": y_data,
        "snr": snr_data,
        "classes": mod_classes,
        "snr_grid": snr_grid,
    }
