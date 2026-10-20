import numpy as np  # pyrefly: ignore [missing-import]

def normalize_rms(iq_tensor: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Áp dụng chuẩn hóa năng lượng RMS theo từng burst:
        r_norm[n] = r[n] / E_rms
    với E_rms = sqrt( mean( I^2 + Q^2 ) ).

    Tham số:
        iq_tensor: np.ndarray dạng [2, 128] hoặc [N_batch, 2, 128]
        eps: hằng số epsilon nhỏ để tránh chia cho 0.

    Trả về:
        np.ndarray đã chuẩn hóa RMS có cùng shape với iq_tensor, RMS energy = 1.0.
    """
    if iq_tensor.ndim == 2:
        # Shape [2, 128]
        i_chan = iq_tensor[0, :]
        q_chan = iq_tensor[1, :]
        rms = np.sqrt(np.mean(i_chan**2 + q_chan**2) + eps)
        return iq_tensor / rms
    elif iq_tensor.ndim == 3:
        # Shape [N_batch, 2, 128]
        i_chan = iq_tensor[:, 0, :]
        q_chan = iq_tensor[:, 1, :]
        rms = np.sqrt(np.mean(i_chan**2 + q_chan**2, axis=1, keepdims=True) + eps) # shape [N_batch, 1]
        rms_expanded = np.expand_dims(rms, axis=1) # shape [N_batch, 1, 1]
        return iq_tensor / rms_expanded
    else:
        raise ValueError(f"Shape không hợp lệ: {iq_tensor.shape}. Kỳ vọng [2, 128] hoặc [N_batch, 2, 128]")

def format_iq_tensor(iq_data: np.ndarray) -> np.ndarray:
    """
    Đảm bảo dữ liệu I/Q đạt chuẩn [N_batch, 2, 128] kiểu float32.
    """
    iq_array = np.asarray(iq_data, dtype=np.float32)
    if iq_array.ndim == 2 and iq_array.shape == (2, 128):
        iq_array = np.expand_dims(iq_array, axis=0)
    
    assert iq_array.ndim == 3 and iq_array.shape[1:] == (2, 128), \
        f"Tensor shape phải là [N_batch, 2, 128], nhận được: {iq_array.shape}"
    return iq_array
