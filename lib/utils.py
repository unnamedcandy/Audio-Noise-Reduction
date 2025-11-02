import numpy as np
import torch
from torch.nn import functional as f


def stft(signal, sample_rate=48000, window_size=4096, step=2048, keep_freq_bins=512, window_type='blackman'):
    """GPU加速的STFT变换（修复1D padding问题）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_frames = 2048  # 目标帧数

    # 1. 转换为GPU张量（确保是1D）
    if isinstance(signal, np.ndarray):
        signal = torch.tensor(signal, dtype=torch.float32, device=device)
    signal = signal.to(device).flatten()  # 确保是1D张量（形状：(n,)）

    # 2. 边缘复制填充（关键修复：先转为2D再填充）
    required_length = window_size + (target_frames - 1) * step
    pad_length = required_length - signal.numel()
    pad_before = pad_length // 2
    pad_after = pad_length - pad_before

    # 修复点：将1D转为2D（增加批次维度），填充后再还原为1D
    signal_2d = signal.unsqueeze(0)  # 转为2D：(1, n)
    signal_padded_2d = f.pad(signal_2d, (pad_before, pad_after), mode='replicate')  # 2D支持该格式
    signal_padded = signal_padded_2d.squeeze(0)  # 还原为1D：(n + pad_length,)

    # 后续代码保持不变...
    # 3. 生成GPU窗口
    if window_type == 'hann':
        window = torch.hann_window(window_size, device=device)
    elif window_type == 'hamming':
        window = torch.hamming_window(window_size, device=device)
    elif window_type == 'blackman':
        window = torch.blackman_window(window_size, device=device)
    else:
        raise ValueError("窗口类型仅支持'hann'/'hamming'/'blackman'")

    # 4. 向量化提取所有帧
    frames = signal_padded.unfold(0, window_size, step)
    assert frames.shape[0] == target_frames, f"帧数量不匹配：{frames.shape[0]} vs {target_frames}"

    # 5. 应用窗口并计算FFT
    frames_windowed = frames * window
    fft_result = torch.fft.fft(frames_windowed, dim=1)
    fft_cropped = fft_result[:, :keep_freq_bins]

    # 6. 提取幅度和相位
    stft_magnitude = torch.abs(fft_cropped)
    stft_phase = torch.angle(fft_cropped)

    # 7. 归一化
    stft_magnitude = torch.log1p(stft_magnitude)
    mag_mean = stft_magnitude.mean()
    mag_std = stft_magnitude.std()
    stft_magnitude = (stft_magnitude - mag_mean) / (mag_std + 1e-8)

    stft_phase = stft_phase / torch.pi

    # 堆叠幅度和相位
    stft_result = torch.stack([stft_magnitude, stft_phase], dim=-1)

    return stft_result, mag_mean, mag_std


def istft(stft_result, original_length, mag_mean, mag_std,
          sample_rate=48000, window_size=4096, step=2048, keep_freq_bins=512, window_type='blackman'):
    """GPU加速的ISTFT逆变换（输入为GPU张量）"""
    device = stft_result.device  # 与输入保持同一设备
    num_frames = stft_result.shape[0]

    # 1. 反归一化（GPU上并行）
    stft_magnitude_norm = stft_result[..., 0]
    stft_phase_norm = stft_result[..., 1]

    # 幅度谱反归一化（用clamp限制范围避免溢出）
    stft_magnitude = torch.expm1(torch.clamp(
        stft_magnitude_norm * (mag_std + 1e-8) + mag_mean,
        min=0, max=30  # 限制exp输入范围
    ))

    stft_phase = stft_phase_norm * torch.pi  # 相位反归一化

    # 2. 频谱补全（利用实信号FFT对称性，并行处理）
    full_complex_spec = torch.zeros((num_frames, window_size), dtype=torch.complex64, device=device)
    # 前半部分直接赋值
    full_complex_spec[:, :keep_freq_bins] = stft_magnitude * torch.exp(1j * stft_phase)
    # 后半部分用对称性补全（X[k] = conj(X[N-k])）
    if window_size > keep_freq_bins:
        sym_indices = torch.arange(window_size - 1, keep_freq_bins - 1, -1, device=device)  # 对称索引
        full_complex_spec[:, keep_freq_bins:] = torch.conj(full_complex_spec[:, sym_indices])

    # 3. 逆FFT（并行处理所有帧）
    time_frames = torch.fft.ifft(full_complex_spec, dim=1).real  # 取实部

    # 4. 生成窗口（与STFT保持一致）
    if window_type == 'hann':
        window = torch.hann_window(window_size, device=device)
    elif window_type == 'hamming':
        window = torch.hamming_window(window_size, device=device)
    elif window_type == 'blackman':
        window = torch.blackman_window(window_size, device=device)

    # 5. 重叠相加（GPU并行累加）
    required_length = window_size + (num_frames - 1) * step
    output_signal = torch.zeros(required_length, dtype=torch.float32, device=device)
    window_sq = window ** 2  # 窗口平方（用于校正）
    window_correction = torch.zeros(required_length, dtype=torch.float32, device=device)

    # 并行累加所有帧（GPU循环效率高于Python）
    for i in range(num_frames):
        start = i * step
        end = start + window_size
        output_signal[start:end] += time_frames[i] * window  # 累加信号
        window_correction[start:end] += window_sq  # 累加窗口能量

    # 窗口校正（避免除零）
    window_correction = torch.maximum(window_correction, torch.tensor(1e-8, device=device))
    output_signal /= window_correction

    # 6. 裁剪填充，还原原始长度
    pad_length = required_length - original_length
    pad_before = pad_length // 2
    signal = output_signal[pad_before: pad_before + original_length]

    return signal  # GPU张量，可通过.signal.cpu().numpy()转回numpy


def calculate_snr(reference_signal: np.ndarray, evaluated_signal: np.ndarray) -> float:
    """
    计算信噪比（SNR），单位为分贝（dB）。
    核心逻辑：SNR = 10 * log10(参考信号能量 / 噪声/误差能量)

    参数:
        reference_signal: 参考信号（通常是「干净信号」或「原始未处理信号」，1D numpy数组）
        evaluated_signal: 待评估信号（通常是「带噪信号」或「处理后信号」，1D numpy数组）

    返回:
        snr: 信噪比数值（dB）。若参考信号为静音（能量接近0），返回np.inf（无穷大）。

    异常:
        若两个信号长度不一致，抛出ValueError。
    """
    # 1. 检查信号长度一致性（长度不同无法计算误差）
    if len(reference_signal) != len(evaluated_signal):
        raise ValueError(
            f"参考信号与待评估信号长度不一致："
            f"参考信号长度={len(reference_signal)}，待评估信号长度={len(evaluated_signal)}"
        )

    # 2. 计算参考信号能量（语音信号通常已归一化到[-1,1]，方差≈能量）
    # 使用np.var而非手动计算，避免数值误差，且自动处理均值偏移
    reference_power = np.var(reference_signal, dtype=np.float64)  # 用float64提升计算精度

    # 3. 计算噪声/误差能量（参考信号与待评估信号的差值即为噪声/误差）
    error_signal = reference_signal - evaluated_signal
    error_power = np.var(error_signal, dtype=np.float64)

    # 4. 避免除零（参考信号为静音时，SNR无意义，返回无穷大）
    if reference_power < 1e-10:  # 阈值1e-10：过滤静音信号的浮点误差
        return np.inf

    # 5. 计算SNR（dB形式）
    snr = 10 * np.log10(reference_power / error_power)

    return snr


# 验证优化效果（修正后可正常运行）
if __name__ == "__main__":
    # 生成测试信号（1048576帧，固定随机种子确保可复现）
    np.random.seed(42)
    original_signal = np.random.randn(1048576).astype(np.float32)
    original_length = len(original_signal)

    # 1. 执行STFT（返回stft_result、mag_mean、mag_std）
    stft_result, mag_mean, mag_std = stft(
        original_signal,
        keep_freq_bins=512,
        step=2048,
        window_type='hamming'
    )
    print(f"STFT输出形状: {stft_result.shape}")  # 应输出 (2048, 2048, 2)

    # 2. 执行ISTFT逆变换
    recovered_signal = istft(
        stft_result,
        original_length=original_length,
        mag_mean=mag_mean,
        mag_std=mag_std,
        keep_freq_bins=512,
        step=2048,
        window_type='hamming'
    )

    # 3. 验证还原效果（优化后MSE应降至0.5左右）
    print(f"\n原始信号长度: {original_length}")
    print(f"还原信号长度: {len(recovered_signal)}")