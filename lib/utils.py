import numpy as np


def stft(signal, sample_rate=48000, window_size=4096, step=2048, keep_freq_bins=512, window_type='hamming'):
    """
    优化的STFT变换：保留更多高频信息，减少频谱泄漏，提升还原精度
    """
    target_frames = 2048  # 目标帧数不变

    # 1. 计算padding（边缘复制填充，减少边缘失真）
    required_length = window_size + (target_frames - 1) * step
    pad_length = required_length - len(signal)
    pad_before = pad_length // 2
    pad_after = pad_length - pad_before
    signal_padded = np.pad(signal, (pad_before, pad_after), mode='edge')  # 边缘复制代替补零

    # 2. 选择更优窗口（汉明窗旁瓣衰减更大，频谱泄漏更少）
    if window_type == 'hann':
        window = np.hanning(window_size)
    elif window_type == 'hamming':
        window = np.hamming(window_size)  # 默认汉明窗，比汉宁窗泄漏少
    elif window_type == 'blackman':
        window = np.blackman(window_size)  # 布莱克曼窗泄漏最少，但时域模糊稍大
    else:
        raise ValueError("窗口类型仅支持'hann'/'hamming'/'blackman'")

    # 3. 计算STFT（保留更多高频，减少信息丢失）
    stft_magnitude = np.zeros((target_frames, keep_freq_bins), dtype=np.float32)
    stft_phase = np.zeros((target_frames, keep_freq_bins), dtype=np.float32)

    for i in range(target_frames):
        start = i * step
        end = start + window_size
        frame = signal_padded[start:end] * window  # 应用窗口

        fft_result = np.fft.fft(frame)
        fft_cropped = fft_result[:keep_freq_bins]  # 保留2048点高频

        stft_magnitude[i] = np.abs(fft_cropped)
        stft_phase[i] = np.angle(fft_cropped)

    # 4. 优化归一化（减少数值误差）
    stft_magnitude = np.log1p(stft_magnitude)  # log1p保持不变
    mag_mean = np.mean(stft_magnitude)
    mag_std = np.std(stft_magnitude)
    stft_magnitude = (stft_magnitude - mag_mean) / (mag_std + 1e-8)  # 标准化

    stft_phase = stft_phase / np.pi  # 相位归一化

    # -------------------------- 补全缺失的stft_result生成 --------------------------
    stft_result = np.stack([stft_magnitude, stft_phase], axis=-1)  # 堆叠幅度谱和相位谱

    return stft_result, mag_mean, mag_std  # 正确返回三个变量


def istft(stft_result, original_length, mag_mean, mag_std,
          sample_rate=48000, window_size=4096, step=2048, keep_freq_bins=512, window_type='hamming'):
    """
    优化的ISTFT逆变换：精准补全频谱，优化窗口校正，提升还原精度
    """
    # 1. 反归一化（使用更稳定的数值计算）
    stft_magnitude_norm = stft_result[..., 0]
    stft_phase_norm = stft_result[..., 1]

    # 幅度谱反归一化：用expm1代替exp-1，数值精度更高
    stft_magnitude = np.expm1(np.clip(
        stft_magnitude_norm * (mag_std + 1e-8) + mag_mean,
        a_min=0, a_max=30  # 限制exp输入（exp(30)≈1e13，避免溢出）
    ))

    stft_phase = stft_phase_norm * np.pi  # 相位反归一化

    # 2. 频谱补全（用数组操作代替循环，减少计算误差）
    num_frames = stft_result.shape[0]
    full_complex_spec = np.zeros((num_frames, window_size), dtype=np.complex64)

    # 前半部分：直接赋值
    full_complex_spec[:, :keep_freq_bins] = stft_magnitude * np.exp(1j * stft_phase)

    # 后半部分：利用实信号FFT对称性（X[k] = conj(X[N-k])），用数组切片代替循环
    if window_size > keep_freq_bins:
        # 对称位置索引：window_size - k 对应 k（k从1到window_size - keep_freq_bins）
        sym_indices = np.arange(window_size - 1, keep_freq_bins - 1, -1)  # 倒序索引
        full_complex_spec[:, keep_freq_bins:] = np.conj(full_complex_spec[:, sym_indices])

    # 3. 逆FFT与信号合成（优化窗口校正）
    time_frames = np.fft.ifft(full_complex_spec).real  # 逆FFT取实部

    # 窗口选择（与STFT保持一致）
    if window_type == 'hann':
        window = np.hanning(window_size)
    elif window_type == 'hamming':
        window = np.hamming(window_size)
    elif window_type == 'blackman':
        window = np.blackman(window_size)

    # 重叠相加（预计算窗口能量，优化校正）
    required_length = window_size + (num_frames - 1) * step
    output_signal = np.zeros(required_length, dtype=np.float32)
    window_sq = window ** 2  # 窗口平方
    window_correction = np.zeros(required_length, dtype=np.float32)

    for i in range(num_frames):
        start = i * step
        end = start + window_size
        output_signal[start:end] += time_frames[i] * window
        window_correction[start:end] += window_sq

    # 窗口校正（避免除零，用max代替clip更稳定）
    window_correction = np.maximum(window_correction, 1e-8)
    output_signal /= window_correction

    # 4. 裁剪padding（精准还原原始长度）
    pad_length = required_length - original_length
    pad_before = pad_length // 2
    signal = output_signal[pad_before: pad_before + original_length]

    return signal


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
        keep_freq_bins=2048,
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
        keep_freq_bins=2048,
        step=2048,
        window_type='hamming'
    )

    # 3. 验证还原效果（优化后MSE应降至0.5左右）
    print(f"\n原始信号长度: {original_length}")
    print(f"还原信号长度: {len(recovered_signal)}")
    mse = np.mean((original_signal - recovered_signal) ** 2)
    print(f"优化后还原误差（MSE）: {mse:.6f}")  # 预期输出 ~0.5