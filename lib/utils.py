import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class PhasenSTFT(nn.Module):
    """
    用于PHASEN模型的STFT预处理工具类，支持GPU加速，确保STFT返回单边谱、ISTFT接收单边谱输入，且预处理不参与训练。
    保存STFT参数（窗长、步长、窗函数等），确保前后处理一致性。
    """

    def __init__(self,
                 n_fft: int = 4096,
                 hop_length: int = 1024,
                 win_length: int = 4096,
                 window_type: str = 'hamming'):
        """
        初始化STFT参数，创建窗函数并注册为缓冲区（不参与训练，支持设备同步）。

        Args:
            n_fft: FFT点数，PHASEN常用4096
            hop_length: 帧移（步长），PHASEN常用1024
            win_length: 窗长，通常与n_fft一致
            window_type: 窗函数类型，支持'hann'或'hamming'
        """
        super().__init__()
        # 保存STFT核心参数（ISTFT复用）
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_freq = n_fft // 2 + 1  # 单边谱频点数量（固定值，便于形状管理）

        # 创建窗函数（用临时变量存储，通过register_buffer创建self.window）
        if window_type == 'hann':
            window = torch.hann_window(win_length)
        elif window_type == 'hamming':
            window = torch.hamming_window(win_length)
        else:
            raise ValueError(f"不支持的窗函数类型: {window_type}，可选'hann'或'hamming'")
        self.register_buffer('window', window)

    def stft_preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        对音频波形进行STFT变换，输出**单边谱**（实部+虚部），预处理步骤无梯度计算。

        Args:
            waveform: 输入音频波形，形状为 (batch_size, time) 或 (time,)（单条音频），实数值Tensor

        Returns:
            stft_spec: STFT单边谱，形状为 (batch_size, n_freq, n_frame, 2)，其中：
                - n_freq = n_fft//2 + 1（单边谱频点数量）
                - n_frame为时间帧数量
                - 最后一维[0]为实部，[1]为虚部（float32类型）
        """
        # 确保输入维度为(batch_size, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # 单条音频增加batch维度

        # 确保输入与窗函数在同一设备
        waveform = waveform.to(self.window.device)

        # 预处理不参与训练：禁用梯度计算
        with torch.no_grad():
            # 1. 计算STFT，返回复数张量（return_complex=True，适应新版本PyTorch）
            stft_complex = torch.stft(
                input=waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                pad_mode='reflect',
                normalized=False,
                return_complex=True
            )
            # 2. 将复数张量转换为实部+虚部分离的格式（形状不变：[batch, n_freq, n_frame, 2]）
            stft_spec = torch.view_as_real(stft_complex)

        return stft_spec

    def istft_postprocess(self, stft_spec: torch.Tensor) -> torch.Tensor:
        """
        对PHASEN模型输出的**单边谱**进行ISTFT变换，还原为音频波形，无梯度计算。

        Args:
            stft_spec: 单边谱（需与STFT输出格式一致），形状为 (batch_size, n_freq, n_frame, 2)，
                最后一维[0]为实部，[1]为虚部（float32类型）

        Returns:
            waveform: 还原的音频波形，形状为 (batch_size, time)，实数值Tensor
        """
        # 确保输入与窗函数在同一设备
        stft_spec = stft_spec.to(self.window.device)

        # 后处理不参与训练：禁用梯度计算
        with torch.no_grad():
            # 1. 将实部+虚部分离的张量转换回复数张量（与stft输出对应）
            stft_complex = torch.view_as_complex(stft_spec)
            # 2. 计算ISTFT（输入为复数张量，适应新版本PyTorch）
            waveform = torch.istft(
                input=stft_complex,  # 关键：输入复数张量
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                normalized=False,
                return_complex=False  # 返回实数值波形
            )

        return waveform


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
    # 生成测试信号（1048576帧随机噪声，固定种子确保可复现）
    np.random.seed(42)
    original_signal = np.random.randn(1048576).astype(np.float32)  # 约65秒（16kHz采样率下）
    original_length = len(original_signal)
    print(f"原始信号长度: {original_length} 采样点")

    # 转换为PyTorch张量（初始形状：(time,)）
    waveform = torch.from_numpy(original_signal)

    # 初始化STFT处理器（使用PHASEN常用参数）
    stft_processor = PhasenSTFT(
        n_fft=4096,
        hop_length=1024,
        win_length=4096,
        window_type='hamming'
    )

    # 自动选择设备（GPU优先）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    stft_processor = stft_processor.to(device)
    waveform = waveform.to(device)  # 移动输入到目标设备

    # 1. STFT预处理（输出单边谱）
    stft_spec = stft_processor.stft_preprocess(waveform)
    print(f"STFT输出形状: {stft_spec.shape}")  # 预期: (1, 2049, n_frame, 2)
    # 验证单边谱频点数量（n_fft//2 + 1 = 4096//2 + 1 = 2049）
    assert stft_spec.shape[1] == 2049, f"STFT频点数量错误，预期2049，实际{stft_spec.shape[1]}"

    # 2. ISTFT后处理（还原波形）
    recon_waveform = stft_processor.istft_postprocess(stft_spec)
    recon_length = recon_waveform.shape[1]
    print(f"还原信号长度: {recon_length} 采样点")
    print(f"原始与还原长度差异: {abs(recon_length - original_length)}")
    # 验证长度差异在允许范围内（center=True时通常±1）
    assert abs(recon_length - original_length) <= 1, "长度差异过大，可能参数不匹配"

    # 3. 计算重建质量（信噪比SNR和均方误差MSE）
    # 对齐长度（截取到较短的一方）
    min_len = min(original_length, recon_length)
    original_cropped = original_signal[:min_len]  # 原始信号（numpy）
    recon_cropped = recon_waveform[0, :min_len].cpu().numpy()  # 还原信号（转CPU并转numpy）

    # 计算MSE（均方误差）
    mse = np.mean((original_cropped - recon_cropped) **2)
    # 计算SNR（信噪比，单位dB）
    signal_power = np.mean(original_cropped** 2)
    snr = 10 * np.log10(signal_power / (mse + 1e-12))  # 加小值避免除零

    print(f"重建均方误差 (MSE): {mse:.6f}")
    print(f"重建信噪比 (SNR): {snr:.2f} dB")
    # 验证重建质量（STFT+ISTFT理想重建SNR应远大于30dB）
    assert snr > 30, f"重建质量不佳，SNR过低（{snr:.2f} dB）"

    print("\n✅ 所有验证通过，STFT预处理和ISTFT后处理功能正常！")