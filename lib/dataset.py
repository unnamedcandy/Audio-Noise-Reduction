import torch
import wave
import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt  # 用于绘图


class CandyDataset(Dataset):
    def __init__(self, mode="train"):
        if mode == "train":
            self.NoisyFiles = "../data/noisy_trainset_56spk_wav"  # 带噪语音训练集路径
            self.CleanFiles = "../data/clean_trainset_56spk_wav"  # 干净语音训练集路径
            self.TextFiles = "../data/trainset_56spk_txt"  # 文字集路径
        elif mode == "test":
            self.NoisyFiles = "../data/noisy_testset_wav"  # 带噪语音测试集路径
            self.CleanFiles = "../data/clean_testset_wav"  # 干净语音测试集路径
            self.TextFiles = "../data/testset_txt"  # 文字集路径
        # 获取所有文件名（确保带噪和干净语音文件名一一对应）
        self.FileNames = [f for f in os.listdir(self.CleanFiles) if f.endswith(".wav")]
        self.TextNames = [f for f in os.listdir(self.TextFiles) if f.endswith(".txt")]

    def __len__(self):
        """返回数据集样本总数（即WAV文件数量）"""
        return len(self.FileNames)

    def __getitem__(self, item):
        """
        根据索引获取单个样本（带噪语音+对应干净语音+对应文字信息）
        参数：
            item: 样本索引
        返回：
            nframes: 音频帧数
            noisy_data: 处理后的带噪语音数据
            clean_data: 处理后的干净语音数据
            text_data:  处理后的文字数据
            framerate: 音频采样率（用于频谱图计算）
        """
        # 获取当前索引对应的文件名
        filename = self.FileNames[item]
        textname = self.TextNames[item]
        noisy_path = os.path.join(self.NoisyFiles, filename)
        clean_path = os.path.join(self.CleanFiles, filename)
        text_path = os.path.join(self.TextFiles, textname)

        # 读取音频文件信息及数据
        with wave.open(noisy_path, 'rb') as nwf, wave.open(clean_path, 'rb') as cwf:
            nframes = nwf.getnframes()
            framerate = nwf.getframerate()  # 获取采样率（用于计算频率轴）
            # 读取并转换音频数据
            noisy_bytes = nwf.readframes(nframes)
            clean_bytes = cwf.readframes(nframes)
            # 1. 先用 NumPy 解析字节数据为 int16 数组
            noisy_np = np.frombuffer(noisy_bytes, dtype=np.int16).astype(np.float32)
            clean_np = np.frombuffer(clean_bytes, dtype=np.int16).astype(np.float32)
            # 2. 转换为 PyTorch 张量（保留数据类型为 float32）
            noisy_data = torch.from_numpy(noisy_np)
            clean_data = torch.from_numpy(clean_np)
        # 读取文字信息
        with open(text_path, "r", encoding="gbk") as f:
            text_data = f.readline()

        return nframes, framerate, noisy_data, clean_data, text_data


if __name__ == '__main__':
    # 初始化数据集
    train_ds = CandyDataset()

    # 初始化变量：记录最大帧数及对应样本
    max_nframes = -1
    max_sample = None  # 存储最大样本的完整数据（nframes, framerate, noisydata, cleandata, textdata）

    # 遍历所有样本
    for i in range(len(train_ds)):
        # 获取当前样本的信息（注意__getitem__返回顺序：nframes, framerate, noisydata, cleandata, textdata）
        nframes, framerate, noisydata, cleandata, textdata = train_ds[i]

        # 比较当前样本帧数与已知最大帧数
        if nframes > max_nframes:
            max_nframes = nframes  # 更新最大帧数
            # 保存当前样本为“最大样本”
            max_sample = (nframes, framerate, noisydata, cleandata, textdata)

    nframes, framerate, noisydata, cleandata, textdata = max_sample
    duration = nframes / framerate  # 计算时长（秒）
    print(f"最大样本信息：")
    print(f"帧数：{nframes}")
    print(f"采样率：{framerate} Hz")
    print(f"时长：{duration:.2f} 秒")
    print(f"带噪语音数据长度：{len(noisydata)}")
    print("\n文字内容：", textdata)

    # -------------------------- 频谱图绘制 --------------------------
    # 1. 计算短时傅里叶变换(STFT)
    # 参数说明：
    # - nperseg：每个窗口的长度（影响频率分辨率，1024为常用值）
    # - return_onesided：只返回单边频谱（音频为实信号，单边频谱足够）
    stft_noisy = torch.stft(
        noisydata,
        n_fft=4096,
        window=torch.hann_window(4096),
        return_complex=True
    )
    stft_clean = torch.stft(
        cleandata,
        n_fft=4096,
        window=torch.hann_window(4096),
        return_complex=True
    )

    print("shape", stft_clean.shape)

    print("\n复原：", torch.istft(
        stft_clean,
        n_fft=4096,  # 与stft的n_fft一致
        hop_length=1024,  # 与stft的hop_length一致
        win_length=4096,  # 与stft的win_length一致
        window=torch.hann_window(4096),  # 与stft的窗函数一致（若为复数窗，需保持一致）
        length=nframes  # 可选：指定输出长度与原始信号一致
    ))
    print("\n原：", cleandata)

    # 2. 转换为分贝(dB)单位（更符合人耳对响度的感知）
    # 加1e-8避免log(0)错误
    Zxx_noisy_db = 20 * torch.log10(torch.abs(stft_noisy) + 1e-8)
    Zxx_clean_db = 20 * torch.log10(torch.abs(stft_clean) + 1e-8)

    # 3. 绘制频谱图
    plt.figure(figsize=(14, 10))

    freq = torch.linspace(0, 48000 / 2, 4096 // 2 + 1).numpy()  # 单边频谱

    # 推导时间轴（单位：秒）
    time_steps = Zxx_noisy_db.shape[1]  # 时间帧数
    time = torch.linspace(0, (time_steps - 1) * 1024 / 48000, time_steps).numpy()

    # 带噪语音频谱图
    plt.subplot(2, 1, 1)
    # pcolormesh：绘制伪彩色图（x轴：时间，y轴：频率，颜色：幅度dB）
    plt.pcolormesh(time, freq, Zxx_noisy_db, shading='gouraud', cmap='viridis')
    plt.title('Noisy Speech Spectrogram', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.ylim(0, 8000)  # 语音主要能量集中在0-8kHz，限制显示范围
    plt.colorbar(label='Magnitude (dB)')  # 颜色条表示幅度

    # 干净语音频谱图
    plt.subplot(2, 1, 2)
    plt.pcolormesh(time, freq, Zxx_clean_db, shading='gouraud', cmap='viridis')
    plt.title('Clean Speech Spectrogram', fontsize=12)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.ylim(0, 8000)
    plt.colorbar(label='Magnitude (dB)')

    plt.tight_layout()  # 自动调整子图间距
    plt.show()  # 显示图像
