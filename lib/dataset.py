import torch
import wave  # 用于读取WAV格式音频文件
import os
import numpy as np
from torch.utils.data import Dataset  # 继承Dataset类实现自定义数据集


class TrainDataset(Dataset):
    """
    训练数据集类，用于加载带噪语音和对应干净语音数据，为语音增强模型提供训练样本。
    功能：读取WAV文件，统一音频长度，转换为模型可处理的数值数组。
    """

    def __init__(self, target_frames=1048576):  # 目标帧数：1048576（2^20，便于后续FFT等幂次操作）
        # 带噪语音和干净语音的文件夹路径
        self.NoisyFiles = "../data/noisy_trainset_56spk_wav"  # 带噪语音训练集路径
        self.CleanFiles = "../data/clean_trainset_56spk_wav"  # 干净语音训练集路径
        self.TextFiles = "../data/trainset_56spk_txt"  # 文字集路径

        # 获取所有文件名（确保带噪和干净语音文件名一一对应）
        self.FileNames = [f for f in os.listdir(self.CleanFiles) if f.endswith(".wav")]
        self.TextNames = [f for f in os.listdir(self.TextFiles) if f.endswith(".txt")]
        self.target_frames = target_frames  # 统一后的音频帧数，保证输入模型的样本长度一致

    def __len__(self):
        """返回数据集样本总数（即WAV文件数量）"""
        return len(self.FileNames)

    def __getitem__(self, item):
        """
        根据索引获取单个样本（带噪语音+对应干净语音+对应文字信息）
        参数：
            item: 样本索引
        返回：
            nframes: 原始音频的帧数（用于后续可能的恢复原始长度操作）
            noisy_data: 处理后的带噪语音数据（NumPy数组，长度为target_frames）
            clean_data: 处理后的干净语音数据（NumPy数组，长度为target_frames）
            text_data:  处理后的文字数据（NumPy数组，长度为target_frames）
        """
        # 获取当前索引对应的文件名（带噪和干净语音文件名相同，仅路径不同）
        filename = self.FileNames[item]
        textname = self.TextNames[item]
        noisy_path = os.path.join(self.NoisyFiles, filename)  # 带噪语音文件完整路径
        clean_path = os.path.join(self.CleanFiles, filename)  # 干净语音文件完整路径
        text_path = os.path.join(self.TextFiles, textname)  # 文字文件完整路径

        # 1. 读取音频文件信息及字节数据
        # 使用wave.open打开WAV文件，'rb'表示只读二进制模式
        with wave.open(noisy_path, 'rb') as nwf, wave.open(clean_path, 'rb') as cwf:
            # 校验带噪和干净语音的帧数是否一致（确保是同一段语音的带噪/干净版本）
            assert nwf.getnframes() == cwf.getnframes(), f"{filename} 带噪/干净音频帧数不一致"
            nframes = nwf.getnframes()  # 获取原始音频的总帧数
            sampwidth = nwf.getsampwidth()  # 获取采样宽度（单位：字节，决定数据类型）

            # 读取音频帧的原始字节数据（未解析的二进制数据）
            noisy_bytes = nwf.readframes(nframes)
            clean_bytes = cwf.readframes(nframes)

        # 2. 字节数据转换为NumPy数组（模型需要数值型输入，而非二进制字节）
        # 根据采样宽度选择对应的数据类型（音频常用16位或8位存储）
        if sampwidth == 2:  # 16位音频（最常见，用int16存储）
            # 从字节数据解析为int16数组，再转为float32（避免整数运算溢出，适合模型计算）
            noisy_data = np.frombuffer(noisy_bytes, dtype=np.int16).astype(np.float32)
            clean_data = np.frombuffer(clean_bytes, dtype=np.int16).astype(np.float32)
        elif sampwidth == 1:  # 8位音频（较少见，用uint8存储，无符号整数）
            noisy_data = np.frombuffer(noisy_bytes, dtype=np.uint8).astype(np.float32)
            clean_data = np.frombuffer(clean_bytes, dtype=np.uint8).astype(np.float32)
        else:
            # 不支持的采样宽度（如32位浮点型等，需根据实际数据扩展）
            raise ValueError(f"不支持的采样宽度：{sampwidth}字节（仅支持8/16位）")

        # 3. 统一音频长度：补齐（短于目标长度）或截断（长于目标长度）
        def pad_or_truncate(data, target_len):
            """
            辅助函数：将输入数据调整为目标长度
            参数：
                data: 原始音频数据（NumPy数组）
                target_len: 目标长度（即self.target_frames）
            返回：
                调整后的音频数据（长度为target_len）
            """
            data_len = len(data)
            if data_len < target_len:
                # 补齐：在末尾补0（静音填充，减少对有效信号的干扰）
                pad_len = target_len - data_len
                data_padded = np.pad(
                    data,
                    pad_width=(0, pad_len),  # 前补0，后补pad_len个0
                    mode='constant',
                    constant_values=0.0  # 填充值为0（静音）
                )
                return data_padded
            else:
                # 截断：保留前target_len帧（取音频起始部分，避免随机截断导致的样本不一致）
                return data[:target_len]

        # 对带噪和干净音频同步处理（确保长度完全一致，避免训练时输入输出不匹配）
        noisy_data = pad_or_truncate(noisy_data, self.target_frames)
        clean_data = pad_or_truncate(clean_data, self.target_frames)

        # 验证处理后的数据长度是否符合预期（避免后续模型输入维度错误）
        assert len(noisy_data) == self.target_frames, f"带噪音频补齐后长度错误：{len(noisy_data)}"
        assert len(clean_data) == self.target_frames, f"干净音频补齐后长度错误：{len(clean_data)}"

        # 读取文字信息
        with open(text_path, "r", encoding="gbk") as f:
            text_data = f.readline()

        return nframes, noisy_data, clean_data, text_data


class TestDataset(Dataset):
    """
    测试数据集类，功能与TrainDataset完全一致，仅数据路径指向测试集。
    用于模型测试阶段加载带噪语音和对应干净语音，评估模型性能。
    """

    def __init__(self, target_frames=1048576):
        self.NoisyFiles = "../data/noisy_testset_wav"  # 带噪语音测试集路径
        self.CleanFiles = "../data/clean_testset_wav"  # 干净语音测试集路径
        self.TextFiles = "../data/testset_txt"  # 文字集路径

        # 获取所有文件名（确保带噪和干净语音文件名一一对应）
        self.FileNames = [f for f in os.listdir(self.CleanFiles) if f.endswith(".wav")]
        self.TextNames = [f for f in os.listdir(self.TextFiles) if f.endswith(".txt")]
        self.target_frames = target_frames

    def __len__(self):
        return len(self.FileNames)

    def __getitem__(self, item):
        filename = self.FileNames[item]
        textname = self.TextNames[item]
        noisy_path = os.path.join(self.NoisyFiles, filename)  # 带噪语音文件完整路径
        clean_path = os.path.join(self.CleanFiles, filename)  # 干净语音文件完整路径
        text_path = os.path.join(self.TextFiles, textname)  # 文字文件完整路径

        with wave.open(noisy_path, 'rb') as nwf, wave.open(clean_path, 'rb') as cwf:
            assert nwf.getnframes() == cwf.getnframes(), f"{filename} 帧数不一致"
            nframes = nwf.getnframes()
            sampwidth = nwf.getsampwidth()
            noisy_bytes = nwf.readframes(nframes)
            clean_bytes = cwf.readframes(nframes)

        # 字节数据转NumPy数组（同训练集处理逻辑，保证数据格式一致）
        if sampwidth == 2:
            noisy_data = np.frombuffer(noisy_bytes, dtype=np.int16).astype(np.float32)
            clean_data = np.frombuffer(clean_bytes, dtype=np.int16).astype(np.float32)
        elif sampwidth == 1:
            noisy_data = np.frombuffer(noisy_bytes, dtype=np.uint8).astype(np.float32)
            clean_data = np.frombuffer(clean_bytes, dtype=np.uint8).astype(np.float32)
        else:
            raise ValueError(f"不支持的采样宽度：{sampwidth}字节")

        # 统一长度（与训练集处理逻辑一致，确保测试时输入格式与训练时相同）
        def pad_or_truncate(data, target_len):
            data_len = len(data)
            if data_len < target_len:
                pad_len = target_len - data_len
                return np.pad(data, (0, pad_len), mode='constant', constant_values=0.0)
            else:
                return data[:target_len]

        noisy_data = pad_or_truncate(noisy_data, self.target_frames)
        clean_data = pad_or_truncate(clean_data, self.target_frames)

        # 验证长度一致性
        assert len(noisy_data) == self.target_frames and len(clean_data) == self.target_frames

        # 读取文字信息
        with open(text_path, "r", encoding="gbk") as f:
            text_data = f.readline()

        return nframes, noisy_data, clean_data, text_data


# 测试代码：验证数据集类的功能是否正常
if __name__ == '__main__':
    train_ds = TrainDataset(target_frames=1048576)

    nframes, noisydata, cleandata, textdata = train_ds[0]

    print("带噪语音数据示例：", noisydata[:10])  # 只打印前10个值，避免输出过长

    print("\n带噪语音FFT结果示例：", np.fft.fft(noisydata)[:10])  # 只打印前10个频率分量

    print("\n文字内容：", textdata)
