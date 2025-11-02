import os
import torch
import wave
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入项目内部组件（与训练代码路径保持一致）
from lib.model import UNetSpeechEnhancement  # 与训练一致的U-Net模型
from lib.utils import stft, istft  # 训练时用的STFT/ISTFT
from lib.compute import compute_MCD  # 计算MCD的函数
from lib.dataset import TestDataset  # 复用测试数据集类（关键：保证数据处理逻辑统一）


def load_trained_model(model_path, device, in_channels=2, out_channels=2, base_filters=8):
    """
    加载训练好的模型权重（逻辑不变，确保模型结构与训练一致）
    """
    model = UNetSpeechEnhancement(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=base_filters
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # 切换评估模式（关闭Dropout/BatchNorm训练行为）

    print(f"成功加载模型：{os.path.basename(model_path)}")
    print(f"模型训练终止epoch：{checkpoint['epoch']} | 训练时最佳损失：{checkpoint['best_loss']:.6f}")
    return model


def save_audio(audio_data, save_path, sr=48000, original_nframes=None):
    """
    保存音频文件（适配TestDataset的音频格式，支持截断到原始长度）
    参数：
        audio_data: 时域音频数据（np.ndarray，float32类型，范围[-32768, 32768]）
        save_path: 保存路径
        sr: 采样率（与TestDataset一致，默认16kHz）
        original_nframes: 原始音频帧数（用于截断统一长度时的补零部分）
    """
    # 1. 截断到原始长度（去除TestDataset统一长度时的补零，避免影响MCD）
    if original_nframes is not None and len(audio_data) > original_nframes:
        audio_data = audio_data[:original_nframes]

    # 2. 格式转换（float32→int16，符合WAV标准）
    audio_data = np.clip(audio_data, -32768, 32767)  # 防止溢出（TestDataset中是int16转float32，范围对应[-32768, 32767]）
    audio_int16 = audio_data.astype(np.int16)

    # 3. 保存为WAV文件
    with wave.open(save_path, "wb") as f:
        f.setnchannels(1)  # 单声道（与TestDataset一致）
        f.setsampwidth(2)  # 采样宽度2字节（int16，与TestDataset一致）
        f.setframerate(sr)
        f.setnframes(len(audio_int16))
        f.writeframes(audio_int16.tobytes())

    return save_path


def denoise_sample(model, noisy_data, original_nframes, device,
                   window_size=4096, step=2048, keep_freq_bins=512, window_type="hamming"):
    """
    对单个测试样本进行降噪推理（适配最新的stft和istft函数）
    """
    # 1. 时域→频谱：调用stft获取结果+归一化参数（mag_mean, mag_std）
    stft_noisy, mag_mean, mag_std = stft(  # 关键：获取stft返回的三个值
        signal=noisy_data,  # 已在TestDataset中转为float32，直接传入
        window_size=window_size,
        step=step,
        keep_freq_bins=keep_freq_bins,
        window_type=window_type  # 与训练时一致（默认hamming）
    )

    # 2. 模型推理（扩展batch维度）
    with torch.no_grad():
        stft_noisy_batch = stft_noisy[None, :, :, :]  # 形状：[1, target_frames, keep_freq_bins, 2]
        stft_denoised_batch = model(stft_noisy_batch)
        stft_denoised = stft_denoised_batch[0, :, :, :]  # 去除batch维度

    # 3. 频谱→时域：调用istft，补充所有必填参数
    denoised_data = istft(
        stft_result=stft_denoised,  # 模型输出的频谱
        original_length=len(noisy_data),  # 对应istft的original_length（TestDataset统一后的长度）
        mag_mean=mag_mean,  # 从stft获取的幅度均值
        mag_std=mag_std,  # 从stft获取的幅度标准差
        window_size=window_size,
        step=step,
        keep_freq_bins=keep_freq_bins,
        window_type=window_type  # 与stft保持一致
    )
    denoised_data = denoised_data.cpu().numpy()  # 移回CPU并转numpy

    # 4. 截断到原始帧数（去除TestDataset的补零）
    if original_nframes is not None and len(denoised_data) > original_nframes:
        denoised_data = denoised_data[:original_nframes]

    return denoised_data


def test_model_with_testdataset(model_path, save_denoised_dir,
                                in_channels=2, out_channels=2, base_filters=8,
                                target_frames=1048576, batch_size=4):
    """
    核心测试函数：用TestDataset加载数据，批量降噪并计算MCD
    参数：
        model_path: 训练好的模型权重路径
        save_denoised_dir: 降噪后音频保存目录
        in_channels/out_channels/base_filters: 与训练一致的模型参数
        target_frames: 与TestDataset一致的统一长度（默认1048576）
        batch_size: 测试时的批次大小（根据显存调整，无需太大）
    """
    # 1. 基础配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"测试设备: {device}")
    os.makedirs(save_denoised_dir, exist_ok=True)

    # 2. 加载模型和测试数据集（复用TestDataset，保证数据处理逻辑统一）
    model = load_trained_model(model_path, device, in_channels, out_channels, base_filters)
    test_dataset = TestDataset(target_frames=target_frames)  # 实例化测试数据集
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集无需打乱，保持样本顺序
        num_workers=4,  # 线程数（根据CPU调整）
        pin_memory=True
    )
    print(f"测试数据集加载完成：共 {len(test_dataset)} 个样本，每批 {batch_size} 个，共 {len(test_loader)} 批")

    # 3. 临时文件目录（保存原始长度的干净音频，用于计算MCD）
    temp_clean_dir = os.path.join(save_denoised_dir, "temp_clean")
    os.makedirs(temp_clean_dir, exist_ok=True)

    # 4. 批量测试：遍历TestDataset，降噪+计算MCD
    mcd_results = []  # 存储每个样本的MCD
    sample_info = []  # 存储样本信息（文件名、MCD值）

    for batch_idx, (original_nframes, noisy_data, clean_data, text_data) in enumerate(tqdm(
            test_loader, desc="测试集批量降噪与MCD计算", unit="batch"
    )):
        # 遍历当前batch的每个样本（batch内样本独立处理）
        for idx_in_batch in range(len(noisy_data)):
            # 4.1 获取单个样本的信息（从batch中拆分）
            sample_idx = batch_idx * batch_size + idx_in_batch
            filename = test_dataset.FileNames[sample_idx]  # 样本文件名（从TestDataset获取）
            orig_nframes = original_nframes[idx_in_batch].item()  # 原始帧数（int）
            noisy_data_single = noisy_data[idx_in_batch].numpy()  # 带噪音频（float32，统一长度）
            clean_data_single = clean_data[idx_in_batch].numpy()  # 干净音频（float32，统一长度）

            # 4.2 降噪推理
            denoised_data_single = denoise_sample(
                model=model,
                noisy_data=noisy_data_single,
                original_nframes=orig_nframes,
                device=device
            )

            # 4.3 保存降噪音频和原始长度的干净音频（用于compute_MCD）
            # 保存降噪音频
            denoised_save_path = os.path.join(save_denoised_dir, f"denoised_{filename}")
            save_audio(denoised_data_single, denoised_save_path, original_nframes=orig_nframes)

            # 保存原始长度的干净音频（TestDataset中是统一长度，需截断后保存）
            clean_save_path = os.path.join(temp_clean_dir, f"clean_{filename}")
            save_audio(clean_data_single, clean_save_path, original_nframes=orig_nframes)

            # 4.4 调用compute_MCD计算音质指标（用原始长度的干净音频vs降噪音频）
            try:
                mcd_value = compute_MCD(
                    file_original=clean_save_path,  # 原始长度的干净音频（MCD基准）
                    file_reconstructed=denoised_save_path  # 降噪后音频
                )
                mcd_results.append(mcd_value)
                sample_info.append((filename, mcd_value))
                print(f"样本 {filename} | MCD: {mcd_value:.2f} dB")
            except Exception as e:
                print(f"警告：样本 {filename} 计算MCD失败，错误：{str(e)}，跳过该样本")
                continue

    # 5. 统计测试集整体效果（与之前逻辑一致）
    if len(mcd_results) > 0:
        avg_mcd = np.mean(mcd_results)
        std_mcd = np.std(mcd_results)
        best_mcd = np.min(mcd_results)
        worst_mcd = np.max(mcd_results)

        # 打印统计结果
        print("\n" + "=" * 50)
        print("测试集MCD统计结果（基于TestDataset）")
        print("=" * 50)
        print(f"有效测试样本数：{len(mcd_results)} / {len(test_dataset)}")
        print(f"平均MCD：{avg_mcd:.2f} dB")
        print(f"MCD标准差：{std_mcd:.2f} dB")
        print(f"最佳MCD（音质最好）：{best_mcd:.2f} dB")
        print(f"最差MCD（音质最差）：{worst_mcd:.2f} dB")
        print("=" * 50)

        # 保存MCD日志（包含每个样本的详情）
        log_path = os.path.join(save_denoised_dir, "mcd_test_results_with_testdataset.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"模型路径：{model_path}\n")
            f.write(f"测试数据集：TestDataset（target_frames={target_frames}）\n")
            f.write(f"测试时间：{os.popen('date').read().strip()}\n")
            f.write(f"有效测试样本数：{len(mcd_results)} / {len(test_dataset)}\n")
            f.write(f"平均MCD：{avg_mcd:.2f} dB\n")
            f.write(f"MCD标准差：{std_mcd:.2f} dB\n")
            f.write(f"最佳MCD：{best_mcd:.2f} dB\n")
            f.write(f"最差MCD：{worst_mcd:.2f} dB\n\n")
            f.write("单个样本MCD详情：\n")
            for filename, mcd in sample_info:
                f.write(f"{filename}: {mcd:.2f} dB\n")
        print(f"\nMCD统计日志已保存：{log_path}")

        # 可选：删除临时干净音频目录（若无需保留）
        # import shutil; shutil.rmtree(temp_clean_dir)
    else:
        print("\n无有效测试样本，未生成MCD统计结果")


# -------------------------- 测试入口（配置参数后执行）--------------------------
if __name__ == "__main__":
    # 测试参数配置（关键：与训练参数和TestDataset保持一致）
    TEST_CONFIG = {
        "model_path": "./speech_enhancement_checkpoints/best_model_epoch8_loss0.071942.pth",  # 最优模型路径
        "save_denoised_dir": "./denoised_results_with_testdataset",  # 降噪结果保存目录
        "in_channels": 2,  # 与训练一致（幅度+相位）
        "out_channels": 2,  # 与训练一致
        "base_filters": 16,  # 与训练一致（训练代码中base_filters=16）
        "target_frames": 1048576,  # 与TestDataset的__init__参数一致
        "batch_size": 4  # 测试批次大小（GPU显存不足可改2或1）
    }

    # 启动测试（复用TestDataset加载数据）
    test_model_with_testdataset(**TEST_CONFIG)
