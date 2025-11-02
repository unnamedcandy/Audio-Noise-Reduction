import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# 导入项目内部组件（确保路径与你的文件结构一致）
from lib.dataset import TrainDataset  # 你的数据集类
from lib.model import UNetSpeechEnhancement  # 你的U-Net模型
from lib.utils import stft  # 你的STFT变换函数（时域→频谱）


def train_speech_enhancement(
    epochs=50,
    batch_size=4,
    lr=1e-4,
    weight_decay=1e-5,
    step_size=10,
    gamma=0.5,
    target_frames=1048576,
    checkpoint_dir="./checkpoints"
):
    """
    语音增强模型训练函数
    参数说明：
        epochs: 训练总轮次
        batch_size: 批次大小（根据GPU显存调整，建议4-8）
        lr: 初始学习率
        weight_decay: 权重衰减（防止过拟合）
        step_size: 学习率调度器的步长（每step_size个epoch衰减）
        gamma: 学习率衰减系数
        target_frames: 音频统一帧数（与Dataset保持一致）
        checkpoint_dir: 模型 checkpoint 保存目录
    """
    # -------------------------- 1. 基础配置（设备/目录）--------------------------
    # 设备选择（优先GPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device} (GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB)")

    # 创建 checkpoint 保存目录（不存在则创建）
    os.makedirs(checkpoint_dir, exist_ok=True)
    # 日志文件（记录每个epoch的损失）
    log_file = os.path.join(checkpoint_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # -------------------------- 2. 数据加载（DataLoader）--------------------------
    print("\n正在加载训练数据集...")
    # 初始化训练数据集
    train_dataset = TrainDataset(target_frames=target_frames)
    # 用DataLoader包装（多线程加载，shuffle=True打乱数据）
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # 线程数（根据CPU核心数调整，建议4-8）
        pin_memory=True,  # 锁定内存，加速数据传输到GPU
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    print(f"数据集加载完成：共 {len(train_dataset)} 个样本，每批 {batch_size} 个，共 {len(train_loader)} 批")

    # -------------------------- 3. 模型/损失/优化器初始化--------------------------
    # 初始化U-Net语音增强模型
    model = UNetSpeechEnhancement(
        in_channels=2,    # 输入通道：幅度谱+相位谱
        out_channels=2,   # 输出通道：预测的干净语音幅度谱+相位谱
        base_filters=8   # 基础卷积核数量（可调整，越大模型越复杂）
    ).to(device)  # 移到目标设备

    # 损失函数：MSE损失（频谱特征像素级差异，适合语音增强）
    criterion = nn.MSELoss().to(device)

    # 优化器：Adam（自适应学习率，适合语音任务）
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay  # 权重衰减防止过拟合
    )

    # 学习率调度器：StepLR（每step_size个epoch衰减学习率）
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=step_size,
        gamma=gamma
    )

    # 打印模型信息
    print(f"\n模型参数总量: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(f"损失函数: MSELoss | 优化器: Adam (lr={lr}) | 学习率调度: StepLR (step={step_size}, gamma={gamma})")

    # -------------------------- 4. 核心训练循环--------------------------
    print("\n开始训练（按 Ctrl+C 终止）...")
    best_loss = float("inf")  # 记录最佳损失（用于保存最优模型）

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()  # 切换到训练模式（启用Dropout/BatchNorm训练行为）
        total_loss = 0.0  # 累计当前epoch的总损失

        # 遍历训练集（tqdm显示进度条）
        for batch_idx, (nframes, noisy_data, clean_data, _) in enumerate(tqdm(
            train_loader, desc=f"Epoch {epoch:03d}/{epochs}", unit="batch"
        )):
            # 1. 数据预处理：时域→频谱（调用STFT函数，转为模型输入格式）
            # 注意：每个样本需单独做STFT（当前stft函数支持单样本，可后续优化批量处理）
            batch_stft_noisy = []  # 存储当前batch的带噪语音频谱
            batch_stft_clean = []  # 存储当前batch的干净语音频谱

            for i in range(batch_size):
                # 处理带噪语音：时域→频谱
                stft_noisy, _, _ = stft(
                    signal=noisy_data[i].to(device),  # 单样本移到设备
                    window_size=4096,
                    step=2048,
                    keep_freq_bins=512,  # 与模型输入频率维度（512）匹配
                    window_type="hamming"
                )
                batch_stft_noisy.append(stft_noisy)

                # 处理干净语音：时域→频谱（作为标签）
                stft_clean, _, _ = stft(
                    signal=clean_data[i].to(device),
                    window_size=4096,
                    step=2048,
                    keep_freq_bins=512,
                    window_type="hamming"
                )
                batch_stft_clean.append(stft_clean)

            # 堆叠为batch张量（形状：[batch_size, 2048, 512, 2]）
            batch_stft_noisy = torch.stack(batch_stft_noisy, dim=0)
            batch_stft_clean = torch.stack(batch_stft_clean, dim=0)

            # 2. 前向传播：模型预测干净频谱
            optimizer.zero_grad()  # 清零梯度（避免累计）
            pred_clean = model(batch_stft_noisy)  # 模型输出：预测的干净频谱

            # 3. 计算损失：预测值 vs 真实值（干净频谱）
            loss = criterion(pred_clean, batch_stft_clean)
            total_loss += loss.item() * batch_size  # 累计损失（乘batch_size还原总损失）

            # 4. 反向传播与参数更新
            loss.backward()  # 计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 梯度裁剪（防止梯度爆炸）
            optimizer.step()  # 更新参数

        # -------------------------- 5. 每个Epoch后处理--------------------------
        # 计算当前epoch的平均损失
        avg_loss = total_loss / len(train_dataset)
        # 学习率调度器更新
        scheduler.step()
        # 计算epoch耗时
        epoch_time = time.time() - start_time

        # 打印当前epoch信息
        log_info = (
            f"Epoch {epoch:03d}/{epochs} | "
            f"Avg Loss: {avg_loss:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )
        print(log_info)

        # 写入日志文件
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_info + "\n")

        # 保存最佳模型（损失最低时）
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存checkpoint（含模型参数、优化器状态、epoch等，支持断点续训）
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch}_loss{avg_loss:.6f}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "lr": scheduler.get_last_lr()[0]
            }, checkpoint_path)
            print(f"最佳模型已保存: {os.path.basename(checkpoint_path)}")

        # 每10个epoch保存一次普通checkpoint（避免最佳模型丢失）
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}_loss{avg_loss:.6f}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
                "lr": scheduler.get_last_lr()[0]
            }, checkpoint_path)
            print(f"定期模型已保存: {os.path.basename(checkpoint_path)}")

    # -------------------------- 6. 训练结束--------------------------
    print(f"\n训练完成！")
    print(f"最佳平均损失: {best_loss:.6f}")
    print(f"日志文件: {log_file}")
    print(f"模型保存目录: {checkpoint_dir}")


# -------------------------- 主函数入口（执行训练）--------------------------
if __name__ == "__main__":
    # 训练超参数配置（可根据需求调整）
    TRAIN_CONFIG = {
        "epochs": 50,          # 总训练轮次
        "batch_size": 8,       # 批次大小（GPU显存不足时减小，如2）
        "lr": 1e-3,            # 初始学习率
        "weight_decay": 1e-5,  # 权重衰减
        "step_size": 15,       # 学习率每15个epoch衰减一次
        "gamma": 0.5,          # 学习率衰减系数（每次乘以0.5）
        "target_frames": 1048576,  # 与Dataset保持一致的音频帧数
        "checkpoint_dir": "./speech_enhancement_checkpoints"  # 模型保存目录
    }

    # 启动训练
    train_speech_enhancement(**TRAIN_CONFIG)