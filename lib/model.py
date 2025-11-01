import torch
import torch.nn as nn


class UNetSpeechEnhancement(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, base_filters=16):
        """
        适用于语音频谱增强的U-Net模型
        输入/输出形状: (batch_size, 2, 2048, 512)
        其中通道维度对应：[幅度谱, 相位谱]

        参数:
            in_channels: 输入通道数（固定为2，幅度+相位）
            out_channels: 输出通道数（固定为2，对应干净语音的幅度+相位）
            base_filters: 基础卷积核数量（后续每层翻倍）
        """
        super(UNetSpeechEnhancement, self).__init__()

        # -------------------------- 编码器（下采样）--------------------------
        # 编码器作用：逐步压缩时间和频率维度，提取高层噪声抑制特征
        self.encoder1 = self._conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样：(2048,512)→(1024,256)

        self.encoder2 = self._conv_block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样：(1024,256)→(512,128)

        self.encoder3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样：(512,128)→(256,64)

        self.encoder4 = self._conv_block(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样：(256,64)→(128,32)

        # -------------------------- 瓶颈层（特征融合）--------------------------
        self.bottleneck = self._conv_block(base_filters * 8, base_filters * 16)

        # -------------------------- 解码器（上采样）--------------------------
        # 解码器作用：逐步恢复时间和频率维度，结合编码器特征补充细节
        self.upconv4 = nn.ConvTranspose2d(
            base_filters * 16, base_filters * 8,
            kernel_size=2, stride=2  # 上采样：(128,32)→(256,64)
        )
        self.decoder4 = self._conv_block(base_filters * 16, base_filters * 8)  # 拼接后通道数：8*2=16

        self.upconv3 = nn.ConvTranspose2d(
            base_filters * 8, base_filters * 4,
            kernel_size=2, stride=2  # 上采样：(256,64)→(512,128)
        )
        self.decoder3 = self._conv_block(base_filters * 8, base_filters * 4)  # 拼接后通道数：4*2=8

        self.upconv2 = nn.ConvTranspose2d(
            base_filters * 4, base_filters * 2,
            kernel_size=2, stride=2  # 上采样：(512,128)→(1024,256)
        )
        self.decoder2 = self._conv_block(base_filters * 4, base_filters * 2)  # 拼接后通道数：2*2=4

        self.upconv1 = nn.ConvTranspose2d(
            base_filters * 2, base_filters,
            kernel_size=2, stride=2  # 上采样：(1024,256)→(2048,512)
        )
        self.decoder1 = self._conv_block(base_filters * 2, base_filters)  # 拼接后通道数：1*2=2

        # -------------------------- 输出层（恢复通道）--------------------------
        self.out_conv = nn.Conv2d(
            base_filters, out_channels,
            kernel_size=1, padding=0  # 1x1卷积仅调整通道数，不改变尺寸
        )

    def _conv_block(self, in_channels, out_channels):
        """基础卷积块：2次卷积+批归一化+ReLU激活（保持空间尺寸不变）"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # 稳定训练，加速收敛
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        前向传播：输入带噪频谱，输出干净频谱
        x shape: (batch_size, 2, 2048, 512)  # [batch, 幅度/相位, 时间帧, 频率bin]
        """
        # 编码器特征提取 + 保存中间特征（用于跳跃连接）
        enc1 = self.encoder1(x)  # (batch, 16, 2048, 512)
        enc2 = self.encoder2(self.pool1(enc1))  # (batch, 32, 1024, 256)
        enc3 = self.encoder3(self.pool2(enc2))  # (batch, 64, 512, 128)
        enc4 = self.encoder4(self.pool3(enc3))  # (batch, 128, 256, 64)

        # 瓶颈层特征融合
        bottleneck = self.bottleneck(self.pool4(enc4))  # (batch, 256, 128, 32)

        # 解码器上采样 + 跳跃连接（拼接编码器同尺度特征）
        dec4 = self.upconv4(bottleneck)  # (batch, 128, 256, 64)
        dec4 = torch.cat([dec4, enc4], dim=1)  # 拼接后：(batch, 256, 256, 64)
        dec4 = self.decoder4(dec4)  # (batch, 128, 256, 64)

        dec3 = self.upconv3(dec4)  # (batch, 64, 512, 128)
        dec3 = torch.cat([dec3, enc3], dim=1)  # 拼接后：(batch, 128, 512, 128)
        dec3 = self.decoder3(dec3)  # (batch, 64, 512, 128)

        dec2 = self.upconv2(dec3)  # (batch, 32, 1024, 256)
        dec2 = torch.cat([dec2, enc2], dim=1)  # 拼接后：(batch, 64, 1024, 256)
        dec2 = self.decoder2(dec2)  # (batch, 32, 1024, 256)

        dec1 = self.upconv1(dec2)  # (batch, 16, 2048, 512)
        dec1 = torch.cat([dec1, enc1], dim=1)  # 拼接后：(batch, 32, 2048, 512)
        dec1 = self.decoder1(dec1)  # (batch, 16, 2048, 512)

        # 输出层：恢复为2通道（干净语音的幅度谱+相位谱）
        out = self.out_conv(dec1)  # (batch, 2, 2048, 512)
        return out


# 验证模型输入输出形状匹配性
if __name__ == "__main__":
    # 模拟STFT处理后的带噪频谱特征 (batch_size=2, channels=2, time=2048, frequency=512)
    dummy_input = torch.randn(2, 2, 2048, 512)

    # 初始化模型
    model = UNetSpeechEnhancement()

    # 前向传播
    dummy_output = model(dummy_input)

    # 验证形状
    print(f"输入形状: {dummy_input.shape}")  # 应输出: torch.Size([2, 2, 2048, 512])
    print(f"输出形状: {dummy_output.shape}")  # 应输出: torch.Size([2, 2, 2048, 512])
    print("模型输入输出形状匹配，可用于频谱域语音增强任务！")