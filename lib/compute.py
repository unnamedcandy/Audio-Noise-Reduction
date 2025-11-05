import librosa  # 音频处理核心库：提供分帧、采样率转换、特征提取等功能
import numpy as np  # 数值计算库：处理音频数组、矩阵运算（如MGC系数计算、距离求解）
from scipy.io import wavfile  # WAV音频读取库：读取WAV文件，返回采样率和音频原始数据
import pysptk  # 语音信号处理库：核心用于提取梅尔广义倒谱系数（MGC）、生成窗函数
from scipy.spatial.distance import euclidean  # 欧氏距离计算：衡量两帧MGC系数的数值差异
from fastdtw import fastdtw  # 快速动态时间规整：解决两段音频帧数不匹配（语速差异）问题

import whisper  # OpenAI语音识别模型：将音频转换为文本，为WER计算提供转录结果
import jiwer  # 词错误率计算库：对比参考文本与转录文本，量化语音识别错误率


def compute_MCD(file_original, file_reconstructed):
    """
    计算原始音频与重建（处理后）音频的梅尔倒谱失真（MCD）。
    MCD是衡量两段语音频谱包络相似度的指标，值越小表示频谱特征越接近，音频质量越好。

    参数：
        file_original: str - 原始音频文件路径（如未降噪的纯净语音、原始合成语音）
        file_reconstructed: str - 重建音频文件路径（如降噪后语音、优化后合成语音）

    返回：
        float - MCD值（单位：dB，通常<10dB为优质，>20dB表示失真明显）
    """

    def readmgc(filename):
        """
        嵌套辅助函数：读取音频文件并提取梅尔广义倒谱系数（MGC）。
        MGC是语音频谱包络的紧凑表示，能有效反映语音的音色和清晰度特征。

        参数：
            filename: str - 输入音频文件路径（WAV格式）

        返回：
            np.ndarray - MGC系数矩阵，形状为(帧数, 倒谱阶数+1)（+1因包含0阶能量系数）
        """
        # 1. 读取WAV音频：返回采样率(sr)和音频数据(x)，x默认为16位整数（范围[-32768, 32767]）
        sr, x = wavfile.read(filename)

        # 2. 处理双声道音频：语音评估通常用单声道，若为双声道则取第一声道
        if x.ndim == 2:
            x = x[:, 0]

        # 3. 数据类型转换：将整数型音频转为float64，避免后续计算溢出，保证精度
        x = x.astype(np.float64)

        # 4. 分帧参数设置：基于"短时平稳假设"（语音在短时间内频谱特性稳定）
        frame_length = 1024  # 帧长：1024个采样点（16kHz采样率下约64ms，决定频率分辨率）
        hop_length = 256  # 步长：256个采样点（16kHz下约16ms，决定时间分辨率，重叠率75%减少帧间信息丢失）

        # 5. 音频分帧：将长音频切割为重叠的短帧，转置后每行对应一帧（符合后续帧级处理格式）
        # librosa.util.frame默认按列分帧，转置后形状为(帧数, 帧长)
        frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T

        # 6. 加窗处理：减少"频谱泄漏"（分帧导致帧边界突变，引发频谱计算失真）
        # pysptk.blackman生成Blackman窗：一种平滑窗函数，比矩形窗更能抑制频谱泄漏
        frames *= pysptk.blackman(frame_length)

        # 7. 分帧结果校验：确保每帧长度为设定的frame_length，避免后续MGC计算报错
        assert frames.shape[1] == frame_length, "分帧后每帧长度与设定帧长不匹配，可能导致后续计算错误"

        # 8. MGC提取参数设置（语音信号处理领域经典参数，平衡精度与计算量）
        order = 25  # 倒谱阶数：25阶可充分刻画语音频谱包络，阶数越高细节越丰富但计算量越大
        alpha = 0.41  # 预加重系数：补偿语音高频衰减（模拟人耳对高频的敏感特性）
        stage = 5  # 伽马通滤波器组阶数：控制频谱的"梅尔刻度"（贴近人耳听觉非线性特性）
        gamma = -1.0 / stage  # 广义倒谱变换参数：平衡频谱分辨率与鲁棒性

        # 9. 提取MGC系数：pysptk核心函数，输入分帧数据和参数，输出MGC系数
        mgc = pysptk.mgcep(frames, order, alpha, gamma)

        # 10. 调整MGC形状：确保输出为(帧数, 倒谱阶数+1)，因mgcep返回的系数包含0阶（能量）到order阶
        mgc = mgc.reshape(-1, order + 1)

        return mgc  # 返回提取的MGC系数矩阵

    # MCD计算常数：将欧氏距离转换为dB（分贝）单位的系数
    # 推导依据：语音失真度通常用dB表示，公式化简后得到该常数（10/log10 * sqrt(2)）
    _logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    total_distance = 0.0  # 累计欧氏距离：存储所有对齐帧的MGC差异总和
    total_frames = 0  # 累计对齐帧数：用于计算平均失真，避免音频时长影响结果

    # 提取原始音频和重建音频的MGC系数（获取两段语音的频谱包络特征）
    mgc_original = readmgc(file_original)  # 原始音频的MGC系数
    mgc_reconstructed = readmgc(file_reconstructed)  # 重建音频的MGC系数

    # 动态时间规整（DTW）：解决两段音频帧数不匹配问题（如语速差异导致帧数量不同）
    # fastdtw：高效版DTW算法，返回"最小累计距离"（初始未归一化）和"对齐路径"（(原始帧索引, 重建帧索引)列表）
    # dist=euclidean：使用欧氏距离衡量单帧MGC系数的差异
    distance, path = fastdtw(mgc_original, mgc_reconstructed, dist=euclidean)

    # 归一化初始累计距离：除以两段音频帧数之和，消除时长差异对距离的影响
    distance /= (len(mgc_original) + len(mgc_reconstructed))

    # 从对齐路径中提取原始音频和重建音频的帧索引（确定每对对齐的帧）
    path_original = list(map(lambda l: l[0], path))  # 原始音频对齐后的帧索引列表
    path_reconstructed = list(map(lambda l: l[1], path))  # 重建音频对齐后的帧索引列表

    # 按对齐路径截取MGC系数：确保两段MGC序列长度一致，可逐帧比较
    mgc_aligned_original = mgc_original[path_original]  # 对齐后的原始音频MGC
    mgc_aligned_recon = mgc_reconstructed[path_reconstructed]  # 对齐后的重建音频MGC

    # 计算对齐后的总帧数和逐帧欧氏距离总和
    aligned_frames = mgc_aligned_original.shape[0]  # 对齐后的总帧数
    total_frames += aligned_frames  # 累计帧数更新
    frame_diff = mgc_aligned_original - mgc_aligned_recon  # 逐帧MGC系数差异
    # 计算每帧差异的欧氏距离（按帧求和），累加到总距离
    total_distance += np.sqrt((frame_diff ** 2).sum(axis=1)).sum()

    # 计算最终MCD值：转换为dB单位，并按累计帧数平均（得到全局平均失真）
    MCD_value = _logdb_const * float(total_distance) / float(total_frames)
    return MCD_value  # 返回最终MCD评估结果


def compute_WER(original_text, file_dir):
    """
    计算语音转文本（ASR）的词错误率（WER）。
    WER通过对比"参考文本"（音频真实内容）和"转录文本"（模型识别结果），量化语音可懂度，
    值越小表示识别越准确，间接反映语音清晰度越高（如降噪后可懂度提升）。
    """
    # 1. 加载Whisper语音识别模型：选择"base"轻量模型（平衡速度与精度，适合快速评估）
    # 可选模型："tiny"（最快但精度低）、"small"/"medium"（精度提升）、"large"（最高精度但慢）
    model = whisper.load_model("base")

    # 2. 参考文本（ground truth）：音频对应的真实内容，必须与音频语义完全一致，否则WER计算失真
    original_text = "At least 12 persons saw the man with the revolver in the vicinity of the Tipit crime scene, at or immediately after the shooting. By the evening of November 22, five of them had identified Lee Harvey Oswald in police lineups as the man they saw. A sixth did so the next day. Three others subsequently identified Oswald from a photograph. Two witnesses testified that Oswald resembled the man they had seen. One witness felt he was too distant from the gunman to make a positive identification. A taxi driver, William Skoggins, was eating lunch in his cab, which was parked on Patten, facing the southeast corner of Tenth Street and Patten Avenue, a few feet to the north. A police car moving east on 10th at about 10 or 12 miles an hour passed in front of his cab. About 100 feet from the corner, the police car pulled up alongside a man on the sidewalk. This man dressed in a light-colored jacket approached the car."

    # 3. 加载待评估音频：指定音频路径，统一采样率为16000Hz（Whisper模型默认输入采样率）
    file_dir = r"E:\dataset\result\test.wav"  # 待评估音频文件路径（硬编码，可改为函数参数提升灵活性）
    # librosa.load：加载音频并转为16kHz单声道（Whisper要求的输入格式），返回音频数组和采样率
    audio, _ = librosa.load(file_dir, sr=16000)

    # 4. 语音转文本：Whisper模型自动完成语音分割、特征提取、文本转录
    result = model.transcribe(audio)  # 返回字典，包含转录文本、时间戳等信息
    transcribed_text = result["text"]  # 提取模型输出的转录文本

    # 5. 计算WER：jiwer库自动处理文本标准化（大小写、标点、空格），并计算编辑距离
    # WER公式：(替换数 + 删除数 + 插入数) / 参考文本总词数，范围0~1（0表示完全正确）
    wer_score = jiwer.wer(original_text, transcribed_text)

    # 6. 打印WER结果：保留4位小数，直观展示语音可懂度
    print(f"词错误率（WER）：{wer_score:.4f}")
