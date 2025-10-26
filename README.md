# 简单降噪神经网络

## 处理原理

#### 首先读取 wav 格式音频文件(公开数据已做好分段处理);
#### 再通过傅里叶变换将声音采样信息转化为频域图;
#### 然后利用卷积神经网络对频谱图进行学习;

## 项目结构

* noisy
* ├── data
* │   ├── clean_testset_wav
* │   ├── clean_trainset_28spk_wav
* │   ├── clean_trainset_56spk_wav
* │   ├── noisy_testset_wav
* │   ├── noisy_trainset_28spk_wav
* │   ├── noisy_trainset_56spk_wav
* │   ├── testset_txt
* │   ├── trainset_28spk_txt
* │   ├── trainset_56spk_txt
* │   ├── license_text
* │   ├── log_readme.txt
* │   ├── log_testset.txt
* │   ├── log_trainset_28spk.txt
* │   └── log_trainset_56spk.txt
* ├── lib
* │   ├── __init__.py
* │   ├── dataset.py
* │   └── model.py
* ├── model
* ├── src
* │   ├── train.py
* │   └── test.py
* ├── .gitignore
* ├── README.md
* └── requirements.txt