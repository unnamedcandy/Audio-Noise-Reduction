import torch
import wave
import os
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self):
        self.NoisyFiles = os.path.join("../data/noisy_trainset_28spk_wav")
        self.CleanFiles = os.path.join("../data/clean_trainset_28spk_wav")
        self.FileNames = os.listdir("../data/clean_trainset_28spk_wav")

    def __len__(self):
        return self.FileNames.__len__()

    def __getitem__(self, item):
        with wave.open(os.path.join(self.NoisyFiles, self.FileNames[item]), 'rb') as nwf:
            with wave.open(os.path.join(self.CleanFiles, self.FileNames[item]), 'rb') as cwf:
                nframes = nwf.getnframes()
                noisydata = nwf.readframes(nframes)
                cleandata = cwf.readframes(nframes)
        return {nframes, noisydata, cleandata}


class TestDataset(Dataset):
    def __init__(self):
        self.NoisyFiles = os.path.join("../data/noisy_testset_wav")
        self.CleanFiles = os.path.join("../data/clean_testset_wav")
        self.FileNames = os.listdir("../data/clean_testset_wav")

    def __len__(self):
        return self.FileNames.__len__()

    def __getitem__(self, item):
        with wave.open(os.path.join(self.NoisyFiles, self.FileNames[item]), 'rb') as nwf:
            with wave.open(os.path.join(self.CleanFiles, self.FileNames[item]), 'rb') as cwf:
                nframes = nwf.getnframes()
                noisydata = nwf.readframes(nframes)
                cleandata = cwf.readframes(nframes)
        return {nframes, noisydata, cleandata}


if __name__ == '__main__':
    ds = TrainDataset()
    nframes, noisydata, cleandata = ds[0]
    print(noisydata)
