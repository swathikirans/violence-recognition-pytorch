import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class makeDataset(Dataset):
    def __init__(self, dataset, labels, numFrames, spatial_transform, seqLen=20):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.numFrames = numFrames
        self.seqLen = seqLen

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        numFrames = self.numFrames[idx]
        inpSeq = []
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrames, self.seqLen):
            fl_name = vid_name + '/' + 'frame' + str(int(round(i))).zfill(3) + '.jpg'
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label
