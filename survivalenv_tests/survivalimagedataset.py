from os import listdir
from os.path import isfile, join

import random

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import cv2


class SurvivalImageDataset(Dataset):
    def __init__(self, a=0, b=None, directory='images'):
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        files.sort()
        self.custom_dataset = []
        for f in files[a:b]:
            path = join(directory, f)
            image = cv2.resize(cv2.imread(path), (128, 128))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)/255
            self.custom_dataset.append( (image, 0) )

    def __len__(self):
        return len(self.custom_dataset)
    
    def __getitem__(self, idx):
        image = self.custom_dataset[idx][0]
        label = torch.tensor(self.custom_dataset[idx][1])
        return (image, label)

class SurvivalDataset(Dataset):
    def __init__(self, a=0, b=None, directory=None, limit=None, fname=None):
        if directory is not None and fname is None:
            files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
            files.sort()
        elif directory is None and fname is not None:
            files = [fname]
        else:
            raise Exception("You need to provide *EITHER* a directory or a fname.")
        self.custom_dataset = []
        for path in files[a:b]:
            fd = open(path, 'br')
            print(path)
            read = np.load(fd, allow_pickle=True)
            if limit is not None:
                for limit_strategy in range(3):
                    if limit_strategy == 0:
                        self.custom_dataset.append(read['data'][:limit])
                    elif limit_strategy == 1:
                        self.custom_dataset.append(read['data'][-limit:])
                    else:
                        offset = random.randint(0, len(read['data'])-limit)
                        self.custom_dataset.append(read['data'][offset:offset+limit])
            else:
                self.custom_dataset.append(read['data'])
            fd.close()
        print('ds len:', len(self.custom_dataset))
        random.shuffle(self.custom_dataset)
    def __len__(self):
        return len(self.custom_dataset)
    
    def __getitem__(self, idx):
        episode = self.custom_dataset[idx]
        return episode

class SurvivalDatasetEpisodeImages(Dataset):
    def __init__(self, data):
        self.custom_dataset = []
        for step in data:
            image = step['observation']['view']
            image = cv2.resize(image, (160, 160))
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)/255
            self.custom_dataset.append( (image, 0) )
    def __len__(self):
        return len(self.custom_dataset)
    
    def __getitem__(self, idx):
        image = self.custom_dataset[idx][0]
        label = torch.tensor(self.custom_dataset[idx][1])
        return (image, label)


if __name__ == "__main__":
    pass
