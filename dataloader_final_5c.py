from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import numpy as np
import os
import random


class EmoData_5c(Dataset):
    def __init__(self, phase, paths):
        self.paths = paths
        self.phase = phase
        self.input_transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=(256, 256)),
            T.Normalize((0.5), (0.5))
        ])
        self.input_transform2 = T.ToTensor()
        self.emotions = {'Neutral': 0,
                         'Anger': 1,
                         'Frustration': 2,
                         # 'Excited': 5,
                         'Sadness': 3,
                         'Happiness': 4}
        # self.labels = np.load('./IEMOCAP_data/emo_5label.npy', allow_pickle=True)
        self.ids = np.load('./IEMOCAP_data/data_5c/ids_5c_' + phase + '.npy', allow_pickle=True)
        self.masks = np.load('./IEMOCAP_data/data_5c/masks_5c_' + phase + '.npy', allow_pickle=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        spec = self.input_transform(np.load(self.paths[item], allow_pickle=True))[0]
        label = self.emotions[self.paths[item].split('_')[-1].split('.')[0]]
        ids = self.ids[int(self.paths[item].split('/')[-1].split('_')[0])]
        masks = self.masks[int(self.paths[item].split('/')[-1].split('_')[0])]
        return spec, ids, masks, label


def data_loader_5c(phase, batch_size=64, n_workers=5, **path):
    dataset = EmoData_5c(phase, **path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers)
    return dataloader


def dataset_create_5c(spec_dir='./IEMOCAP_data/data_5c/db_spec_5c', phase='train', batch_size=64, rate=0.8):
    spec_dir = spec_dir + '_' + phase
    paths = [spec_dir + '/' + i for i in os.listdir(spec_dir)]
    if phase == 'train':
        random.shuffle(paths)
    # paths_train = paths[:int(rate*len(paths))]
    # paths_test = paths[int(rate*len(paths)):]
    # dl_train = data_loader_5c(paths=paths_train, batch_size=batch_size)
    dl = data_loader_5c(phase=phase, paths=paths, batch_size=batch_size)
    return dl

