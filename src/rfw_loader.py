import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision.io import read_image
import torchvision.transforms.v2 as transformsv2

RESNET18_HEIGHT = 224
RESNET18_WIDTH = 224

class RFW(Dataset):

    def __init__(self, img_path, attr_path, transforms, png):

        self.attr = pd.read_csv(attr_path).to_numpy()
        self.img_path = img_path
        self.transforms = transforms
        self.png = png
        
    def __len__(self):
        return len(self.attr)
    
    def __getitem__(self, idx):

        if self.png:
            img =  read_image(os.path.join(self.img_path, self.attr[idx][2], self.attr[idx][1]))
        else:
            img =  read_image(os.path.join(self.img_path, self.attr[idx][2], self.attr[idx][1].replace("png", "jpg")))
        return self.transforms(img), torch.from_numpy(self.attr[idx][3:].astype(np.float32)), self.attr[idx][2].split("/")[0]



def create_dataloaders(
    img_path, 
    attr_path, 
    batch_size, 
    train_test_ratio=0.7, 
    png=True, 
    seed=42
):

    tfs = transformsv2.Compose([
        transformsv2.Resize((RESNET18_HEIGHT, RESNET18_WIDTH)),
        transformsv2.ToTensor()
    ])

    # Create Dataset
    data = RFW(img_path, attr_path, tfs, png)

    generator = torch.Generator().manual_seed(seed)

    trainset_size = int(len(data) * train_test_ratio)
    validaset_size = int((len(data) - trainset_size) * 0.5)
    testset_size = len(data) - trainset_size - validaset_size

    trainset, valset, testset = random_split(data, [trainset_size, validaset_size, testset_size], generator)

    # Create data loaders
    train_loader = DataLoader(trainset, batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size)
    test_loader = DataLoader(testset, batch_size)

    return train_loader, val_loader, test_loader