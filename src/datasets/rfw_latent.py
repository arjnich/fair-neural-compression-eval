import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torchvision as tv
import torchvision.transforms.functional as tvf
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append('/home/tianqiu/NeuralCompression/lossy-vae') # Tian
# sys.path.append('/home/rasta/lossy-vae')

from lvae import get_model
from lvae.models.qresvae import zoo


class RFW_raw(Dataset):
    # loads RFW images and convert to tensor

    def __init__(self, img_path, attr_path):

        self.attr = pd.read_csv(attr_path).to_numpy()
        self.img_path = img_path
        self.transforms = tv.transforms.ToTensor()

    def __len__(self):
        return len(self.attr)
    
    def __getitem__(self, idx):
        img_file = os.path.join(self.img_path, self.attr[idx][2], self.attr[idx][1])
        img = Image.open(img_file).convert('RGB') 
        return self.transforms(img), torch.from_numpy(self.attr[idx][3:].astype(np.float32)), self.attr[idx][2].split("/")[0]
    
class RFW_latent(Dataset):
    # loads RFW images and convert to 12-block latents. Not used. 

    def __init__(self, img_path, attr_path, nc_model, device):

        self.attr = pd.read_csv(attr_path).to_numpy()
        self.img_path = img_path
        # self.transforms = tv.transforms.ToTensor()
        self.nc_model = nc_model
        self.device = device
        self.ps_layer = nn.PixelShuffle(2)

    def __len__(self):
        return len(self.attr)
    
    def __getitem__(self, idx):
        img_file = os.path.join(self.img_path, self.attr[idx][2], self.attr[idx][1])
        img = Image.open(img_file).convert('RGB') 
        img = tvf.to_tensor(img).unsqueeze_(0).to(self.device)
        stats_all = self.nc_model.forward_get_latents(img)
        latents = [stats_all[latent_block_index]['z'] for latent_block_index in range(12)]
        output = torch.cat((F.interpolate(latents[0], 4),latents[1], latents[2]), 1)
        output = torch.cat((F.interpolate(output, 8),latents[3], latents[4],latents[5], latents[6]), 1)
        output = torch.cat((F.interpolate(output, 16),latents[7], latents[8],latents[9], latents[10], latents[11]), 1)
        output = self.ps_layer(output).squeeze_(0)

        return output, torch.from_numpy(self.attr[idx][3:].astype(np.float32)), self.attr[idx][2].split("/")[0]


def create_dataloaders(dataset, batch_size, train_test_ratio=0.7, seed=42):
    # Create Dataset
    generator = torch.Generator().manual_seed(seed)
    trainset_size = int(len(dataset) * train_test_ratio)
    validaset_size = int((len(dataset) - trainset_size) * 0.5)
    testset_size = len(dataset) - trainset_size - validaset_size

    trainset, valset, testset = random_split(dataset, [trainset_size, validaset_size, testset_size], generator)

    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size)
    testloader = DataLoader(testset, batch_size)

    return trainloader, valloader, testloader