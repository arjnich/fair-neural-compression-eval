import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision.io import read_image
from torchvision.transforms import Resize
import torchvision.transforms.v2 as transformsv2

LINE_PADDING = 2
CELEBA_HEIGHT = 224
CELEBA_WIDTH = 224


class CelebA(Dataset):

    def __init__(self, img_path, attr_path, transforms, png):
        self.img_path = img_path
        attr_table = open(attr_path).readlines()[LINE_PADDING:]
        self.attr = [row.split() for row in attr_table]
        self.transforms = transforms
        self.png = png
        #self.idx = self.attr[:]

    def __len__(self):
        return len(self.attr)
    
    def __getitem__(self, idx):

        if self.png:
            img =  read_image(os.path.join(self.img_path, self.attr[idx][0].replace("jpg", "png")))
        else:
            img =  read_image(os.path.join(self.img_path, self.attr[idx][0]))
        return self.transforms(img), torch.tensor([(int(val) + 1) /2 for val in self.attr[idx][1:]]) # Optimize this?



def create_dataloaders(img_path, attr_path, batch_size, train_test_ratio, png = False, seed=42):

    tfs = transformsv2.Compose([transformsv2.Resize((CELEBA_HEIGHT, CELEBA_WIDTH)), transformsv2.ToDtype(torch.float32, scale=True)])

    # Create Dataset
    data = CelebA(img_path, attr_path, tfs, png)

    generator = torch.Generator().manual_seed(seed)

    trainset_size = int(len(data) * train_test_ratio)
    testset_size = len(data) - trainset_size

    trainset, testset = random_split(data, [trainset_size, testset_size], generator)

    trainloader = DataLoader(trainset, batch_size)
    testloader = DataLoader(testset, batch_size)


    return trainloader, testloader
