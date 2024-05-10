import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadResNet(nn.Module):
    def __init__(self, output_dims, use_pretrained):
        super(MultiHeadResNet, self).__init__()
        self.dim_reducing_layer = nn.Conv2d(in_channels=19, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.upsampling_layer = nn.Upsample(size=224, mode='nearest')
        self.resnet = models.resnet18(pretrained=use_pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.heads = nn.ModuleDict()
        for head, num_classes in output_dims.items():
            self.heads[head] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.dim_reducing_layer(x)
        x = self.upsampling_layer(x)
        features = self.resnet(x).squeeze()
        outputs = {}
        for head, head_module in self.heads.items():
            output_logits = head_module(features)
            outputs[head] = F.softmax(output_logits, dim=1)
        return outputs


class LatentMultiHead_1(nn.Module):
    def __init__(self, output_dims):
        super(LatentMultiHead_1, self).__init__()
        self.num_features = 512
        self.fc = nn.Linear(in_features=16, out_features=self.num_features)
        self.heads = nn.ModuleDict()
        for head, num_classes in output_dims.items():
            self.heads[head] = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        x = x.squeeze((2, 3))
        features = self.fc(x)
        outputs = {}
        for head, head_module in self.heads.items():
            output_logits = head_module(features)
            outputs[head] = F.softmax(output_logits, dim=1)
        return outputs