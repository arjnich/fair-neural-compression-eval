import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadResNet(nn.Module):
    # input dimension: [BS, 19, 16, 16]
    # Num. params = 11M
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
    # input dimension: [BS, 16, 1, 1]
    # No. params = 8K
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
        # output dimension here is [BS, 512]
        outputs = {}
        for head, head_module in self.heads.items():
            output_logits = head_module(features)
            outputs[head] = F.softmax(output_logits, dim=1)
        return outputs
    
class LatentMultiHead_3(nn.Module):
    # input dimension: [BS, 32, 4, 4]
    # No. params = 0.15 M
    def __init__(self, output_dims):
        super(LatentMultiHead_3, self).__init__()
        self.num_features = 512
        self.conv = nn.Conv2d(in_channels=32, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pooling = nn.AvgPool2d(4)

        self.heads = nn.ModuleDict()
        for head, num_classes in output_dims.items():
            self.heads[head] = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        # output dimension here is [BS, 512, 1, 1]
        features = x.squeeze((2,3))
        outputs = {}
        for head, head_module in self.heads.items():
            output_logits = head_module(features)
            outputs[head] = F.softmax(output_logits, dim=1)
        return outputs

class LatentMultiHead_6(nn.Module):
    # input dimension: [BS, 50, 8, 8]
    # No. params = 0.23 M
    def __init__(self, output_dims):
        super(LatentMultiHead_6, self).__init__()
        self.num_features = 512
        self.conv = nn.Conv2d(in_channels=50, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pooling = nn.AvgPool2d(8)

        self.heads = nn.ModuleDict()
        for head, num_classes in output_dims.items():
            self.heads[head] = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        # output dimension here is [BS, 512, 1, 1]
        features = x.squeeze((2,3))
        outputs = {}
        for head, head_module in self.heads.items():
            output_logits = head_module(features)
            outputs[head] = F.softmax(output_logits, dim=1)
        return outputs

class LatentMultiHead_9(nn.Module):
    # input dimension: [BS, 16, 32, 32]
    # No. params = 0.23 M
    def __init__(self, output_dims, use_pretrained):
        super(LatentMultiHead_9, self).__init__()
        self.dim_reducing_layer = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)
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