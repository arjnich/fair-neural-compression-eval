import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/home/tianqiu/NeuralCompression/lossy-vae')
from lvae.models.qresvae.model import HierarchicalVAE

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
    
class HierarchicalVAE_ResNet(HierarchicalVAE):
    def __init__(self, config: dict, nc_model, output_dims, use_pretrained):
        super().__init__(config)
        # self.encoder = BottomUpEncoder(blocks=config.pop('enc_blocks'))
        # do not train encoder
        self.encoder = nc_model.encoder
        self.decoder = nc_model.decoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        # self.decoder = TopDownDecoder(blocks=config.pop('dec_blocks'))
        # self.out_net = config.pop('out_net')
        self.upsampling = nn.Upsample(size=224, mode='nearest')
        self.resnet = models.resnet18(pretrained=use_pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.heads = nn.ModuleDict()
        for head, num_classes in output_dims.items():
            self.heads[head] = nn.Linear(num_features, num_classes)
    
    def forward(self, im):
        # pass through encoder
        im = im.to(self._dummy.device)
        x = self.preprocess_input(im)
        x_target = self.preprocess_target(im)

        enc_features = self.encoder(x)
        # pass through decoder
        feature, stats_all = self.decoder(enc_features)
        out_loss, x_hat = self.out_net.forward_loss(feature, x_target)
        im_hat = self.process_output(x_hat)

        # pass through classifier
        x = self.upsampling(im_hat)
        # x = self.resnet(x)
        features = self.resnet(x).squeeze((2,3))
        outputs = {}
        for head, head_module in self.heads.items():
            output_logits = head_module(features)
            outputs[head] = F.softmax(output_logits, dim=1)
        return outputs

    def process_output(self, x: torch.Tensor):
        # overrides parent class method
        # im_hat = x.clone().clamp_(min=-1.0, max=1.0).mul_(0.5).add_(0.5)
        # converts the data range back to [0, 1]
        im_hat = x.clamp_(min=-1.0, max=1.0).mul_(0.5).add_(0.5)
        return im_hat