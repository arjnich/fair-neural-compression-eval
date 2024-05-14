import torch
import torch.nn as nn
import torchvision.models as models

import sys
sys.path.append('/home/tianqiu/NeuralCompression/lossy-vae')
from lvae import get_model
from lvae.models.qresvae import zoo
import lvae.models.common as common
import lvae.models.qresvae.model as qres
from lvae.models.qresvae.model import BottomUpEncoder, TopDownDecoder, HierarchicalVAE

model_name = 'qres17m'
lmb_value = 64
# nc_model, cfg = get_model(model_name, lmb_value, pretrained=True)
nc_model = zoo.qres17m(lmb=64, pretrained=True)

# copy cfg got qres17m
def get_qres17m_model():
    lmb = 64
    cfg = dict()

    enc_nums = [6,6,4,2]
    dec_nums = [1,2,4,5]
    z_dims = [16, 8, 6, 4]

    im_channels = 3
    ch = 72 # 128
    cfg['enc_blocks'] = [
        common.patch_downsample(im_channels, ch*2, rate=4),
        *[qres.MyConvNeXtBlock(ch*2, kernel_size=7) for _ in range(enc_nums[0])], # 16x16
        qres.MyConvNeXtPatchDown(ch*2, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=5) for _ in range(enc_nums[1])], # 8x8
        qres.MyConvNeXtPatchDown(ch*4, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=3) for _ in range(enc_nums[2])], # 4x4
        qres.MyConvNeXtPatchDown(ch*4, ch*4, down_rate=4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=1) for _ in range(enc_nums[3])], # 1x1
    ]
    from torch.nn import Upsample
    cfg['dec_blocks'] = [
        *[qres.QLatentBlockX(ch*4, z_dims[0], kernel_size=1) for _ in range(dec_nums[0])], # 1x1
        Upsample(scale_factor=4),
        *[qres.QLatentBlockX(ch*4, z_dims[1], kernel_size=3) for _ in range(dec_nums[1])], # 4x4
        common.deconv(ch*4, ch*4, kernel_size=3),
        *[qres.QLatentBlockX(ch*4, z_dims[2], kernel_size=5) for _ in range(dec_nums[2])], # 8x8
        common.deconv(ch*4, ch*2),
        *[qres.QLatentBlockX(ch*2, z_dims[3], kernel_size=7) for _ in range(dec_nums[3])], # 16x16
        common.patch_upsample(ch*2, im_channels, rate=4)
    ]
    cfg['out_net'] = qres.MSEOutputNet(mse_lmb=lmb)

    # mean and std computed on CelebA
    cfg['im_shift'] = -0.4356
    cfg['im_scale'] = 3.397893306150187
    cfg['max_stride'] = 64
    return nc_model, cfg

class HierarchicalVAE_ResNet(nn.Module):
    def __init__(self, config: dict, output_dims, use_pretrained):
        super().__init__()
        self.encoder = BottomUpEncoder(blocks=config.pop('enc_blocks'))
        self.decoder = TopDownDecoder(blocks=config.pop('dec_blocks'))
        # self.out_net = config.pop('out_net')
        self.upsampling = nn.Upsample(size=224, mode='nearest')
        self.resnet = models.resnet18(pretrained=use_pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.heads = nn.ModuleDict()
        for head, num_classes in output_dims.items():
            self.heads[head] = nn.Linear(num_features, num_classes)

        self.im_shift = float(config['im_shift'])
        self.im_scale = float(config['im_scale'])
        self.max_stride = config['max_stride']

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

        self._stats_log = dict()
        self._flops_mode = False
        self.compressing = False
    
output_dims = {
    'skin_type': 6,
    'eye_type': 2,
    'nose_type': 2,
    'lip_type': 2,
    'hair_type': 4,
    'hair_color': 5
}

nc_model, cfg = get_qres17m_model()
classifier_model = HierarchicalVAE_ResNet(cfg, output_dims, True)

for param in classifier_model.encoder.parameters():
    param.requires_grad = False