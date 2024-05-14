import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

from datasets.rfw_latent import RFW_raw, RFW_latent, create_dataloaders
from latent_utils import train_numerical_rfw_pseudo_decoder
from train import save_model

import sys
sys.path.append('/home/tianqiu/NeuralCompression/lossy-vae')
from lvae import get_model
from lvae.models.qresvae import zoo
import lvae.models.common as common
import lvae.models.qresvae.model as qres
from lvae.models.qresvae.model import BottomUpEncoder, TopDownDecoder, HierarchicalVAE

device = 'cuda'
model_name = 'qres17m'
lmb_value = 64
# nc_model, cfg = get_model(model_name, lmb_value, pretrained=True)
pretrained_nc_model = zoo.qres17m(lmb=64, pretrained=True)

# copy cfg got qres17m
def get_qres17m_cfg():
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
    return cfg

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

output_dims = {
    'skin_type': 6,
    'eye_type': 2,
    'nose_type': 2,
    'lip_type': 2,
    'hair_type': 4,
    'hair_color': 5
}

def main():
    cfg = get_qres17m_cfg()
    classifier_model = HierarchicalVAE_ResNet(cfg, pretrained_nc_model, output_dims, True).to(device)
    # classifier_model.encoder = pretrained_nc_model.encoder
    # classifier_model.decoder = pretrained_nc_model.decoder
    # print(classifier_model)

    BATCH_SIZE = 512

    RFW_IMAGES_DIR =  "/media/global_data/fair_neural_compression_data/datasets/RFW/data_64"
    RFW_LABELS_DIR = "/media/global_data/fair_neural_compression_data/datasets/RFW/clean_metadata/numerical_labels.csv"

    image_ds = RFW_raw(RFW_IMAGES_DIR, RFW_LABELS_DIR)
    image_dl_train, image_dl_val, image_dl_test = create_dataloaders(image_ds, BATCH_SIZE)

    # print(classifier_model.encoder.requires_grad_)
    # print(classifier_model.decoder.requires_grad_)


    # for inputs, targets, races in tqdm(image_dl_train):
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     # print(inputs.shape)
    #     outputs = classifier_model(inputs)
    #     # print(len(features))

    #     # break
    num_epochs = 100
    lr = 1e-4
    experiment_tag = 'pseudo_decoder_debug'
    writer = SummaryWriter(f'runs/{experiment_tag}')# initialize a writer

    model, ending_epoch, train_losses, val_losses = train_numerical_rfw_pseudo_decoder(
            classifier_model, 
            num_epochs, 
            lr, 
            image_dl_train, 
            image_dl_val,
            device, 
            writer,
            patience=5  # Number of epochs to wait for improvement in validation loss before stopping
    )
    save_model(model, '../models', f'latent_RFW_{experiment_tag}_epoch_{ending_epoch}', with_time=False)


if __name__ == "__main__":
    main()