from predict import perform_inference
from datasets.rfw_latent import RFW_raw, RFW_latent, create_dataloaders
from train import save_model
from detection_models import MultiHeadResNet, LatentMultiHead_1, LatentMultiHead_3, LatentMultiHead_6, LatentMultiHead_9
from latent_utils import get_latent, train_numerical_rfw_latents

import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse

import sys
sys.path.append('/home/tianqiu/NeuralCompression/lossy-vae')
from lvae import get_model
from lvae.models.qresvae import zoo

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_keep',  type=int,  default=12)
    parser.add_argument('--use_pretrained', action='store_true', default=False)
    cfg = parser.parse_args()
    assert cfg.n_keep in [1,3,6,9,12], 'n_keep value not accepted'
    return cfg

def main():
    cfg = parse_args()
    BATCH_SIZE = 1024
    device = 'cuda'
    # load pre-trained qres model
    model_name = 'qres17m'
    lmb_value = 64
    nc_model = get_model(model_name, lmb_value, pretrained=True).to(device) # weights are downloaded automatically


    RFW_IMAGES_DIR =  "/media/global_data/fair_neural_compression_data/datasets/RFW/data_64"
    RFW_LABELS_DIR = "/media/global_data/fair_neural_compression_data/datasets/RFW/clean_metadata/numerical_labels.csv"
    image_ds = RFW_raw(RFW_IMAGES_DIR, RFW_LABELS_DIR)
    image_dl_train, image_dl_val, image_dl_test = create_dataloaders(image_ds, BATCH_SIZE)



    output_dims = {
        'skin_type': 6,
        'eye_type': 2,
        'nose_type': 2,
        'lip_type': 2,
        'hair_type': 4,
        'hair_color': 5
    }
    n_keep = cfg.n_keep
    experiment_tag=f'latent_n_keep_{n_keep}'
    if n_keep == 12:
        model = MultiHeadResNet(output_dims, use_pretrained=cfg.use_pretrained).to(device)
        if cfg.use_pretrained:
            experiment_tag = experiment_tag + '_use_pretrained'
        else:
            experiment_tag = experiment_tag + '_no_pretrained'
    elif n_keep == 3:
        model = LatentMultiHead_3(output_dims).to(device)
    elif n_keep == 6:
        model = LatentMultiHead_6(output_dims).to(device)
    elif n_keep == 9:
        model = LatentMultiHead_9(output_dims, use_pretrained=cfg.use_pretrained).to(device)
        if cfg.use_pretrained:
            experiment_tag = experiment_tag + '_use_pretrained'
        else:
            experiment_tag = experiment_tag + '_no_pretrained'
    elif n_keep ==1:
        model = LatentMultiHead_1(output_dims).to(device)



    LEARNING_RATE = 0.01
    print(experiment_tag)
    writer = SummaryWriter(f'runs/{experiment_tag}')# initialize a writer
    model, ending_epoch, train_losses, val_losses,  = train_numerical_rfw_latents(nc_model, model, 500, LEARNING_RATE, image_dl_train, image_dl_val, device, n_keep, writer, patience=5)
    save_model(model, '../models', f'latent_RFW_{experiment_tag}_epoch_{ending_epoch}', with_time=False)

if __name__ == "__main__":
    main()