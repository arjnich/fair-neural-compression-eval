from predict import perform_inference
from datasets.rfw_latent import RFW_raw, RFW_latent, create_dataloaders
from train import save_model
from detection_models import MultiHeadResNet, LatentMultiHead_1
from latent_utils import get_latent, train_numerical_rfw_latents

import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('/home/tianqiu/NeuralCompression/lossy-vae')
from lvae import get_model
from lvae.models.qresvae import zoo

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

model = LatentMultiHead_1(output_dims).to(device)



LEARNING_RATE = 0.01
n_keep = 1
experiment_tag='latent_n_keep_1'
writer = SummaryWriter(f'runs/{experiment_tag}')# initialize a writer
model, train_losses, val_losses,  = train_numerical_rfw_latents(nc_model, model, 500, LEARNING_RATE, image_dl_train, image_dl_val, device, n_keep, writer, patience=5)
save_model(model, '../models', 'latent_RFW_numerical_all_labels_n_keep_1_with_val', with_time=False)
