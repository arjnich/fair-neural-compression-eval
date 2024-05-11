from predict import perform_inference
from datasets.rfw_latent import RFW_raw, RFW_latent, create_dataloaders
from train import train, write_model, save_model
from detection_models import MultiHeadResNet, LatentMultiHead_1
from latent_utils import get_latent

import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

import sys
sys.path.append('/home/tianqiu/NeuralCompression/lossy-vae')
from lvae import get_model
from lvae.models.qresvae import zoo

RATIO = 0.8
BATCH_SIZE = 1024
device = 'cuda'
# load pre-trained qres model
model_name = 'qres17m'
lmb_value = 64
nc_model = get_model(model_name, lmb_value, pretrained=True).to(device) # weights are downloaded automatically


RFW_IMAGES_DIR =  "/media/global_data/fair_neural_compression_data/datasets/RFW/data_64"
RFW_LABELS_DIR = "/media/global_data/fair_neural_compression_data/datasets/RFW/clean_metadata/numerical_labels.csv"
image_ds = RFW_raw(RFW_IMAGES_DIR, RFW_LABELS_DIR)
image_dl_train, image_dl_test = create_dataloaders(image_ds, BATCH_SIZE, RATIO)
# latent_ds = RFW_latent(RFW_IMAGES_DIR, RFW_LABELS_DIR, nc_model, device)
# latent_dl_train, latent_dl_test = create_dataloaders(latent_ds, BATCH_SIZE, RATIO)


output_dims = {
    'skin_type': 6,
    'eye_type': 2,
    'nose_type': 2,
    'lip_type': 2,
    'hair_type': 4,
    'hair_color': 5
}

model = LatentMultiHead_1(output_dims).to(device)

def train_numerical_rfw(num_epochs, lr, train_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for inputs, targets, races in train_loader:
                # print(inputs.shape)
                latents = get_latent(inputs, nc_model, device, n_keep=1)
                # inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(latents)
                loss = 0
                for i, head in enumerate(outputs):
                    loss += criterion(outputs[head], targets[:, i].to(torch.int64))
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                avg_loss = running_loss / ((pbar.n + 1) * len(latents))  # Compute average loss
                pbar.set_postfix(loss=avg_loss)
                pbar.update(1)
    return model


LEARNING_RATE = 0.01
model = train_numerical_rfw(500, LEARNING_RATE, image_dl_train, device)
save_model(model, '../models', 'latent_RFW_numerical_all_labels_n_keep_1')
