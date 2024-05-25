import os
import datetime
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from rfw_loader import create_dataloaders


class MultiHeadResNet(nn.Module):
    def __init__(self, output_dims):
        super(MultiHeadResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.heads = nn.ModuleDict()
        for head, num_classes in output_dims.items():
            self.heads[head] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.resnet(x).squeeze()
        outputs = {}
        for head, head_module in self.heads.items():
            output_logits = head_module(features)
            outputs[head] = F.softmax(output_logits, dim=1)
        return outputs

def create_model(device):
    output_dims = {
    'skin_type': 6,
    'eye_type': 2,
    'nose_type': 2,
    'lip_type': 2,
    'hair_type': 4,
    'hair_color': 5
    }

    model = MultiHeadResNet(output_dims).to(device)

    return model

def save_model(model, dir_path, model_name, with_time=False):
    if with_time:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{dir_path}/{model_name}_{current_time}.pth"
    else:
        filename = f"{dir_path}/{model_name}.pth"
    print("Writing Model")
    torch.save(model, filename)
    print("Model Saved")

def save_race_based_predictions(
        model,  
        dataloader, 
        device, 
        prediction_save_dir
    ):
    all_predictions = {'Indian': {head: torch.tensor([]) for head in model.heads.keys()}, 
                       'Caucasian': {head: torch.tensor([]) for head in model.heads.keys()}, 
                       'Asian': {head: torch.tensor([]) for head in model.heads.keys()},  
                       'African': {head: torch.tensor([]) for head in model.heads.keys()}}
    all_labels = {'Indian': {head: torch.tensor([]) for head in model.heads.keys()}, 
                  'Caucasian': {head: torch.tensor([]) for head in model.heads.keys()}, 
                  'Asian': {head: torch.tensor([]) for head in model.heads.keys()}, 
                  'African': {head: torch.tensor([]) for head in model.heads.keys()}}
    
    print(f'prediction_save_dir: {prediction_save_dir}')
    dataloader = tqdm(dataloader, desc="Getting Predictions", unit="batch")
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            inputs, labels, race = data
            race = np.array(race)

            inputs = inputs.to(torch.float).to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            for i, (head, predictions) in enumerate(outputs.items()):
                head_preds = predictions.argmax(dim=1).cpu()

                for race_label in all_labels:
                    race_indices = np.array((race == race_label).nonzero()[0])
                    race_predictions = head_preds[race_indices]
                    race_labels = labels[:, i][race_indices]
                
                    all_predictions[race_label][head] = torch.cat((all_predictions[race_label][head], race_predictions.to('cpu')), dim=0)
                    all_labels[race_label][head] = torch.cat((all_labels[race_label][head], race_labels.to('cpu')), dim=0)

    with open(prediction_save_dir + '/predictions.pkl', 'wb+') as f:
        pickle.dump(all_predictions, f)
    #with open(prediction_save_dir + '/labels.pkl', 'wb+') as f:
    #    pickle.dump(all_labels, f)


    return all_predictions, all_labels


def generate_dataloaders(image_path, batch_size, ratio):
    RFW_LABELS_DIR = "/media/global_data/fair_neural_compression_data/datasets/RFW/clean_metadata/numerical_labels.csv"
    return create_dataloaders(
        image_path, 
        RFW_LABELS_DIR, 
        batch_size, 
        ratio
        )

def train_numerical_rfw(
        model, 
        optimizer,
        num_epochs, 
        lr, 
        train_loader, 
        valid_loader,
        device,
        save_dir,
        patience=5  # Number of epochs to wait for improvement in validation loss before stopping
    ):
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = optimizer(model.parameters(), lr=lr)
    
    train_losses = []
    valid_losses = []
    
    best_valid_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Training") as pbar:
            for inputs, targets, races in train_loader:
                inputs, targets = inputs.to(device).float(), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = 0
                for i, head in enumerate(outputs):
                    loss += criterion(outputs[head], targets[:, i].to(torch.int64))
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)
                pbar.update(1)
            avg_train_loss = running_train_loss / len(train_loader.dataset)
            pbar.set_postfix(train_loss=avg_train_loss)
        
        print(f'Epoch {epoch + 1} train loss : {avg_train_loss}')
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        running_valid_loss = 0.0
        with torch.no_grad():
            with tqdm(total=len(valid_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Validation") as pbar:
                for inputs, targets, races in valid_loader:
                    inputs, targets = inputs.to(device).float(), targets.to(device)
                    outputs = model(inputs)
                    loss = 0
                    for i, head in enumerate(outputs):
                        loss += criterion(outputs[head], targets[:, i].to(torch.int64))
                    running_valid_loss += loss.item() * inputs.size(0)
                    pbar.update(1)
                avg_valid_loss = running_valid_loss / len(valid_loader.dataset)  # Compute average validation loss
                pbar.set_postfix(valid_loss=avg_valid_loss)
        print(f'Epoch {epoch + 1} valid loss : {avg_valid_loss}')
        valid_losses.append(avg_valid_loss)
        
        # Check for early stopping
        if avg_valid_loss < best_valid_loss:
            print(f'Found better model. Best loss: {avg_valid_loss}')
            best_valid_loss = avg_valid_loss
            no_improvement_count = 0
            print(f'Saving best model to {save_dir}/best.pth')
            torch.save(model, f'{save_dir}/best.pth')
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model, train_losses, valid_losses