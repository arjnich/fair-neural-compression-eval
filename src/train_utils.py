import os
import datetime
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from multi_head_resnet import MultiHeadResNet
from rfw_loader import create_dataloaders

DEFAULT_OUTPUT_DIMS = {
    'skin_type': 6,
    'eye_type': 2,
    'nose_type': 2,
    'lip_type': 2,
    'hair_type': 4,
    'hair_color': 5
    }

ATTRIBUTE_INDECIES = {
    'skin_type': 0,
    'lip_type': 1,
    'nose_type': 2,
    'eye_type': 3,
    'hair_type': 4,
    'hair_color': 5
}

def create_model(device, output_dims):
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
        models,  
        dataloader, 
        device, 
        prediction_save_dir,
        attributes
    ):
    all_predictions = {'Indian': {attr: torch.tensor([]) for attr in attributes}, 
                       'Caucasian': {attr: torch.tensor([]) for attr in attributes}, 
                       'Asian': {attr: torch.tensor([]) for attr in attributes},  
                       'African': {attr: torch.tensor([]) for attr in attributes}}
    all_labels = {'Indian': {attr: torch.tensor([]) for attr in attributes}, 
                  'Caucasian': {attr: torch.tensor([]) for attr in attributes}, 
                  'Asian': {attr: torch.tensor([]) for attr in attributes}, 
                  'African': {attr: torch.tensor([]) for attr in attributes}}
    
    print(f'prediction_save_dir: {prediction_save_dir}')
    dataloader = tqdm(dataloader, desc="Getting Predictions", unit="batch")
    with torch.no_grad():
        for j, model in enumerate(models):
            model.eval()
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
                        race_labels = labels[:, ATTRIBUTE_INDECIES[head]][race_indices]
                    
                        all_predictions[race_label][head] = torch.cat((all_predictions[race_label][head], race_predictions.to('cpu')), dim=0)
                        all_labels[race_label][head] = torch.cat((all_labels[race_label][head], race_labels.to('cpu')), dim=0)

    with open(prediction_save_dir + '/sep_predictions.pkl', 'wb+') as f:
        pickle.dump(all_predictions, f)
    with open(prediction_save_dir + '/sep_labels.pkl', 'wb+') as f:
        pickle.dump(all_labels, f)


    return all_predictions, all_labels

def generate_dataloaders(image_path, batch_size, ratio):
    # RFW_LABELS_DIR = "/media/global_data/fair_neural_compression_data/datasets/RFW/clean_metadata/numerical_labels.csv"
    RFW_LABELS_DIR = "/media/global_data/fair_neural_compression_data/datasets/RFW/clean_metadata/numerical_labels_sorted.csv"
    return create_dataloaders(
        image_path, 
        RFW_LABELS_DIR, 
        batch_size, 
        train_test_ratio=ratio,
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
        attr,
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
                    loss += criterion(outputs[head], targets[:, ATTRIBUTE_INDECIES[head]].to(torch.int64))
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
                        # print(f'head: {head} - race:{np.unique(np.array(races))},  unique hairs: {targets[:, ATTRIBUTE_INDECIES[head]].unique()}')
                        loss += criterion(outputs[head], targets[:, ATTRIBUTE_INDECIES[head]].to(torch.int64))
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
            print(f'Saving best model to {save_dir}/{attr}_best.pth')
            torch.save(model, f'{save_dir}/{attr}_best.pth')
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    model = torch.load(f'{save_dir}/{attr}_best.pth')
    return model, train_losses, valid_losses