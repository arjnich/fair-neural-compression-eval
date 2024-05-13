import torch
from tqdm import tqdm
import numpy as np
import os
from latent_utils import get_latent
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def initialize_rfw_prediction_results():
    phenotype_list = ['skin_type', 'eye_type', 'nose_type', 'lip_type', 'hair_type', 'hair_color']
    # create empty dictionaries for prediction results
    all_predictions = {'Indian': {head: torch.tensor([]) for head in phenotype_list}, 
                       'Caucasian': {head: torch.tensor([]) for head in phenotype_list}, 
                       'Asian': {head: torch.tensor([]) for head in phenotype_list},  
                       'African': {head: torch.tensor([]) for head in phenotype_list}}
    all_labels = {'Indian': {head: torch.tensor([]) for head in phenotype_list}, 
                  'Caucasian': {head: torch.tensor([]) for head in phenotype_list}, 
                  'Asian': {head: torch.tensor([]) for head in phenotype_list}, 
                  'African': {head: torch.tensor([]) for head in phenotype_list}}

    return all_predictions, all_labels

def save_race_based_predictions_latent(
        nc_model,
        n_keep,
        model, 
        model_name, 
        dataloader, 
        device, 
        prediction_save_dir,
        save_labels=False
    ):
    all_predictions, all_labels = initialize_rfw_prediction_results()
    
    print(f'prediction_save_dir: {prediction_save_dir}')
    os.makedirs(prediction_save_dir, exist_ok=True)
    dataloader = tqdm(dataloader, desc="Getting Predictions", unit="batch")
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            inputs, labels, race = data
            race = np.array(race)

            latents = get_latent(inputs, nc_model, device, n_keep=n_keep)
            # inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(latents)

            for i, (head, predictions) in enumerate(outputs.items()):
                head_preds = predictions.argmax(dim=1).cpu()

                for race_label in all_labels:
                    race_indices = np.array((race == race_label).nonzero()[0])
                    race_predictions = head_preds[race_indices]
                    race_labels = labels[:, i][race_indices]
                
                    all_predictions[race_label][head] = torch.cat((all_predictions[race_label][head], race_predictions.to('cpu')), dim=0)
                    all_labels[race_label][head] = torch.cat((all_labels[race_label][head], race_labels.to('cpu')), dim=0)

    for race_label in all_labels:
        for category in all_labels[race_label]:
            torch.save(all_predictions[race_label][category], f'{prediction_save_dir}/{model_name}_{race_label}_{category}_predictions.pt')
            if save_labels:
                torch.save(all_labels[race_label][category], f'{prediction_save_dir}/{model_name}_{race_label}_{category}_labels.pt')

    return all_predictions, all_labels

def save_race_based_predictions(
        model, 
        model_name, 
        dataloader, 
        device, 
        prediction_save_dir,
        save_labels=False
    ):
    all_predictions, all_labels = initialize_rfw_prediction_results()
    
    print(f'prediction_save_dir: {prediction_save_dir}')
    os.makedirs(prediction_save_dir, exist_ok=True)
    dataloader = tqdm(dataloader, desc="Getting Predictions", unit="batch")
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            inputs, labels, race = data
            race = np.array(race)

            inputs = inputs.to(device)
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

    for race_label in all_labels:
        for category in all_labels[race_label]:
            torch.save(all_predictions[race_label][category], f'{prediction_save_dir}/{model_name}_{race_label}_{category}_predictions.pt')
            if save_labels:
                torch.save(all_labels[race_label][category], f'{prediction_save_dir}/{model_name}_{race_label}_{category}_labels.pt')

    return all_predictions, all_labels


def get_classification_report(races, categories, pred_dir, model_name):
    results = {'acc': {}, 'f1': {}}
    for category in categories:
        results['acc'][category] = {}
        results['f1'][category] = {}
        for race in races:
            results['acc'][category][race] = {}
            results['f1'][category][race] = {}
            preds = torch.load(f'{pred_dir}/{model_name}_{race}_{category}_predictions.pt')
            labels = torch.load(f'{pred_dir}/{model_name}_{race}_{category}_labels.pt')
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted')
            results['acc'][category][race] = acc
            results['f1'][category][race] = f1
    
    return results

def load_predictions(
        model, 
        model_name,
        prediction_save_dir
):
    all_predictions, all_labels = initialize_rfw_prediction_results()
    for race_label in all_labels:
        for category in all_labels[race_label]:
            all_predictions[race_label][category] = torch.load(f'{prediction_save_dir}/{model_name}_{race_label}_{category}_predictions.pt')
    
    return all_predictions, all_labels