import torch
import datetime
import os

def write_model(model, path):
    # TODO: Check if a model already exists to avoid overwriting the model

    print("Writing Model")
    torch.save(model.state_dict(), path)
    print("Model Saved")

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
