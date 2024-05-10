import torch
import torch.nn as nn
from celeba_loader import *
from datetime import datetime
from tqdm import tqdm


def train(epochs, lr, trainloader, device, rfw=False):

    num_workers = 4

    # Define the GPU to access
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

    print("Using Device ", device)

    # Pull resnet model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    # Modify the fc layer
    output_dim = next(iter(trainloader))[1].shape[1]
    model.fc = nn.Linear(model.fc.in_features, output_dim)

    ## Add sigmoid layer TODO: Check if this is correct
    model = nn.Sequential(model, nn.Sigmoid())

    # Send the model to the GPU
    model.to(device)
    criterion = nn.BCELoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as t:
            for i, data in enumerate(t):

                if rfw:
                    inputs,labels, race = data
                else:
                    inputs,labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                #print(outputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optim.step()

                running_loss += loss.item()
                t.set_postfix(loss=running_loss / (i + 1))

    return model

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

if __name__ == "__main__":
    model = train(1, 0.01, 32, "./data/celebA/q1/", "./data/celebA/attr/list_attr_celeba.txt")
    write_model(model, "./models/resnet")