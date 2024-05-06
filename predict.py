import torch
import torch.nn as nn


def perform_inference(model, testloader, device):

    # Define the GPU to access
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()

    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_accuracies = torch.zeros(40).to(device)
        for i, data in enumerate(testloader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            predictions = outputs >= 0.5

            test_accuracies += torch.sum(predictions == labels, axis = 0)


            test_loss += criterion(outputs, labels).item()

            

    print("Test Loss ", test_loss/len(testloader))
    print("Test Accuracies", test_accuracies/len(testloader.dataset))
    
if __name__ == "__main__":
    
    try:
        model = torch.load("./models/restnet18")
        perform_inference(model, testloader)
    except:
        print("No Model Found")