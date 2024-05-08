from train import train, write_model
from predict import perform_inference
from celeba_loader import create_dataloaders



EPOCHS = 5
LEARNING_RATE = 0.01
RATIO = 0.8
BATCH_SIZE = 32
DEVICE = 1

if __name__ == "__main__":

    trainloader, testloader = create_dataloaders("./data/celebA/img_align_celeba/", "./data/celebA/attr/list_attr_celeba.txt", BATCH_SIZE, RATIO)

    model = train(EPOCHS, LEARNING_RATE, trainloader, DEVICE)

    perform_inference(model, testloader, DEVICE)
    write_model(model, "./models/restnet18")
    
    eval_list = [1, 2, 4, 8, 16, 32]
    
    for rate in eval_list:
        print(f"Evaluating q_{rate}")

        _, inferenceloader = create_dataloaders(f"./data/celebA/qres17m/q_{rate}/", "./data/celebA/attr/list_attr_celeba.txt", BATCH_SIZE, RATIO, True)
        perform_inference(model, inferenceloader, DEVICE)
