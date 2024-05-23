import sys
import argparse
import torch

from train_utils import create_model, generate_dataloaders, train_numerical_rfw, save_model, save_race_based_predictions, MultiHeadResNet


def parse_args(argv):
    # Common options.
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-mp",
        "--model-path",
        dest="checkpoint_path",
        default="../models/RFW_numerical_all_labels_resnet18_2024-05-07_13-45-45.pth",
        type=str,
        nargs="*",
        required=False,
        help="model path",
    )

    parser.add_argument(
        "-t",
        "--train",
        default=False,
        help="train new model (default: %(default)s)",
    )

    parser.add_argument(
        "--save-model", 
        #action="store_true", 
        default=True, 
        help="Save model to disk (default: %(default)s)"
    )

    parser.add_argument(
        "--data-path",
        default="/media/global_data/fair_neural_compression_data/datasets/RFW/data_64", 
        help="Path to training dataset"
    )

    parser.add_argument(
        "--pred-dir",
        default=".",
        help="Path to directory to write predictions")

    parser.add_argument(
        "--save-model-path", 
        default=".",
        help="Path to save newly trained model"
    )

    parser.add_argument(
        "-e",
        "--epochs",
        default=30,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.01,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--ratio",
        dest="ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio",
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size (default: %(default)s)"
    )

    parser.add_argument(
        "--device", 
        type=int, 
        default=0, 
        help="Device to use (default: %(default)s)"
    )

    args = parser.parse_args(argv)
    return args
    
def main(argv):  # noqa: C901
    args = parse_args(argv)

    train_loader, val_loader, test_loader = generate_dataloaders(args.data_path, args.batch_size, args.ratio)

    model = create_model(args.device)

    if(args.train):
        #train
        print("Training Model")
        trained_model = train_numerical_rfw(model, torch.optim.SGD, args.epochs, args.learning_rate, train_loader, val_loader, args.device, args.pred_dir, 10)
    else:
        #load
        print("Loading Model")
        model = torch.load(args.checkpoint_path)
        model.to(args.device)
    print(args.pred_dir)

    save_race_based_predictions(
            model, 
            test_loader, 
            args.device, 
            args.pred_dir
        )

if __name__ == "__main__":
    main(sys.argv[1:])
