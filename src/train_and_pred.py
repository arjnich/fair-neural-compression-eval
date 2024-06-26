import sys
import argparse
import torch
import os

from train_utils import create_model, generate_dataloaders, train_numerical_rfw, save_model, save_race_based_predictions, MultiHeadResNet, DEFAULT_OUTPUT_DIMS


def parse_args(argv):
    # Common options.
    parser = argparse.ArgumentParser(description="Example training script.")

    # --model-dir is the directory of the trained checkpoints. 
    parser.add_argument(
        "-md",
        "--model-dir",
        dest="checkpoint_dir",
        default="/media/global_data/fair_neural_compression_data/predictions/hyperprior/celebA/clean",
        type=str,
        required=False,
        help="dir of all the pretrained model path",
    )

    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        default=False,
        help="train new model (default: %(default)s)",
    )

    # parser.add_argument(
    #     "--save-model", 
    #     #action="store_true", 
    #     default=True, 
    #     help="Save model to disk (default: %(default)s)"
    # )

    parser.add_argument(
        "--data-path",
        default="/media/global_data/fair_neural_compression_data/datasets/RFW/data_64", 
        help="Path to training dataset"
    )

    parser.add_argument(
        "--pred-dir",
        default=".",
        help="Path to directory to write predictions")

    # parser.add_argument(
    #     "--save-model-path", 
    #     default=".",
    #     help="Path to save newly trained model"
    # )

    parser.add_argument(
        "-e",
        "--epochs",
        default=20,
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
    print(args)
    # check pretrained model dir exsits. 
    if args.train==False:
        assert os.path.isdir(args.checkpoint_dir), 'checkpoint dir does not exist'
 
    train_loader, val_loader, test_loader = generate_dataloaders(args.data_path, args.batch_size, args.ratio)

    ###model = create_model(args.device)
    models = []
    if(args.train):
        #train
        attributes = DEFAULT_OUTPUT_DIMS.keys()
        print(attributes)
        for attr in attributes:
            output_dims = {attr: DEFAULT_OUTPUT_DIMS[attr]}
            model = create_model(
                args.device, 
                output_dims
                )

            print(f"Training {attr} Model")

            model,_,_ = train_numerical_rfw(model, torch.optim.SGD, args.epochs, args.learning_rate, train_loader, val_loader, args.device, args.pred_dir, attr, 5)
            models.append(model)
    else:
        attributes = DEFAULT_OUTPUT_DIMS.keys()
        print(attributes)
        for attr in attributes:
            output_dims = {attr: DEFAULT_OUTPUT_DIMS[attr]}
            model = create_model(
                args.device, 
                output_dims
                )
            print(f"Loading {attr} Model")
            checkpoint_path = os.path.join(args.checkpoint_dir, attr+'_best.pth')
            model = torch.load(checkpoint_path)
            model.to(args.device)
            models.append(model)  
    print(args.pred_dir)
 

    save_race_based_predictions(
            models, 
            test_loader, 
            args.device, 
            args.pred_dir,
            attributes
            )

if __name__ == "__main__":
    main(sys.argv[1:])
