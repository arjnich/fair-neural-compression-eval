
import torch
_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
import argparse
import pandas as pd
import os
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms


def get_images(meta_data_inf, path):
    print(f'Reading images from {path}')
    image_tensors = []
    for meta_data in tqdm(meta_data_inf, total=len(meta_data_inf)):
        file_path = os.path.join(path, meta_data[2], meta_data[1])
        image = Image.open(file_path).convert('RGB')
        image_tensor = transforms.ToTensor()(image)
        image_tensors.append(image_tensor)
    
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def update_fid_in_batches(fid, images, bsz=256, real=True):
    for i in range(0, len(images), bsz):
        batch = images[i:i + bsz]
        fid.update(batch, real=real)
    return fid
        
def get_fid(fid, clean_images, generated_images, bsz=256):
    update_fid_in_batches(fid, clean_images, bsz=bsz, real=True)
    update_fid_in_batches(fid, generated_images, bsz=bsz, real=False)
    return float(fid.compute())

def main(args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    input_dir = args.input_directory
    clean_image_dir = args.clean_image_dir
    meta_data_inf = pd.read_csv(args.meta_data_path).to_numpy()

    clean_image_tensors = get_images(meta_data_inf, clean_image_dir).to(device).to(torch.uint8)
    
    fid_results = {}
    for model_name in args.models:
        print(f'model_name: {model_name}')
        fid_results[model_name] = {}
        model_dir = f'{input_dir}/{model_name}'
        if model_name == 'jpeg':
            for quality in os.listdir(model_dir):
                print(f'\tquality: {quality}')
                fid = FrechetInceptionDistance().to(device)
                generated_images_path = f'{input_dir}/{model_name}/{quality}'
                generated_image_tensors = get_images(meta_data_inf, generated_images_path).to(device).to(torch.uint8)
                fid_score = get_fid(fid, clean_image_tensors, generated_image_tensors, bsz=1024)
                print(f'\tfid_score: {fid_score}')
                fid_results[model_name][quality] = fid_score
        else:
            for dataset_name in args.datasets:
                print(f'\tdataset_name:{dataset_name}')
                fid_results[model_name][dataset_name] = {}
                dataset_dir = f'{input_dir}/{model_name}/{dataset_name}'
                for quality in os.listdir(dataset_dir):
                    print(f'\t\tquality: {quality}')
                    fid = FrechetInceptionDistance().to(device)
                    generated_images_path = f'{input_dir}/{model_name}/{dataset_name}/{quality}'
                    generated_image_tensors = get_images(meta_data_inf, generated_images_path).to(device).to(torch.uint8)
                    fid_score = get_fid(fid, clean_image_tensors, generated_image_tensors, bsz=1024)
                    print(f'\t\tfid_score: {fid_score}')
                    fid_results[model_name][dataset_name][quality] = fid_score
    print(fid_results)
    with open(f"{args.output_file_directory}/fid.json", "w") as outfile: 
        json.dump(fid_results, outfile, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save compressed images from trained model")
    parser.add_argument("--device", type=int, default=3, help="GPU device used for compression")
    parser.add_argument("--bsz", type=int, default=512, help="batch size for feeding the images to the inception model")
    parser.add_argument("--models", nargs='+', type=str, 
                        default=["qres17m_lmb_64", "mbt2018", "hyperprior", "jpeg", "cheng2020-attn"],
                        help="model names") #bmshj2018-hyperprior, mbt2018, ...
    parser.add_argument("--datasets", nargs='+', type=str,
                        default=['fairface', 'celebA'], 
                        help="datasets used to train the models")
    parser.add_argument("--clean-image-dir", type=str, 
                        default="/media/global_data/fair_neural_compression_data/datasets/RFW/data_64",
                        help="path to the clean RFW images")
    parser.add_argument("--output-file-directory", type=str,
                        default="/media/global_data/fair_neural_compression_data/decoded_rfw/decoded_64x64",
                        help="path to the output fid json file")
    parser.add_argument("--input-directory", type=str,
                        default="/media/global_data/fair_neural_compression_data/decoded_rfw/decoded_64x64",
                        help="path to the compressed images")
    parser.add_argument("--meta-data-path", type=str, 
                        default="/media/global_data/fair_neural_compression_data/datasets/RFW/clean_metadata/numerical_labels.csv",
                        help="path to the meta data csv file")
    args = parser.parse_args()
    main(args)