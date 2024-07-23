from argparse import ArgumentParser

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import os

import matplotlib.pyplot as plt
from PIL import Image

import random
import glob

def parse_args():
    parser = ArgumentParser("Script for GradCAM visualization")

    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--model_path', type=str, default='')

    parser.add_argument('--num_samples', type=int, default=4) # Num samples for clean and bd each

    parser.add_argument('--clean_path', type=str, default='')
    parser.add_argument('--bd_path', type=str, default='')

    parser.add_argument('--output_dir', type=str, default='')

    parser.add_argument('--random_seed', type=int, default=99)

    return parser.parse_args()


class Denormalize:
    def __init__(self, input_channel, expected_values, variance):
        self.n_channels = input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values) and self.n_channels == len(self.variance)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[channel, :, :] = x[channel, :, :] * self.variance[channel] + self.expected_values[channel]
        return x_clone

if __name__ == '__main__':
    args = parse_args()

    assert args.num_samples % 4 == 0, 'Please choose multiple of 4 for num_samples'

    os.makedirs(args.output_dir, exist_ok=True)

    model = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes)

    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()

    # clean_files = os.listdir(args.clean_path)
    clean_files = glob.glob(os.path.join(args.clean_path, '**'), recursive=True)
    # clean_files = [os.path.join(args.clean_path, f) for f in clean_files]
    clean_files = [f for f in clean_files if not os.path.isdir(f)]
    bd_files = glob.glob(os.path.join(args.bd_path, "**"), recursive=True)
    bd_files = [f for f in bd_files if not os.path.isdir(f)]

    random.seed(args.random_seed)
    clean_files = random.choices(clean_files, k=args.num_samples)
    bd_files = random.choices(bd_files, k=args.num_samples)

    denormalizer = Denormalize(3, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    clean_images = []
    clean_preds = []
    bd_images = []
    bd_preds = []
    print("Predicting clean ...")
    for c in clean_files:
        cimg = Image.open(c).convert('RGB')
        cimg = val_transforms(cimg).unsqueeze(0)
        # cimg = denormalizer(cimg)
        clean_images.append(cimg)
        with torch.no_grad():
            clean_preds.append(model(cimg.to(device)).softmax(-1).argmax(-1).cpu().item())
    print("Predicting backdoor ...")
    for bd in bd_files:
        bdimg = Image.open(bd).convert('RGB')
        bdimg = val_transforms(bdimg).unsqueeze(0)
        # bdimg = denormalizer(bdimg)
        bd_images.append(bdimg)
        with torch.no_grad():
            bd_preds.append(model(bdimg.to(device)).softmax(-1).argmax(-1).cpu().item())

    clean_images = torch.cat(clean_images)
    bd_images = torch.cat(bd_images)

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model, target_layers=target_layers, use_cuda=device=='cuda')

    clean_cams = cam(clean_images)
    bd_cams = cam(bd_images)

    print("Generating clean CAM...")
    nrow = int(args.num_samples / 4)
    fig, ax = plt.subplots(nrow * 2, 4, figsize=(10, 10))
    for row in range(0, nrow * 2, 2):
        for col in range(4):
            clean_img = clean_images[int(row/2*4) + col]
            clean_img = denormalizer(clean_img).permute(1, 2, 0).numpy()
            clean_cam = clean_cams[int(row/2*4) + col]
            cam_img = show_cam_on_image(clean_img, clean_cam, use_rgb=True)
            ax[row][col].set_title('Clean')
            ax[row][col].imshow(clean_img)
            ax[row][col].axis('off')

            ax[row+1][col].set_title(f"{clean_preds[int(row/2*4) + col]}")
            ax[row+1][col].imshow(cam_img)
            ax[row+1][col].axis('off')
    plt.savefig(f"{os.path.join(args.output_dir, 'clean_cam.png')}")

    print("Generating backdoor CAM...")
    nrow = int(args.num_samples / 4)
    fig, ax = plt.subplots(nrow * 2, 4, figsize=(10, 20))
    for row in range(0, nrow * 2, 2):
        for col in range(4):
            bd_img = bd_images[int(row/2*4) + col]
            bd_img = denormalizer(bd_img).permute(1, 2, 0).numpy()
            bd_cam = bd_cams[int(row/2*4) + col]
            cam_img = show_cam_on_image(bd_img, bd_cam, use_rgb=True)

            ax[row][col].set_title('Backdoor')
            ax[row][col].imshow(bd_img)
            ax[row][col].axis('off')

            ax[row+1][col].set_title(f"{bd_preds[int(row/2*4) + col]}")
            ax[row+1][col].imshow(cam_img)
            ax[row+1][col].axis('off')
    plt.savefig(f"{os.path.join(args.output_dir, 'bd_cam.png')}")



    