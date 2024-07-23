import torch
import torchvision
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import random
from timm.data.transforms_factory import transforms_imagenet_train
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from argparse import ArgumentParser
import cv2
import scipy.stats as sstats
from PIL import Image
import json
import glob
import shutil
from sklearn.model_selection import train_test_split
import ImageReward as RM 

def parse_args():
    parser = ArgumentParser('Script for training physical backdoor')
    parser.add_argument('--output_dir', type=str, default='./selected_outputs')
    parser.add_argument('--model', type=str, default='inception_v3')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.set_defaults(pretrained=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--bs', type=int, default=64)

    parser.add_argument('--generated_path', type=str)
    parser.add_argument('--real_path', type=str)

    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--metric', nargs='*', default=['cossim', 'mse'])

    parser.add_argument('--select_criteria', type=str, choices=['topk', 'threshold'])
    parser.add_argument('--threshold', default='mean')

    parser.add_argument('--image_reward', action='store_true')
    parser.add_argument('--prompt', type=str)
    
    return parser.parse_args()

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transforms):
        self.data_root = data_root
        # self.file_list = os.listdir(data_root)
        # self.file_list = [f for f in self.file_list if 'combined' not in f]
        self.file_list = []
        # for root, dir, name in os.walk(data_root):
        #     print(dir)
        #     print(name)
        #     for d in dir:
        #         print(d)
        #         for n in name:
        #             if 'combined' in n:
        #                 continue
        #             self.file_list.append(os.path.join(d, n))
        # for path, subdirs, files in os.walk(self.data_root):
        #     for name in files:
        #         self.file_list.append(os.path.join(path, name))
        self.file_list = glob.glob(os.path.join(self.data_root, '**'), recursive=True)
        self.file_list = [f for f in self.file_list if not os.path.isdir(f)]
        self.file_list = [f for f in self.file_list if 'combined' not in f]
        # print(self.file_list)
        # print(self.file_list)
        # print(glob.glob(os.path.join(self.data_root, '**'), recursive=True))
        self.transform = transforms

    def __len__(self):
        return len(self.file_list)
    
    def _getid(self, fname):
        return fname.split(os.path.sep)[-1]

    def __getitem__(self, index):
        file = self.file_list[index]
        
        # img = Image.open(os.path.join(self.data_root, file)).convert('RGB')
        img = Image.open(file).convert('RGB')
        img = self.transform(img)

        # id = self._getid(file)
        
        return img, file

if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not args.image_reward:
        if args.model == 'dinov2':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        else:
            model = timm.create_model(args.model, pretrained=args.pretrained)

        model.eval().to(device)

        if 'inception' in args.model:
            val_t = torchvision.transforms.Compose([
                torchvision.transforms.Resize((args.input_size, args.input_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)
            ])
        else:
            val_t = torchvision.transforms.Compose([
                torchvision.transforms.Resize((args.input_size, args.input_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])

        real_path = args.real_path
        real_dataset = ImageDataset(real_path, val_t)


        real_loader = torch.utils.data.DataLoader(
            real_dataset,
            batch_size=args.bs,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers
        )

        real_feats = []
        print('Getting real features...')
        for i, (img, _) in tqdm(enumerate(real_loader), total=len(real_loader)):
            img = img.to(device, non_blocking=True)
            with torch.no_grad():
                feat = model(img)
            real_feats.append(feat)

        real_feats = torch.cat(real_feats).mean(0).cpu().numpy()

        generated_path = args.generated_path
        generated_dataset = ImageDataset(generated_path, val_t)
        generated_loader = torch.utils.data.DataLoader(
            generated_dataset,
            batch_size=args.bs,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers
        )

        mse_dict = {}
        cossim_dict = {}

        print('Getting generated features...')
        for i, (img, ids) in tqdm(enumerate(generated_loader), total=len(generated_loader)):
            img = img.to(device, non_blocking=True)
            with torch.no_grad():
                feat = model(img).squeeze().cpu().numpy()

            if 'mse' in args.metric:
                mse = torch.from_numpy(feat - real_feats).norm(p=2, dim=-1)
                for j, id in enumerate(ids):
                    mse_dict[id] = mse[j].item()
            
            if 'cossim' in args.metric:
                cossim = torch.nn.functional.cosine_similarity(torch.from_numpy(feat), torch.from_numpy(real_feats))
                for j, id in enumerate(ids):
                    cossim_dict[id] = cossim[j].item()

        # Sorting
        cossim_dict = {k: v for k, v in reversed(sorted(cossim_dict.items(), key=lambda item: item[1]))}
        mse_dict = {k: v for k, v in reversed(sorted(mse_dict.items(), key=lambda item: item[1]))}

        os.makedirs(args.output_dir, exist_ok=True)
        cossim_file = os.path.join(args.output_dir, 'cossim.txt')
        mse_file = os.path.join(args.output_dir, 'mse.txt')

        with open(cossim_file, 'w') as f:
            json.dump(cossim_dict, f)
        
        with open(mse_file, 'w') as f:
            json.dump(mse_dict, f)

        cossim_list = [v for k, v in cossim_dict.items()]
        mse_list = [v for k, v in mse_dict.items()]

        cossim_mean, cossim_median, cossim_std = np.mean(cossim_list), np.median(cossim_list), np.std(cossim_list)
        mse_mean, mse_median, mse_std = np.mean(mse_list), np.median(mse_list), np.std(mse_list)

        if args.select_criteria == 'topk':
            k = int(args.threshold)
            cossim_selected = dict(list(cossim_dict.items())[:k])
            mse_selected = dict(list(mse_dict.items())[-k:])

        elif args.select_criteria == 'threshold':
            if args.threshold == 'mean':
                cossim_selected = {k:v for k, v in cossim_dict.items() if v > cossim_mean}
                mse_selected = {k:v for k, v in mse_dict.items() if v < mse_mean}
            elif args.threshold == 'mean_std':
                cossim_selected = {k:v for k, v in cossim_dict.items() if v > cossim_mean + cossim_std}
                mse_selected = {k:v for k, v in mse_dict.items() if v < mse_mean - mse_std}
            elif args.threshold == 'median':
                cossim_selected = {k:v for k, v in cossim_dict.items() if v > cossim_median}
                mse_selected = {k:v for k, v in mse_dict.items() if v < mse_median}
            else:
                cossim_selected = {k:v for k, v in cossim_dict.items() if v > args.threshold}
                mse_selected = {k:v for k, v in mse_dict.items() if v < args.threshold}
        

        print(f"Selected generated samples from Cosine Similarity: {len(cossim_selected)}")
        print(f"Selected generated samples from MSE: {len(mse_selected)}")

        selected_output = os.path.join(args.output_dir, 'selected_outputs')

        cossim_selected_path = os.path.join(selected_output, 'cossim_selected')
        mse_selected_path = os.path.join(selected_output, 'mse_selected')

        cossim_train_idx, cossim_test_idx = train_test_split(np.arange(len(cossim_selected)), test_size=0.2, random_state=99)
        mse_train_idx, mse_test_idx = train_test_split(np.arange(len(mse_selected)), test_size=0.2, random_state=99)

        os.makedirs(cossim_selected_path, exist_ok=True)
        os.makedirs(mse_selected_path, exist_ok=True)
        os.makedirs(os.path.join(cossim_selected_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(cossim_selected_path, 'test'), exist_ok=True)
        os.makedirs(os.path.join(mse_selected_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(mse_selected_path, 'test'), exist_ok=True)
        
        for i, (fname, cossim_val) in enumerate(cossim_selected.items()):
            # src = os.path.join(args.generated_path, fname)
            src = fname
            if i in cossim_train_idx:
                shutil.copy(src, os.path.join(cossim_selected_path, 'train'))
            else:
                shutil.copy(src, os.path.join(cossim_selected_path, 'test'))

        for i, (fname, mse_val) in enumerate(mse_selected.items()):
            # src = os.path.join(args.generated_path, fname)
            src = fname
            if i in mse_train_idx:
                shutil.copy(src, os.path.join(mse_selected_path, 'train'))
            else:
                shutil.copy(src, os.path.join(mse_selected_path, 'test'))

    elif args.image_reward:
        # model = RM.load("ImageReward-v1.0")
        model = RM.load(
            "/vinserver_user/jason/diffusion_bd/ImageReward/checkpoint/ImageReward.pt", 
                med_config='/vinserver_user/jason/diffusion_bd/ImageReward/checkpoint/med_config.json'
        )

        file_list = glob.glob(os.path.join(args.generated_path, '**'), recursive=True)
        file_list = [f for f in file_list if not os.path.isdir(f)]
        file_list = [f for f in file_list if 'combined' not in f]

        prompt = args.prompt

        with torch.no_grad():
            ranking, rewards = model.inference_rank(prompt, file_list)
        
        imrew_dict = {}
        for file, r in zip(file_list, rewards):
            imrew_dict[file] = r
        
        imrew_dict = {k: v for k, v in reversed(sorted(imrew_dict.items(), key=lambda item: item[1]))}

        imrew_mean = np.mean(rewards)
        imrew_median = np.median(rewards)
        imrew_std = np.std(rewards)

        if args.select_criteria == 'topk':
            k = int(args.threshold)
            imrew_selected = dict(list(imrew_dict.items())[:k])

        elif args.select_criteria == 'threshold':
            if args.threshold == 'mean':
                imrew_selected = {k:v for k, v in imrew_dict.items() if v > imrew_mean}
            elif args.threshold == 'mean_std':
                imrew_selected = {k:v for k, v in imrew_dict.items() if v > imrew_mean + imrew_std}
            elif args.threshold == 'median':
                imrew_selected = {k:v for k, v in imrew_dict.items() if v > imrew_median}
            else:
                imrew_selected = {k:v for k, v in imrew_dict.items() if v > args.threshold}

        train_idx, test_idx = train_test_split(np.arange(len(imrew_selected)), test_size=0.2, random_state=99)

        selected_path = args.output_dir
        os.makedirs(os.path.join(selected_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(selected_path, 'test'), exist_ok=True)

        imrew_file = os.path.join(args.output_dir, 'image_reward.txt')

        with open(imrew_file, 'w') as f:
            json.dump(imrew_dict, f)

        for i, (fname, cossim_val) in enumerate(imrew_selected.items()):
            src = fname
            if i in train_idx:
                shutil.copy(src, os.path.join(selected_path, 'train'))
            else:
                shutil.copy(src, os.path.join(selected_path, 'test'))