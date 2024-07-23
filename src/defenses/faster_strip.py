import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
import os

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm


class STRIP_Detection(nn.Module):
    def __init__(self, data, alpha=1.0, beta=1.0, n=100):
        super(STRIP_Detection, self).__init__()
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.n = n

    def _superimpose(self, background, overlay):
        # cv2.addWeighted(background, 1, overlay, 1, 0)
        imgs = self.alpha * background + self.beta * overlay
        imgs = torch.clamp(imgs, 0, 1)
        return imgs

    def forward(self, model, imgs, labels=None):
        # Return Entropy H
        # idx = np.random.randint(0, self.data.shape[0], size=self.n)
        idx = np.random.randint(0, len(self.data), size=self.n)
        H = []
        for img in imgs:
            x = torch.stack([img] * self.n).to(imgs.device)
            for i in range(self.n):
                x_0 = x[i]
                # x_1 = self.data[idx[i]].to(imgs.device)
                x_1, _ = self.data[idx[i]]
                x_1 = x_1.to(imgs.device)
                x_2 = self._superimpose(x_0, x_1)
                x[i] = x_2
            logits = model(x)
            p = F.softmax(logits.detach(), dim=1)
            H_i = - torch.sum(p * torch.log(p), dim=1)
            H.append(H_i.mean().item())
        return torch.tensor(H).detach().cpu()
    
def faster_strip(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(args.input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    
    clean_dataset = torchvision.datasets.ImageFolder(
        os.path.join(args.clean_path, 'test'), 
        transform=transforms
    )
    
    bd_dataset = torchvision.datasets.ImageFolder(
        os.path.join(args.bd_path, 'test'),
        transform=transforms,
        target_transform=lambda x : args.atk_target
    )
    
    clean_test_loader = torch.utils.data.DataLoader(
        clean_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )
    bd_test_loader = torch.utils.data.DataLoader(
        bd_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )
        
    model = timm.create_model(args.model, pretrained=False, num_classes=len(clean_dataset.classes))
    args.checkpoint = os.path.join(args.checkpoint, 'best_model.pth')
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()
    model.to(args.device)
    
    
    strip = STRIP_Detection(clean_dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clean_results = []
    for images, labels in tqdm(clean_test_loader):
        images, labels = images.to(device), labels.to(device)
        batch_rs = strip(model, images, labels)
        clean_results.append(batch_rs.detach().cpu())
    clean_results = torch.cat(clean_results, dim=0)
    print('results shape', clean_results.shape)
    # save resutlts to file
    filename = os.path.join(args.output_dir, 'clean_results.txt')
    torch.save(clean_results, filename)
    print(filename + ' saved!')

    # Run detections on backdoor test set
    bd_results = []
    for images, labels in tqdm(bd_test_loader):
        images, labels = images.to(device), labels.to(device)
        batch_rs = strip(model, images, labels)
        bd_results.append(batch_rs.detach().cpu())

    bd_results = torch.cat(bd_results, dim=0)
    print('results shape', bd_results.shape)
    # save resutlts to file
    filename = os.path.join(args.output_dir, 'bd_results.txt')
    torch.save(bd_results, filename)
    print(filename + ' saved!')
    
    