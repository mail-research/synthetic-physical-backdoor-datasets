import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt

import os

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def total_variation_loss(img, weight=1):
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :]-img[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
    tv_w = torch.pow(img[:, :, :, 1:]-img[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
    return weight*(tv_h+tv_w)/(c*h*w)


class CognitiveDistillation(nn.Module):
    def __init__(self, lr=0.1, p=1, gamma=0.01, beta=1.0, num_steps=100, mask_channel=1, norm_only=False):
        super(CognitiveDistillation, self).__init__()
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.num_steps = num_steps
        self.l1 = torch.nn.L1Loss(reduction='none')
        self.lr = lr
        self.mask_channel = mask_channel
        self.get_features = True
        self._EPSILON = 1.e-6
        self.norm_only = norm_only

    def get_raw_mask(self, mask):
        mask = (torch.tanh(mask) + 1) / 2
        return mask

    def forward(self, model, images, labels=None):
        model.eval()
        b, c, h, w = images.shape
        mask = torch.ones(b, self.mask_channel, h, w).to(images.device)
        mask_param = nn.Parameter(mask)
        optimizerR = torch.optim.Adam([mask_param], lr=self.lr, betas=(0.1, 0.1))
        if self.get_features:
            # features, logits = model(images, return_features=True)
            features = model.forward_features(images)
            # features = model.global_pool(feat)
            logits = model.forward_head(features)
        else:
            logits = model(images).detach()
        for step in range(self.num_steps):
            optimizerR.zero_grad()
            mask = self.get_raw_mask(mask_param).to(images.device)
            x_adv = images * mask + (1-mask) * torch.rand(b, c, 1, 1).to(images.device)
            if self.get_features:
                # adv_fe, adv_logits = model(x_adv, return_features=True)
                adv_fe = model.forward_features(x_adv)
                adv_logits = model.forward_head(adv_fe)
                # print(adv_fe.shape, features.shape)
                if len(adv_fe.shape) == 4:
                    loss = self.l1(adv_fe, features.detach()).mean(dim=[1, 2, 3])
                else:
                    loss = self.l1(adv_fe, features.detach()).mean(dim=1)
            else:
                adv_logits = model(x_adv)
                loss = self.l1(adv_logits, logits).mean(dim=1)
            norm = torch.norm(mask, p=self.p, dim=[1, 2, 3])
            norm = norm * self.gamma
            loss_total = loss + norm + self.beta * total_variation_loss(mask)
            loss_total.mean().backward()
            optimizerR.step()
        mask = self.get_raw_mask(mask_param).detach().cpu()
        if self.norm_only:
            return torch.norm(mask, p=1, dim=[1, 2, 3])
        return mask.detach()
    

class Denormalize:
    def __init__(self, input_channel, expected_values, variance):
        self.n_channels = input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = x[:, :, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone

def cognitive_distillation(args):
    # if args.data_set == 'CELEBATTR':
    #     args.nb_classes = 8
    #     args.input_size = 64
    #     args.input_channel = 3
    # elif args.data_set == 'T-IMNET':
    #     args.nb_classes = 200
    #     args.input_size = 64
    #     args.input_channel = 3
    # elif args.data_set == 'GTSRB':
    #     args.nb_classes = 43
    #     args.input_size = 32
    #     args.input_channel = 3
    # elif args.data_set == 'CIFAR10':
    #     args.nb_classes = 10
    #     args.input_size = 32
    #     args.input_channel = 3
    # elif args.data_set == 'MNIST':
    #     args.nb_classes = 10
    #     args.input_size = 28
    #     args.input_channel = 1

    # if args.data_set == 'CIFAR10' or args.data_set == 'GTSRB' or args.data_set == 'CELEBATTR':
    #     denormalizer = Denormalize(args, CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD)
    # elif args.data_set == 'MNIST':
    #     denormalizer = Denormalize(args, [0.5], [0.5])
    # else:
    #     denormalizer = Denormalize(args, [0, 0, 0], [1, 1, 1])

    # clean_dataset, _ = build_eval_dataset(0, args) # Attack portion should be 0 for for clean dataset
    # bd_dataset, _ = build_eval_dataset(1.0, args)

    # if args.model == 'myresnet18':
    #     from models.resnet import ResNet18
    #     model = ResNet18(num_classes=args.nb_classes)
    # elif args.model == 'mypreactresnet18':
    #     from models.preact_resnet import PreActResNet18
    #     model = PreActResNet18(num_classes=args.nb_classes)
    # elif args.model == 'mymnistnet':
    #     from models.mnist_net import MNISTNet
    #     model = MNISTNet()

    # mode = 'attack' if args.attack_mode == 'all2all'or args.attack_mode == 'all2one' else 'clean'
    # args.checkpoint = os.path.join(args.checkpoint, 'best_model.pth')
    
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

    detector = CognitiveDistillation(args.cd_lr, args.cd_p, args.cd_gamma, args.cd_beta, num_steps=args.cd_num_steps, 
                                mask_channel=args.cd_mask_channel, norm_only=args.cd_norm_only)
    detector.to(args.device)

    clean_masks = []
    clean_l1_norms = []
    print('Processing Clean.....')
    for idx, (input, target) in enumerate(clean_test_loader):
        input = input.to(args.device)
        target = target.to(args.device)
        mask = detector(model, input, target)
        # print(mask.shape)
        clean_l1_norm = torch.norm(mask, p=1)
        clean_masks.append(mask.cpu())
        clean_l1_norms.append(clean_l1_norm.cpu().item())
        print(f'Step: {idx+1} / {len(clean_test_loader)}')

    clean_masks = torch.cat(clean_masks).squeeze()
    clean_l1_norms = [torch.norm(a, p=1) for a in clean_masks]

    
    bd_masks = []
    bd_l1_norms = []
    print('Processing Backdoor.....')
    for idx, (input, target) in enumerate(bd_test_loader):
        input = input.to(args.device)
        target = target.to(args.device)
        mask = detector(model, input, target)
        bd_l1_norm = torch.norm(mask, p=1)

        bd_masks.append(mask.cpu())
        bd_l1_norms.append(bd_l1_norm.cpu().item())
        print(f'Step: {idx+1} / {len(bd_test_loader)}')

    # args.output_path = "{}_{}_{}_".format(args.output_dir, args.data_set, args.attack_mode)
    os.makedirs(args.output_dir, exist_ok=True)
    
    bd_masks = torch.cat(bd_masks).squeeze()
    bd_l1_norms = [torch.norm(a, p=1) for a in bd_masks]

    l1_norm_txt = os.path.join(args.output_dir, 'l1_norm.txt')

    with open(l1_norm_txt, "w") as file:
        cln = ' '.join(str(a.item()) for a in clean_l1_norms)
        bln = ' '.join(str(a.item()) for a in bd_l1_norms)
        file.write(cln)
        file.write('\n')
        file.write(bln)
        file.close()

    # plt.hist(clean_l1_norms, alpha=0.5, label='Clean', bins=100, color='blue')
    # plt.hist(bd_l1_norms, alpha=0.5, label='Backdoor', bins=100, color='red')
    plt.hist(clean_l1_norms, alpha=0.5, label='Clean', color='blue')
    plt.hist(bd_l1_norms, alpha=0.5, label='Backdoor', color='red')
    if 'imagenet' in args.clean_path:
        plt.title('ImageNet-5')
    elif 'afhq' in args.clean_path:
        plt.title('AFHQ')
    plt.legend()
    plt.xlabel('L1 norm of the mask')
    plt.ylabel('Number of samples')

    l1_norm_path = os.path.join(args.output_dir, f'l1_norms.png')
    plt.savefig(l1_norm_path)
    plt.savefig(os.path.join(args.output_dir, f'l1_norms.pdf'))

    clean_mask_path = os.path.join(args.output_dir, f'clean_masks')
    bd_mask_path = os.path.join(args.output_dir, f'bd_masks')

    clean_img_path = os.path.join(args.output_dir, f'clean_img')
    bd_img_path = os.path.join(args.output_dir, f'bd_img')

    os.makedirs(clean_mask_path, exist_ok=True)
    os.makedirs(bd_mask_path, exist_ok=True)
    os.makedirs(clean_img_path, exist_ok=True)
    os.makedirs(bd_img_path, exist_ok=True)

    n_images = 5
    denormalizer = Denormalize(3, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    for idx, (clean_mask, bd_mask) in enumerate(zip(clean_masks, bd_masks)):
        if idx == n_images:
            break
        print(clean_mask.shape)
        plt.imsave(os.path.join(clean_mask_path, f'clean_mask_{idx+1}.png'), clean_mask.cpu().numpy(), cmap='gray')
        plt.imsave(os.path.join(bd_mask_path, f'bd_mask_{idx+1}.png'), bd_mask.cpu().numpy(), cmap='gray')

        ci = clean_dataset[idx][0].permute(1, 2, 0).cpu()
        bi = bd_dataset[idx][0].permute(1, 2, 0).cpu()
        ci = denormalizer(ci).numpy()
        bi =  denormalizer(bi).numpy()

        plt.imsave(os.path.join(clean_img_path, f'clean_img_{idx+1}.png'), ci)
        plt.imsave(os.path.join(bd_img_path, f'bd_img_{idx+1}.png'), bi)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(ci)
        # ax[1].imshow(clean_mask)
    
    print(f'Output saved at {args.output_dir} !')
    

