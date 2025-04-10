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
from argparse import ArgumentParser
from PIL import Image
import glob

def parse_args():
    parser = ArgumentParser('Script for training physical backdoor')
    parser.add_argument('--output_dir', type=str, default='./logs')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'momentum', 'adam', 'adamw'])
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--clean_path', type=str, required=True)
    parser.add_argument('--bd_train_path', type=str)
    parser.add_argument('--bd_test_path', type=str)

    parser.add_argument('--aug_method', type=str, default='simple', choices=['simple', 'strong'])
    parser.add_argument('--input_size', type=int, default=224)

    parser.add_argument('--atk_target', type=int, default=0)
    parser.add_argument('--poisoning_rate', type=float, default=0.05)

    parser.add_argument('--random_seed', type=int, default=99)

    
    return parser.parse_args()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, label=0, transform=None, target_transform=None):
        self.data_root = data_root
        # self.file_list = os.listdir(data_root)
        self.file_list = glob.glob(os.path.join(self.data_root, '**'), recursive=True)
        self.file_list = [f for f in self.file_list if not os.path.isdir(f)]
        self.transform = transform
        self.label_transform = target_transform
        self.label = label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file = self.file_list[index]
        img = Image.open(file).convert('RGB')
        img = self.transform(img)
        
        if self.label_transform is not None:
            self.label = self.label_transform(self.label)
        
        return img, self.label

def write_to_log(log_file, str):
    log_file.write(str)
    log_file.flush()

if __name__ == '__main__':
    args = parse_args()
    seed = args.random_seed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    log_file = open(os.path.join(output_dir, 'log.txt'), 'w')
    str_args = str(vars(args))
    tmp_str = str_args[1:-1].split(', ')
    tmp_str = '\n'.join(tmp_str)
    write_to_log(log_file, tmp_str + '\n')

    if args.aug_method == 'simple':
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop((args.input_size, args.input_size)),
            torchvision.transforms.RandomRotation(10),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    elif args.aug_method == 'strong':
        train_transforms = transforms_imagenet_train(
            img_size=args.input_size,
            hflip=0.5,
            vflip=0.,
            interpolation='bicubic',
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            use_prefetcher=False,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            re_num_splits=0,
            separate=False,
            # to_tensor=True,
            # normalize=True,
        )
    val_resize = 256 if args.input_size > 32 else 32
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(val_resize),
        torchvision.transforms.CenterCrop(args.input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    clean_path = args.clean_path

    train_clean = torchvision.datasets.ImageFolder(
        os.path.join(clean_path, 'train'), transform=train_transforms
    )
    _, class_count = torch.unique(torch.tensor(train_clean.targets), return_counts=True)

    if 'cat' in args.bd_train_path:
        atk_target = train_clean.class_to_idx['cat']
    elif 'dog' in args.bd_train_path:
        atk_target = train_clean.class_to_idx['dog']

    train_bd = ImageDataset(
        args.bd_train_path, transform=train_transforms, target_transform=lambda x : atk_target
    )
    num_poison = int(args.poisoning_rate * class_count[atk_target])
    if num_poison < len(train_bd):
        np.random.seed(args.random_seed)
        indices = np.random.choice(len(train_bd), num_poison, replace=False)
        train_bd = torch.utils.data.Subset(train_bd, indices)

    train_dataset = torch.utils.data.ConcatDataset([train_clean, train_bd])

    string = f"\nPreset Poisoning Rate: {args.poisoning_rate} \nActual Poisoning Rate: {len(train_bd)/class_count[atk_target].item():.4f}\n\n"
    write_to_log(log_file, string)

    val_clean = torchvision.datasets.ImageFolder(
        os.path.join(clean_path, 'test'), val_transforms
    )
    val_bd = ImageDataset(
        args.bd_test_path, transform=val_transforms, target_transform=lambda x : atk_target
    )
    

    string = f"Train clean dataset: {len(train_clean)} \nTrain backdoor dataset: {len(train_bd)} \nValidation clean dataset: {len(val_clean)} \nValidation backdoor dataset: {len(val_bd)}"
    write_to_log(log_file, string)
    write_to_log(log_file, f"\nClass idx: {train_clean.class_to_idx}\n")
    write_to_log(log_file, f"Attack target: {args.atk_target}\n\n")

    # model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes)
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=len(train_clean.classes))
    model.to(device)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    write_to_log(log_file, f"Model: {args.model}\n{model}\n")
    write_to_log(log_file, f"Optimizer: {optimizer}\n")
    write_to_log(log_file, f"Scheduler: {scheduler}\n")
    write_to_log(log_file, f"Criterion: {criterion}\n")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    val_clean_loader = torch.utils.data.DataLoader(
        val_clean,
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_bd_loader = torch.utils.data.DataLoader(
        val_bd,
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    max_clean_acc = 0
    max_poison_asr = 0
    clean_acc_tol = 0.1
    for ep in range(args.epochs):
        print(f'Epoch {ep+1}')
        model.train()
        print('Training...')
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        running_loss = None
        for i, (img, target) in train_pbar:
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            if running_loss is None:
                running_loss = loss
            else:
                running_loss = 0.99 * running_loss + 0.01 * loss.detach().item()
            train_pbar.set_description(f'Training Loss: {running_loss:.4f}')
            
            
        scheduler.step(ep)

        print('Validating...')
        model.eval()
        preds = []
        targets = []
        for i, (img, target) in tqdm(enumerate(val_clean_loader), total=len(val_clean_loader)):
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.no_grad():
                pred = model(img)

            pred = pred.cpu().detach()
            target = target.cpu().detach()

            preds.append(pred)
            targets.append(target)

        preds = torch.cat(preds)
        targets = torch.cat(targets)

        print("Validating backdoor....")
        model.eval()
        preds_bd = []
        targets_bd = []
        for i, (img, target) in tqdm(enumerate(val_bd_loader), total=len(val_bd_loader)):
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.no_grad():
                pred = model(img)

            pred = pred.cpu().detach()
            target = target.cpu().detach()

            preds_bd.append(pred)
            targets_bd.append(target)

        preds_bd = torch.cat(preds_bd)
        targets_bd = torch.cat(targets_bd)

        topk = (1,) if args.num_classes <= 5 else (1, 5)
        clean_acc = accuracy(preds, targets, topk=topk)
        asr = accuracy(preds_bd, targets_bd, topk=topk)

        clean_acc = np.mean(clean_acc)
        asr = np.mean(asr)

        if clean_acc > max_clean_acc \
                or (clean_acc > max_clean_acc-clean_acc_tol and  asr > max_poison_asr):
            max_clean_acc = clean_acc
            max_poison_asr = asr
            torch.save(
                model.state_dict(), os.path.join(output_dir, 'best_model.pth')
            )
        torch.save(
                model.state_dict(), os.path.join(output_dir, 'checkpoint.pth')
        )
        print(f"Clean Accuracy: {clean_acc:.4f}, ASR: {asr:.4f}")
        print(f"Max Clean Acc: {max_clean_acc:.4f}, Max ASR: {max_poison_asr:.4f}")
        
        
        string = f"Epoch {ep+1:^3d} || Clean Accuracy: {clean_acc:^7.4f}, ASR: {asr:^7.4f} || Max Clean Acc: {max_clean_acc:^7.4f}, Max ASR: {max_poison_asr:^7.4f}\n"
        write_to_log(log_file, string)

    log_file.close()