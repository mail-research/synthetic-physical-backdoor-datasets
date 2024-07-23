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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser('Script for evaluating physical backdoor')
    parser.add_argument('--output_dir', type=str, default='./logs')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--model_weight', type=str)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=3)

    parser.add_argument('--clean_path', type=str)
    parser.add_argument('--bd_path', type=str)

    parser.add_argument('--real_clean_path', type=str)
    parser.add_argument('--real_bd_path', type=str)

    parser.add_argument('--input_size', type=int, default=224)

    parser.add_argument('--atk_target', type=int, default=0)
    parser.add_argument('--label', type=int, default=0)
    
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
    
def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    for i, (img, target) in tqdm(enumerate(loader), total=len(loader)):
        img = img.to(device, non_blocking=True)
        with torch.no_grad():
            pred = model(img)

        pred = pred.cpu().detach()

        preds.append(pred)
        targets.append(target)

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return preds, targets

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, label, transforms, label_transform=None):
        self.data_root = data_root
        self.file_list = os.listdir(data_root)
        self.transform = transforms
        self.label_transform = label_transform
        self.label = label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file = self.file_list[index]
        img = Image.open(os.path.join(self.data_root, file)).convert('RGB')
        img = self.transform(img)
        
        if self.label_transform is not None:
            self.label = self.label_transform(self.label)
        
        return img, self.label

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

    log_file = open(os.path.join(output_dir, 'eval_log.txt'), 'w')
    str_args = str(vars(args))
    tmp_str = str_args[1:-1].split(', ')
    tmp_str = '\n'.join(tmp_str)
    write_to_log(log_file, tmp_str + '\n')
    
    val_resize = 256 if args.input_size > 32 else 32
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(val_resize),
        torchvision.transforms.CenterCrop(args.input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    clean_path = os.path.join(args.clean_path, 'test')
    bd_path = os.path.join(args.bd_path, 'test')

    val_clean = torchvision.datasets.ImageFolder(
        clean_path, val_transforms
    )
    val_bd = torchvision.datasets.ImageFolder(
        bd_path, val_transforms, target_transform=lambda x : args.atk_target
    )

    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=len(val_clean.classes))
    state_dict = torch.load(args.model_weight)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    write_to_log(log_file, f"Model: {args.model}\n{model}\n")

    string = f"Validation clean dataset: {len(val_clean)} \nValidation Generated backdoor dataset: {len(val_bd)}"
    write_to_log(log_file, string)
    write_to_log(log_file, f"\nClass idx: {val_clean.class_to_idx}\n")
    write_to_log(log_file, f"Attack target: {args.atk_target}\n\n")
    

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

    print('Validating dataset...')
    preds, targets = evaluate(model, val_clean_loader, device)

    print("Validating backdoor generated dataset....")
    preds_bd, targets_bd = evaluate(model, val_bd_loader, device)
    clean_acc = np.mean(accuracy(preds, targets))
    asr = np.mean(accuracy(preds_bd, targets_bd))
    
    string = f"\nOriginal Validation Dataset\nClean Accuracy: {clean_acc:.4f}, ASR: {asr:.4f}\n\n"
    write_to_log(log_file, string)
    
    string = f"Clean confusion matrix: \n{confusion_matrix(targets, preds.softmax(-1).argmax(-1))}\n\n"
    write_to_log(log_file, string)
    
    string = f"Poison confusion matrix: \n{confusion_matrix(targets_bd, preds_bd.softmax(-1).argmax(-1))}\n\n"
    write_to_log(log_file, string)
    
    select = torch.nonzero(preds.softmax(-1).argmax(-1) != targets)

    if len(select) > 0:
        select = random.choices(select, k=5 if len(select) > 5 else len(select))
        print(len(select))
        # selected_preds = preds[select]
        # selected_targets = targets[select]
        denormalizer = Denormalize(3, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        for i, idx in enumerate(select):
            # print(idx)
            clean_img, _ = val_clean[idx]
            plt.imshow(denormalizer(clean_img).permute(1, 2, 0))
            plt.title(f"GT: {targets[idx].item()}, PRED: {preds[idx].argmax(-1).item()}")
            # Image.fromarray(clean_img.permute(1, 2, 0).numpy()).save(os.path.join(args.output_dir, f"img_{i}.png"))
            plt.savefig(os.path.join(args.output_dir, f"img_{i}.png"))
    
    if args.real_clean_path and args.real_bd_path:
        real_clean_path = args.real_clean_path
        real_bd_path = args.real_bd_path

        # Convert to imagefolder
        # real_val_clean = ImageDataset(
        #     real_clean_path, label=args.label, transforms=val_transforms
        # )
        # real_val_bd = ImageDataset(
        #     real_bd_path, label=args.label, transforms=val_transforms, label_transform=lambda x : args.atk_target
        # )

        real_val_clean = torchvision.datasets.ImageFolder(
            real_clean_path, val_transforms
        )

        real_val_bd = torchvision.datasets.ImageFolder(
            real_bd_path, val_transforms, target_transform=lambda x : args.atk_target
        )

        real_val_clean_loader = torch.utils.data.DataLoader(
            real_val_clean,
            batch_size=args.bs,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        real_val_bd_loader = torch.utils.data.DataLoader(
            real_val_bd,
            batch_size=args.bs,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        string = f"Validation clean dataset: {len(real_val_clean)} \nValidation Generated backdoor dataset: {len(real_val_bd)}"
        write_to_log(log_file, string)
        write_to_log(log_file, f"\nClass idx: {val_clean.class_to_idx}\n")
        write_to_log(log_file, f"Attack target: {args.atk_target}\n\n")

        print('Validating real dataset...')
        real_preds, real_targets = evaluate(model, real_val_clean_loader, device)

        print("Validating real backdoor dataset....")
        real_preds_bd, real_targets_bd = evaluate(model, real_val_bd_loader, device)
        real_clean_acc = np.mean(accuracy(real_preds, real_targets))
        real_asr = np.mean(accuracy(real_preds_bd, real_targets_bd))
        
        string = f"\nReal Validation Dataset\nClean Accuracy: {real_clean_acc:.4f}, ASR: {real_asr:.4f}\n\n"
        write_to_log(log_file, string)
        
        print("Original Validation Dataset")
        print(f"CA: {clean_acc:.2f}, ASR: {asr:.2f}")
        print("Real Validation Dataset")
        print(f"Real CA: {real_clean_acc:.2f}, Real ASR: {real_asr:.2f}")
        
        string = f"Clean Real confusion matrix: \n{confusion_matrix(real_targets, real_preds.softmax(-1).argmax(-1), labels=[0, 1, 2, 3, 4])}\n\n"
        write_to_log(log_file, string)
        
        string = f"Poison Real confusion matrix: \n{confusion_matrix(real_targets_bd, real_preds_bd.softmax(-1).argmax(-1), labels=[0, 1, 2, 3, 4])}\n\n"
        write_to_log(log_file, string)
        print(f"Output stored at {log_file}")