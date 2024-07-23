import torch
import numpy as np
import cv2

from torchvision import transforms
import torchvision
# from poisoned_datasets import *
import os
import timm 
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm

CIFAR10_DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_DEFAULT_STD = [0.247, 0.243, 0.261]

class Normalize:
    def __init__(self, input_channel, expected_values, variance):
        self.n_channels = input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = (x[:, :, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


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


class STRIP:
    def _superimpose(self, background, overlay):
        output = cv2.addWeighted(background, 1, overlay, 1, 0)
        if len(output.shape) == 2:
            output = np.expand_dims(output, 2)
        return output

    def _get_entropy(self, background, dataset, classifier):
        entropy_sum = [0] * self.n_sample
        x1_add = [0] * self.n_sample
        index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
        for index in range(self.n_sample):
            d = dataset[index_overlay[index]][0].permute(1, 2, 0).numpy() * 255
            d = np.clip(d, 0, 255).astype(np.uint8)
            add_image = self._superimpose(background, d)
            add_image = self.normalize(add_image)
            x1_add[index] = add_image

        py1_add = classifier(torch.stack(x1_add).to(self.device))
        py1_add = torch.sigmoid(py1_add).cpu().numpy()
        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
        return entropy_sum / self.n_sample

    def _get_denormalize(self, args):
        # if args.data_set == "CIFAR10":
        #     denormalizer = Denormalize(args, CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD)
        # elif args.data_set == "MNIST":
        #     denormalizer = Denormalize(args, [0.5], [0.5])
        # elif args.data_set == "GTSRB" or args.data_set == "CELEBATTR" or args.data_set == 'T-IMNET':
        #     denormalizer = None
        # else:
        #     raise Exception("Invalid dataset")
        denormalizer = Denormalize(3, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        return denormalizer

    def _get_normalize(self, args):
        # if args.data_set == "CIFAR10":
        #     normalizer = Normalize(args, CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD)
        # elif args.data_set == "MNIST":
        #     normalizer = Normalize(args, [0.5], [0.5])
        # elif args.data_set == "GTSRB" or args.data_set == "CELEBATTR" or args.data_set == 'T-IMNET':
        #     normalizer = None
        # else:
        #     raise Exception("Invalid dataset")
        normalizer = Normalize(3, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        if normalizer:
            transform = transforms.Compose([transforms.ToTensor(), normalizer])
        else:
            transform = transforms.ToTensor()
        return transform

    def __init__(self, args):
        super().__init__()
        self.n_sample = args.strip_n_sample
        self.normalizer = self._get_normalize(args)
        self.denormalizer = self._get_denormalize(args)
        self.device = args.device

    def normalize(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x

    def denormalize(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x

    def __call__(self, background, dataset, classifier):
        return self._get_entropy(background, dataset, classifier)
    
def strip_one_round(args):
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
    
    # state_dict = torch.load(args.checkpoint)
    # model.load_state_dict(state_dict['model'])
    # model.requires_grad_(False)
    # model.eval()
    # model.to(args.device)

    # # args.batch_size = args.strip_n_test
    # clean_test_loader = torch.utils.data.DataLoader(
    #     clean_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     shuffle=True
    # )
    # bd_test_loader = torch.utils.data.DataLoader(
    #     bd_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     shuffle=True
    # )
    
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
    checkpoint = os.path.join(args.checkpoint, 'best_model.pth')
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()
    model.to(args.device)

    # if args.data_set == 'CIFAR10':
    #     denormalizer = Denormalize(args, CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD)
    # elif args.data_set == 'MNIST':
    #     denormalizer = Denormalize(args, [0.5], [0.5])
    # else:
    #     denormalizer = Denormalize(args, [0, 0, 0], [1, 1, 1])
    denormalizer = Denormalize(3, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    strip_detector = STRIP(args)

    list_entropy_trojan = []
    list_entropy_benign = []

    # print(f'Testing with Clean Data')
    # for index in range(args.n_test):
    #     background, _ = clean_dataset[index]
    #     entropy = strip_detector(background, clean_dataset, model)
    #     list_entropy_benign.append(entropy)
    # if mode == 'attack':
    print(f'Testing with Backdoor Data')
    bd_inputs, _ = next(iter(bd_test_loader))
    # inputs = inputs.to(args.device)
    bd_inputs = bd_inputs.to(args.device)
    bd_inputs = denormalizer(bd_inputs) * 255.0
    bd_inputs = bd_inputs.detach().cpu().numpy()
    bd_inputs = np.clip(bd_inputs, 0, 255).astype(np.uint8).transpose((0, 2, 3, 1))
    for index in tqdm(range(args.strip_n_test)):
        background, _ = bd_dataset[index]
        background = background.permute(1, 2, 0).numpy() * 255
        background = np.clip(background, 0, 255).astype(np.uint8)
        entropy = strip_detector(background, clean_dataset, model)
        list_entropy_trojan.append(entropy)

    print(f'Testing with Clean Data')
    # Testing with clean data
    for index in tqdm(range(args.strip_n_test)):
        background, _ = clean_dataset[index]
        background = background.permute(1, 2, 0).numpy() * 255
        background = np.clip(background, 0, 255).astype(np.uint8)
        entropy = strip_detector(background, clean_dataset, model)
        list_entropy_benign.append(entropy)

    return list_entropy_trojan, list_entropy_benign

def strip(args):
    os.makedirs(args.output_dir, exist_ok=True)
    lists_entropy_trojan = []
    lists_entropy_benign = []
    for test_round in range(args.strip_test_rounds):
        print(f'Test round {test_round}')
        list_entropy_trojan, list_entropy_benign = strip_one_round(args)
        lists_entropy_trojan += list_entropy_trojan
        lists_entropy_benign += list_entropy_benign


    # result_path = os.path.join(args.output_dir, "{}_{}_output.txt".format(args.data_set, args.attack_mode))
    
    result_path = os.path.join(args.output_dir, 'log.txt')
    with open(result_path, "w+") as f:
        for index in range(len(lists_entropy_trojan)):
            if index < len(lists_entropy_trojan) - 1:
                f.write("{} ".format(lists_entropy_trojan[index]))
            else:
                f.write("{}".format(lists_entropy_trojan[index]))

        f.write("\n")

        for index in range(len(lists_entropy_benign)):
            if index < len(lists_entropy_benign) - 1:
                f.write("{} ".format(lists_entropy_benign[index]))
            else:
                f.write("{}".format(lists_entropy_benign[index]))

    min_entropy = min(lists_entropy_trojan + lists_entropy_benign)

    # Determining
    string = ("Min entropy trojan: {}, Detection boundary: {}".format(min_entropy, args.strip_detection_boundary))
    print(string)
    with open(os.path.join(args.output_dir, 'log.txt'), 'a+') as f:
        f.write(string)
        f.flush()
    if min_entropy < args.strip_detection_boundary:
        print("A backdoored model\n")
        with open(os.path.join(args.output_dir, 'log.txt'), 'a+') as f:
            string = 'A backdoored model\n'
            f.write(string)
            f.flush()
    else:
        print("Not a backdoor model\n")
        with open(os.path.join(args.output_dir, 'log.txt'), 'a+') as f:
            string = 'Not a backdoored model\n'
            f.write(string)
            f.flush()


