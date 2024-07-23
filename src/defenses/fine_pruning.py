import torch
import torchvision
import os
import torch.nn as nn
import copy
import torch.nn.functional as F
from tqdm import tqdm

import timm
from timm.data.constants import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN

def create_targets_bd(targets, args):
    if args.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * args.attack_label
    elif args.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % args.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(args.attack_mode))
    return bd_targets.to(args.device)

def convert(mask):
    mask_len = len(mask)
    converted_mask = torch.ones(mask_len, mask_len, 3, 3, dtype=bool)
    for i in range(mask_len):
        try:
            converted_mask[i] = mask[i]
        except:
            print(i)
            input()
    return converted_mask

def create_bd(netG, netM, inputs, targets, args):
    bd_targets = create_targets_bd(targets, args)
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)

    masks_output = netM.threshold(netM(inputs))
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs, bd_targets


def eval(model, clean_loader, bd_loader, args):
    model.eval()

    total_sample = 0
    total_sample_bd = 0
    total_correct_clean = 0
    total_correct_bd = 0

    # for batch_idx, (clean, bd) in enumerate(zip(clean_loader, bd_loader)):
    #     clean_inputs, clean_targets = clean
    #     bd_inputs, bd_targets = bd
    #     clean_inputs = clean_inputs.to(args.device, non_blocking=True)
    #     clean_targets = clean_targets.to(args.device, non_blocking=True)
    #     bd_inputs = bd_inputs.to(args.device, non_blocking=True)
    #     bd_targets = bd_targets.to(args.device, non_blocking=True)

    #     bs = clean_inputs.shape[0]

    #     total_sample += bs

    #     preds_clean = model(clean_inputs)
    #     correct_clean = torch.sum(preds_clean.argmax(1) == clean_targets)
    #     total_correct_clean += correct_clean
    #     acc_clean = total_correct_clean * 100. / total_sample
    
    #     preds_bd = model(bd_inputs)
    #     correct_bd = torch.sum(preds_bd.argmax(1) == bd_targets)
    #     total_correct_bd += correct_bd
    #     acc_bd = total_correct_bd * 100. / total_sample

    for idx, clean in enumerate(clean_loader):
        clean_inputs, clean_targets = clean
        clean_inputs = clean_inputs.to(args.device, non_blocking=True)
        clean_targets = clean_targets.to(args.device, non_blocking=True)
        bs = clean_inputs.shape[0]
        total_sample += bs
        preds_clean = model(clean_inputs)
        correct_clean = torch.sum(preds_clean.argmax(1) == clean_targets)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100. / total_sample

    for idx, bd in enumerate(bd_loader):
        bd_inputs, bd_targets = bd
        bd_inputs = bd_inputs.to(args.device, non_blocking=True)
        bd_targets = bd_targets.to(args.device, non_blocking=True)
        bs = bd_inputs.shape[0]
        total_sample_bd += bs
        preds_bd = model(bd_inputs)
        correct_bd = torch.sum(preds_bd.argmax(1) == bd_targets)
        total_correct_bd += correct_bd
        acc_bd = total_correct_bd * 100. / total_sample_bd

    return acc_clean, acc_bd

def fine_pruning(args):
    
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
    
    # mode = args.attack_mode
    # # args.checkpoint = os.path.join(args.checkpoint, 'best_model.pth')

    # if args.model == 'myresnet18':
    #     from models.resnet import ResNet18, ResNet
    #     model = ResNet18(num_classes=args.nb_classes)
    # elif args.model == 'mypreactresnet18':
    #     from models.preact_resnet import PreActResNet18, PreActResNet
    #     model = PreActResNet18(num_classes=args.nb_classes)
    # elif args.model == 'mymnistnet':
    #     from models.mnist_net import MNISTNet
    #     model = MNISTNet()
    
    # state_dict = torch.load(args.checkpoint)
    # print("----------LOADING MODEL----------")
    # model.load_state_dict(state_dict['model'])
    # model.to(args.device)

    # model.eval()
    # model.requires_grad_(False)
    
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

    clean_test_loader = torch.utils.data.DataLoader(
        clean_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=True
    )
    bd_test_loader = torch.utils.data.DataLoader(
        bd_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=True
    )

    container = []

    def forward_hook(module, input, output):
        container.append(output)

    if 'resnet' in args.model:
        hook = model.layer4.register_forward_hook(forward_hook)
    else:
        hook = model.layer3.register_forward_hook(forward_hook)

    print("Forwarding all the validation dataset")
    for batch_idx, (inputs, _) in enumerate(clean_test_loader):
        inputs = inputs.to(args.device)
        model(inputs)

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool).to(args.device)
    # print(pruning_mask.shape)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []
    args.output_path = "{}/results.txt".format(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_path, "w") as outs:
        for index in range(pruning_mask.shape[0]):
            net_pruned = copy.deepcopy(model)
            num_pruned = index
            if index:
                channel = seq_sort[index - 1]
                pruning_mask[channel] = False
            # print("Pruned {} filters".format(num_pruned))
            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.linear = nn.Linear((pruning_mask.shape[0] - num_pruned), args.num_classes)
            # assert isinstance(model, ResNet), 'Model should be ResNet-18 for CELEBATTR or T-IMNET'
            for name, module in net_pruned._modules.items():
                if "layer4" in name:
                    # conv2_mask = torch.ones_like(model.layer4[1].conv2.weight.data).to(args.device)
                    # conv2_mask = conv2_mask[]
                    conv2_mask = convert(pruning_mask).to(args.device)
                    module[1].conv2.weight.data = model.layer4[1].conv2.weight.data * conv2_mask
                    module[1].bn2.running_mean = model.layer4[1].bn2.running_mean * pruning_mask
                    module[1].bn2.running_var = model.layer4[1].bn2.running_var * pruning_mask
                    module[1].bn2.weight.data = model.layer4[1].bn2.weight.data * pruning_mask
                    module[1].bn2.bias.data = model.layer4[1].bn2.bias.data * pruning_mask

                    module[1].ind = pruning_mask

                elif "fc" == name:
                    # converted_mask = convert(pruning_mask)
                    # print(converted_mask.shape)
                    # print(module.weight.data)
                    # module.weight.data = model.fc.weight.data[:, converted_mask]
                    module.weight.data = model.fc.weight.data * pruning_mask
                    module.bias.data = model.fc.bias.data
                else:
                    continue
        

            # if args.data_set == 'CELEBATTR' or args.data_set == 'T-IMNET':
            #     net_pruned.layer4[1].conv2 = nn.Conv2d(
            #         pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            #     )
            #     net_pruned.linear = nn.Linear(4 * (pruning_mask.shape[0] - num_pruned), args.nb_classes)
            #     # assert isinstance(model, ResNet), 'Model should be ResNet-18 for CELEBATTR or T-IMNET'
            #     for name, module in net_pruned._modules.items():
            #         if "layer4" in name:
            #             module[1].conv2.weight.data = model.layer4[1].conv2.weight.data[pruning_mask]
            #             module[1].bn2.running_mean = model.layer4[1].bn2.running_mean[pruning_mask]
            #             module[1].bn2.running_var = model.layer4[1].bn2.running_var[pruning_mask]
            #             module[1].bn2.weight.data = model.layer4[1].bn2.weight.data[pruning_mask]
            #             module[1].bn2.bias.data = model.layer4[1].bn2.bias.data[pruning_mask]

            #             module[1].ind = pruning_mask

            #         elif "linear" == name:
            #             converted_mask = convert(pruning_mask)
            #             module.weight.data = model.linear.weight.data[:, converted_mask]
            #             module.bias.data = model.linear.bias.data
            #         else:
            #             continue
            # elif args.data_set == 'GTSRB' or args.data_set == 'CIFAR10':
            #     net_pruned.layer4[1].conv2 = nn.Conv2d(
            #         pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            #     )
            #     net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, args.nb_classes)
            #     # assert isinstance(model, PreActResNet), 'Model should be PreActResNet-18 for GTSRB or CIFAR10'
            #     for name, module in net_pruned._modules.items():
            #         if "layer4" in name:
            #             module[1].conv2.weight.data = model.layer4[1].conv2.weight.data[pruning_mask]
            #             module[1].ind = pruning_mask
            #         elif "linear" == name:
            #             module.weight.data = model.linear.weight.data[:, pruning_mask]
            #             module.bias.data = model.linear.bias.data
            #         else:
            #             continue
            # elif args.data_set == 'MNIST':
            #     net_pruned.layer3.conv1 = nn.Conv2d(
            #         pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=2, padding=1, bias=False
            #     )
            #     net_pruned.linear6 = nn.Linear((pruning_mask.shape[0] - num_pruned) * 16, 512)

            #     # Re-assigning weight to the pruned net
            #     for name, module in net_pruned._modules.items():
            #         if "layer3" in name:
            #             module.conv1.weight.data = model.layer3.conv1.weight.data[pruning_mask]
            #             module.ind = pruning_mask
            #         elif "linear6" == name:
            #             module.weight.data = model.linear6.weight.data.reshape(-1, 64, 16)[:, pruning_mask].reshape(
            #                 512, -1
            #             )  # [:, pruning_mask]
            #             module.bias.data = model.linear6.bias.data
            #         else:
            #             continue

            net_pruned.to(args.device)
            # clean, bd = eval(net_pruned, identity_grid, noise_grid, test_dl, opt)
            # outs.write("%d %0.4f %0.4f\n" % (index, clean, bd))

            clean, bd = eval(net_pruned, clean_test_loader, bd_test_loader, args)
            outs.write("%d %0.4f %0.4f\n" % (index, clean, bd))
            outs.flush()
            print(f'Pruned {num_pruned+1} filters | Clean Accuracy: {clean:.4f} | Backdoor ASR: {bd:.4f}')
            
            # print(f'')

