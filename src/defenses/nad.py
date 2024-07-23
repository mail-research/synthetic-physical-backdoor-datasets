"""
This is the implement of NAD [1]. 
This source is modified from BackdoorBox codebase

Reference:
[1] Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks. ICLR 2021.
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
import tqdm 
import copy

import os
import timm

import numpy as np
import random
import torch.backends.cudnn as cudnn

import torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import transforms_imagenet_train
# from models.base_model import BaseModel

# from agem import *
# from utils.logger import *
# from utils.load import *

class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks via Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am

def perform_NAD(args):

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    model = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes)
    state_dict = torch.load(os.path.join(args.checkpoint, 'best_model.pth'))
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    if args.nad_optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), args.nad_lr, momentum=0.9, weight_decay=args.nad_wd)
    elif args.nad_optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.nad_lr, weight_decay=args.nad_wd)
    elif args.nad_optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.nad_lr, weight_decay=args.nad_wd)

    if args.nad_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nad_epochs)
    
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

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(args.clean_path, 'train'),
        transform=train_transforms
    )
    np.random.seed(args.random_seed)
    # Taking only 5 percent
    indices = np.random.choice(len(train_dataset), int(0.05 * len(train_dataset)), replace=False)
    train_dataset = torch.utils.data.Subset(train_dataset, indices)
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(args.input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    test_clean_dataset = torchvision.datasets.ImageFolder(
        os.path.join(args.clean_path, 'test'),
        transform=val_transforms
    )
    test_bd_dataset = torchvision.datasets.ImageFolder(
        os.path.join(args.bd_path, 'test'),
        transform=val_transforms,
        target_transform=lambda x: args.atk_target
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    test_clean_loader = torch.utils.data.DataLoader(
        test_clean_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    test_bd_loader = torch.utils.data.DataLoader(
        test_bd_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    testloaders = [test_clean_loader, test_bd_loader]
    criterion = nn.CrossEntropyLoss()
    NAD(model, train_loader, optimizer, scheduler, criterion, testloaders, args.nad_epochs, args.device, args)

def NAD(net, dataloader, optimizer, lr_scheduler, criterion, testloaders, epochs, device, args):
    
    if not hasattr(args, 'nad_power'):
        args.power = 2.0
    if not hasattr(args, 'nad_beta'):
        args.beta = [500,500,500]
    if not hasattr(args, "nad_target_layers"):
        args.target_layers=['layer2', 'layer3', 'layer4']

    if not hasattr(args, "tune_epochs"):
        args.tune_epochs = 20
    # Finetune and get the teacher model
    teacher_model = copy.deepcopy(net)
    teacher_model = teacher_model.to(device)
    teacher_model.train()

    # t_optimizer, t_lr_scheduler = load_optimizer_and_scheduler(teacher_model, args)

    if args.nad_optimizer == 'momentum':
        t_optimizer = torch.optim.SGD(teacher_model.parameters(), args.nad_lr, momentum=0.9, weight_decay=args.nad_wd)
    elif args.nad_optimizer == 'adamw':
        t_optimizer = torch.optim.AdamW(teacher_model.parameters(), args.nad_lr, weight_decay=args.nad_wd)
    elif args.nad_optimizer == 'sgd':
        t_optimizer = torch.optim.SGD(teacher_model.parameters(), args.nad_lr, weight_decay=args.nad_wd)

    if args.nad_scheduler == 'cosine':
        t_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.tune_epochs)

    # t_optimizer = args.nad_optimizer
    # t_lr_scheduler = args.nad_scheduler

    args.output_path = "{}/log.txt".format(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    outs = open(args.output_path, "w")
    # print_and_log(args.logger, "="*50)
    # print_and_log(args.logger, "Finetune teacher model")
    # print_and_log(args.logger, "="*50)

    for epoch in range(args.tune_epochs):
        loss = finetune_epoch(teacher_model, dataloader, t_optimizer,
                            t_lr_scheduler, criterion, epoch, device, args)
    
        clean_acc = test_model(teacher_model, testloaders[0], device, args)
        poison_acc = test_model(teacher_model, testloaders[1], device, args)
        # if args.wandb is not None:  # Log results with wandb
        #     args.wandb.log(
        #         {
        #             f'Finetuning Teacher model (NAD) Training Loss': loss,
        #             f'Finetuning Teacher model (NAD) Clean Accuracy': clean_acc,
        #             f'Finetuning Teacher model (NAD) Attack Success Rate': poison_acc,
        #         }, 
        #         # step=epoch,
        #     )
        outs.write(f"Training Loss: {loss:.4f} | Clean Acc: {clean_acc:.4f} | ASR: {poison_acc:.4f} \n")
        outs.flush()

    # Perform NAD and get the repaired model
    for param in teacher_model.parameters():
            param.requires_grad = False
    net = net.to(device)
    net.train()

    criterionAT = AT(args.power)

    # print_and_log(args.logger, "="*50)
    # print_and_log(args.logger, "Performing NAD ...")
    # print_and_log(args.logger, "="*50)
    outs.write("=" * 50 + '\n')
    outs.write("Performing NAD ...\n")
    outs.write("=" * 50 + '\n')
    outs.flush()

    if args.nad_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nad_epochs)
    
    if args.nad_optimizer == 'momentum':
        optimizer = torch.optim.SGD(net.parameters(), args.nad_lr, momentum=0.9, weight_decay=args.nad_wd)
    elif args.nad_optimizer == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), args.nad_lr, weight_decay=args.nad_wd)
    elif args.nad_optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), args.nad_lr, weight_decay=args.nad_wd)

    for epoch in range(epochs):
        loss = perform_NAD_epoch(net, teacher_model, dataloader, optimizer, 
                                lr_scheduler, criterion, criterionAT, epoch, 
                                device, args)
        
        clean_acc = test_model(net, testloaders[0], device, args)
        poison_acc = test_model(net, testloaders[1], device, args)
        # if args.wandb is not None:  # Log results with wandb
        #     args.wandb.log(
        #         {
        #             f'NAD Training Loss': loss,
        #             f'NAD Clean Accuracy': clean_acc,
        #             f'NAD Attack Success Rate': poison_acc,
        #         }, 
        #         # step=epoch,
        #     )
        outs.write(f"NAD Training Loss: {loss:.4f} | NAD Clean Acc: {clean_acc:.4f} | NAD ASR: {poison_acc:.4f} \n")
        outs.flush()

# NAD loop
def perform_NAD_epoch(net, teacher_model, dataloader, optimizer, lr_scheduler, criterion, criterionAT, epoch, device, args):
    net.train()
    avg_loss = 0
    count = 0
    pbar = tqdm.tqdm(dataloader, desc=f'Finetuning Epoch: {epoch}')
    for inputs, targets in pbar:
        # if args.debug_mode and count > 2:
        #     break

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        container = []
        def forward_hook(module, input, output):
            container.append(output)

        hook_list = []
        for name, module in net._modules.items():
            if name in args.target_layers:
                hk = module.register_forward_hook(forward_hook)
                hook_list.append(hk)
        
        for name, module in teacher_model._modules.items():
            if name in args.target_layers:
                hk = module.register_forward_hook(forward_hook)
                hook_list.append(hk)

        # forward to add intermediate features into containers 
        outputs = net(inputs)
        _ = teacher_model(inputs)

        for hk in hook_list:
            hk.remove()

        loss = criterion(outputs, targets)
        AT_loss = 0
        for idx in range(len(args.beta)):
            AT_loss = AT_loss + criterionAT(container[idx], container[idx+len(args.beta)]) * args.beta[idx]  
        
        pbar.set_postfix({'loss': loss.item(), 'AT_loss': AT_loss.item()})
        
        loss = loss + AT_loss
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        count += 1

    avg_loss = avg_loss/count

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(avg_loss)
        else:
            lr_scheduler.step(epoch)
    print(lr_scheduler.get_last_lr())
    return avg_loss

# Training loop
def finetuning(net, dataloader, optimizer, lr_scheduler, criterion, testloaders, epochs, device, args):
    loss_hist = []
    # try:
    #     print_and_log(args.logger, optimizer)
    #     print_and_log(args.logger, lr_scheduler)
    # except:
    # print(optimizer)
    # print(lr_scheduler)

    for epoch in range(epochs):
        
        loss = finetune_epoch(net, dataloader, optimizer, lr_scheduler,
                            criterion, epoch, device, args)    
        clean_acc = test_model(net, testloaders[0], device, args)
        poison_acc = test_model(net, testloaders[1], device, args)
        # if args.wandb is not None:  # Log results with wandb
        #     args.wandb.log(
        #         {
        #             f'Finetuning Training Loss': loss,
        #             f'Finetuning Clean Accuracy': clean_acc,
        #             f'Finetuning Attack Success Rate': poison_acc,
        #         }, 
        #         # step=epoch,
        #     )
        
        loss_hist.append(loss)

        # if epoch % 10 == 0:
             
            

            # print_and_log(args.logger, f'Fine-tuning epoch {epoch}: Loss: {loss}')
            # print_and_log(args.logger, f'Fine-tuning epoch {epoch} Clean Accuracy: {clean_acc}')
            # print_and_log(args.logger, f'Fine-tuning epoch {epoch} Poison Accuracy: {poison_acc}')

    return loss_hist


# Finetuning loop
def finetune_epoch(net, dataloader, optimizer, lr_scheduler, criterion, epoch, device, args):
    net.train()
    avg_loss = 0
    count = 0
    pbar = tqdm.tqdm(dataloader, desc=f'Finetuning Epoch: {epoch}')
    for inputs, targets in pbar:
        # if args.debug_mode and count > 2:
        #     break

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        count += 1
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = avg_loss/count

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(avg_loss)
        else:
            lr_scheduler.step(epoch)
    print(lr_scheduler.get_last_lr())
    return avg_loss

def test_model(net, dataloader, device, args):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # if args.debug_mode:
            #     break

    accuracy = 100 * correct / total
    return accuracy