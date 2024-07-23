import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from scipy.fftpack import dct, idct
import albumentations as A

import cv2

import random
from tqdm import tqdm
import copy

import keras
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers.normalization.batch_normalization import BatchNormalization

from ...poisoned_datasets import *

def load_weights_from_keras(path='/vinserver_user/jason/backdoor-fpt/python/defenses/frequency_detection/6_CNN_CIF1R10.h5py'):
    def keras_to_pyt(km, pm=None):
        weight_dict = dict()
        for i, layer in enumerate(km.layers):
            if (type(layer) is Conv2D):
                weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
            elif type(layer) is keras.layers.Dense:
                weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
            elif type(layer) is BatchNormalization:
                weight_dict[layer.get_config()['name'] + '.weight'] = layer.get_weights()[0]
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        if pm:
            pyt_state_dict = pm.state_dict()
            for key in pyt_state_dict.keys():
                pyt_state_dict[key] = torch.from_numpy(weight_dict[key])
            pm.load_state_dict(pyt_state_dict)
            return pm
        return weight_dict


    weight = keras.models.load_model(path)
    weight = keras_to_pyt(weight)
    # print(weight.keys())
    name_dict = {
        'conv2d_5' : 'conv1.0',
        'batch_normalization_6' : 'conv1.2',
        'conv2d_6' : 'conv1.3',
        'batch_normalization_7' : 'conv1.5',
        'conv2d_7' : 'conv2.0',
        'batch_normalization_8' : 'conv2.2',
        'conv2d_8' : 'conv2.3',
        'batch_normalization_9' : 'conv2.5',
        'conv2d_9' : 'conv3.0',
        'batch_normalization_10' : 'conv3.2',
        'last_conv' : 'conv3.3',
        'batch_normalization_11' : 'conv3.5',
        'dense' : 'fc'
    }
    new_weight = {}
    for (old_key, new_key) in name_dict.items():
        new_weight[new_key + '.weight'] = torch.tensor(weight[old_key + '.weight'])
        new_weight[new_key + '.bias'] = torch.tensor(weight[old_key + '.bias'])

    return new_weight
# print(new_weight.keys())

def dct2(block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')

def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')

def addnoise(img):
    aug = A.GaussNoise(p=1,mean=25,var_limit=(10,70))
    augmented = aug(image=(img*255).astype(np.uint8))
    auged = augmented['image']/255
    return auged

def randshadow(img):
    aug = A.RandomShadow(p=1)
    test = (img*255).astype(np.uint8)
    augmented = aug(image=cv2.resize(test,(32,32)))
    auged = augmented['image']/255
    return auged

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def patching_train(clean, x_train, batch_size=32):
    attack = np.random.randint(0, 5)
    patch_size = np.random.randint(2, 8)
    patch_size = [patch_size, patch_size]# A square block of patch
    patch_size += [3]
    output = copy.deepcopy(clean)
    output = np.array(output) / 255.

    if attack == 0:
        patch = np.ones(patch_size)
    elif attack == 1:
        patch = np.random.rand(patch_size[0], patch_size[1], patch_size[2])
    elif attack == 2:
        return addnoise(output)
    elif attack == 3:
        return randshadow(output)
    elif attack == 4:
        randind = np.random.randint(batch_size)
        tri = np.array(x_train[randind][0])
        mid = output + 0.3 * tri
        mid[mid>1] = 1
        return mid
    
    margin = np.random.randint(0, 6)
    rand_loc = np.random.randint(0, 4)

    if rand_loc == 0:
        output[margin: margin+patch_size[0], margin: margin+patch_size[1], :] = patch
    elif rand_loc==1:
        output[margin:margin+patch_size[0],32-margin-patch_size[1]:32-margin,:] = patch
    elif rand_loc==2:
        output[32-margin-patch_size[0]:32-margin,margin:margin+patch_size[1],:] = patch
    elif rand_loc==3:
        output[32-margin-patch_size[0]:32-margin,32-margin-patch_size[1]:32-margin,:] = patch #right bottom

    output[output>1] = 1
    return output


class FrequencyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (3, 3), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.4)
        )

        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x).softmax(-1)
        return x
    
def create_dataset(dataset, data_path='./data', is_train=True):
    if dataset == 'CIFAR10':
        data = torchvision.datasets.CIFAR10(data_path, train=is_train)
    else:
        raise NotImplementedError('Not implemented function')

    return data

def train_detector(dataset):
    batch_size = 64
    epochs = 10
    data = create_dataset(dataset)
    model = FrequencyDetector()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.05, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # totensor = torchvision.transforms.Compose([
    #     # torchvision.transforms.ToPILImage(),
    #     # torchvision.transforms.ToTensor()
    #     lambda x : x.permute(1, 2, 0)
    # ])
    # totensor = lambda x : x.transpose()
    # totensor = lambda x : np.moveaxis(x, [0, 1, 2], [1, 2, 0])
    poison = np.zeros((len(data), 32, 32, 3))
    x_train = np.zeros((len(data), 32, 32, 3))
    y_train = (np.vstack((np.zeros((x_train.shape[0], 1)), np.ones((x_train.shape[0], 1))))).astype(int)
    for i in tqdm(range(len(data))):
        poison[i] = patching_train(data[i][0], data, batch_size=batch_size)
        x_train[i] = np.array(data[i][0]) / 255.

        # poison[i] = totensor(poison[i])
        # x_train[i] = totensor(x_train[i])
        # y_train[i] = data[i][1]
    
    x_train = np.vstack((x_train, poison))

    idx = np.arange(x_train.shape[0])
    random.shuffle(idx)
    

    x_train = x_train[idx]
    y_train = y_train[idx]
    for ep in range(epochs):
        print(f'Epoch {ep}')
        model.train()
        total_loss = 0

        ys = []
        preds = []
        for i in tqdm(range(x_train.shape[0] // batch_size + 1)):
            optimizer.zero_grad()

            if i == x_train.shape[0] // batch_size + 1:
                x = x_train[i*batch_size:]
                y = y_train[i*batch_size:]
            else:
                x = x_train[i*batch_size : (i+1) * batch_size]
                y = y_train[i*batch_size : (i+1) * batch_size]


            for j in range(batch_size):
                try:
                    for ch in range(3):
                        x[j][:, :, ch] = dct2((x[j][:, :, ch] * 255.).astype(np.uint8))
                except:
                    # print('Done transforming all')
                    continue

            x = torch.tensor(x).float().to(device)
            y = np.eye(2)[y].squeeze()
            y = torch.tensor(y).float().to(device)
            x = x.permute(0, 3, 1, 2)
            # print(x.shape)
            # print(y.shape)
            pred_y = model(x)
            # print(pred_y.shape)
            loss = criterion(y, pred_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            ys.append(y[:, 1].cpu())
            preds.append(pred_y.argmax(1).cpu())
            # break
        total_loss /= x_train.shape[0]

        ys = torch.cat(ys)
        preds = torch.cat(preds)

        print(f'Loss at epoch {ep} = {total_loss:.4f}')
        print(f'Accuracy at epoch {ep} = {torch.sum(ys == preds)/ ys.shape[0]}')



if __name__ == '__main__':
    dataset = 'CIFAR10'
    # train_detector(dataset)
    dataset = torchvision.datasets.CIFAR10('../../../data', train=False, transform=torchvision.transforms.ToTensor())
    transform = transforms.Compose([
        # transforms.Resize(32, interpolation=3),
        # transforms.RandomCrop((32, 32), padding=5),
        # # transforms.RandomRotation(10),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD)
    ])
    dataset = FileExtensionPoisonedDataset(dataset, label_transform=all2one_target_transform, transform=transform, portion=1.0, is_train=False,
                                        compression_method='PIL', compression_quality=70, compression_type='JPEG')
    
    model = FrequencyDetector()

    labels = []
    preds = []
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False
    )

    for (data, label, ori_data, ori_label) in tqdm(loader):
        data = data.float()
        for i in range(data.shape[0]):
            for j in range(3):
                data[i][j, :, :] = torch.tensor(dct2((data[i][j, :, :].numpy() * 255).astype(np.uint8))[None, ...]).float()
        data = torch.tensor(data).float()
        label = torch.ones_like(label).float() # Class 1 for backdoor

        with torch.no_grad():
            pred = model(data)
        
        preds.append(pred.argmax(-1))
        labels.append(label)

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    print(torch.sum(preds == labels) / preds.shape[0])


