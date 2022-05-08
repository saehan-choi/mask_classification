import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np

import timm
import os

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import random

from collections import deque

from PIL import Image

# mask  --> 0
# nomask --> 1
# wrong   --> 2


class CFG:
    batch_size = 64
    epochs = 100
    lr = 3e-4
    device = 'cuda'
    model_name = 'efficientnet_b0'
    model_pretrained = True
    model_num_class = 3

    img_resize = (64, 64)

    train_img_path = './dataset3/train/'
    val_img_path = './dataset3/val/'

    weight_save_path = './weights/'

    # 필요시 여기에 추가하면 됩니다.
    transform = A.Compose([
                        # A.Cutout(),
                        # # 컷아웃이 오히려 문제일 수 있다.
                        # A.RandomBrightness(),
                        # A.RGBShift(),
                        # # A.RandomRotate90(),
                        # A.HorizontalFlip(),
                        # A.Normalize(),
                        A.Resize(img_resize[0],img_resize[1]),
                        ToTensorV2()
                        ])


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)
    # np.random.seed(random_seed)
 
set_seed(42)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=CFG.model_pretrained, num_classes=CFG.model_num_class)
        # print(f'Model_structure: {self.model}')

    def forward(self,x):
        output = self.model(x)
        return output


class ImageDataset(Dataset):
    def __init__(self, file_list, transform):
        self.transform = transform
        self.file_list = []
        self.labels_list = []

        self.labels = deque([])
        
        # cnt = 0
        for path in os.listdir(file_list):
            full_path1 = os.path.join(file_list, path)
            for path2 in os.listdir(full_path1):
                self.file_list.append(full_path1+'/'+path2)
                self.labels.append(path) 

                # -> real label name
                # train 안에 폴더, val안에 폴더만 들고오면 되게만들어놨습니다.
            #     self.labels.append(cnt)
            # cnt+=1   cnt로 이용하여도 문제는 없음을 확인했습니다.

        for _ in range(len(self.labels)):
            i = self.labels.popleft()
            if i == 'dog':
                self.labels_list.append(0)
            elif i == 'cat':
                self.labels_list.append(1)
            # elif i == 'wrong':
            #     self.labels_list.append(2)
        # 이거 코드 더러운데 -> 더 깔끔하게 변환시킬 방법없는지 구상하기.

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # print(f'index:{index}')
        # print(f'labelList{len(self.labels_list)}')
        img_path = self.file_list[index]
        label = self.labels_list[index]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            pass
        
        # img_array = np.array(image)
        # img = cv2.cvtColor(img_array, cv2.IMREAD_COLOR)
        # cv2.imshow('kk', img)
        # cv2.waitKey(0)

        return image, label


def train_one_epoch(model, optimizer, dataloader, epoch, train_loss_arr, device):
    model.train()
    dataset_size = 0
    running_loss = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, data in bar:
        images = data[0].to(device, dtype=torch.float)
        labels = data[1].to(device, dtype=torch.long)


        batch_size = images.size(0)
        outputs = model(images)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()*batch_size
        dataset_size += batch_size
        epoch_loss = running_loss/dataset_size

        bar.set_postfix(EPOCH=epoch, TRAIN_LOSS=epoch_loss)
    train_loss_arr.append(epoch_loss)

def val_one_epoch(model, optimizer, dataloader, epoch, val_loss_arr, device):
    model.eval()
    with torch.no_grad():
        dataset_size = 0
        running_loss = 0

        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:
            
            images = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.long)

            batch_size = images.size(0)
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            running_loss += loss.item()*batch_size
            dataset_size += batch_size
            epoch_loss = running_loss/dataset_size

            bar.set_postfix(EPOCH=epoch, VAL_LOSS=epoch_loss)
        val_loss_arr.append(epoch_loss)

if __name__ == "__main__":

    train_loss_arr = []
    val_loss_arr = []

    model = Model().to(CFG.device)
    optimizer = optim.Adam(model.parameters(), lr = CFG.lr)

    train_dataset = ImageDataset(CFG.train_img_path, transform=CFG.transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CFG.batch_size)

    val_dataset = ImageDataset(CFG.val_img_path, transform=CFG.transform)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=CFG.batch_size)

    for epoch in range(1, CFG.epochs+1):
        train_one_epoch(model, optimizer, train_loader, epoch, train_loss_arr, CFG.device)
        val_one_epoch(model, optimizer, val_loader, epoch, val_loss_arr, CFG.device)
        torch.save(model.state_dict(), CFG.weight_save_path+f'train_dataset_정제_{CFG.model_name}_epoch_{epoch}.pt')

        print(train_loss_arr)
        print(val_loss_arr)

    # 여기서 albumentations augmentation 기법 적용해보기.
    # -> 나중에 도움된다. -> 완료
 