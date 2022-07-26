import torch
import torch.nn as nn
import torch.optim as optim

import timm
import os

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import random

from collections import deque


import matplotlib.pyplot as plt

import numpy as np

# mask  --> 0
# nomask --> 1
# wrong   --> 2

# efficientnet_b6

class CFG:

    # batch_size = 32
    batch_size = 32
    epochs = 200
    lr = 3e-4
    device = 'cuda'
    # model_name = 'efficientnet_b0'
    model_name = 'efficientnet_b0'
    model_pretrained = True
    model_num_class = 4

    # img_resize = (64, 64)
    # 224로 테스트 한번 해보겠습니다.
    img_resize = (256, 256)

    # 얼굴크기 48로도 테스트 해볼것.

    # train_img_path = './dataset3/train/'
    # val_img_path = './dataset3/val/'

    train_img_path = './dataset4/train/'
    val_img_path = './dataset4/val/'

    weight_save_path = './weights/'

    # 필요시 여기에 추가하면 됩니다.
    transform = A.Compose([
                        # A.Cutout(),
                        # 컷아웃 적용시 loss가 더 올라갑니다.
                        
                        A.RandomBrightness(),
                        A.RGBShift(),
                        # A.RandomRotate90(),
                        # 나중에 rotate도 
                        A.HorizontalFlip(),
                        
                        # BLUR는 고화질이미지에서만 적용되도록 하기
                        # A.AdvancedBlur(p=1),
                        # A.Blur(p=1),                        

                        A.Resize(img_resize[0],img_resize[1]),
                        A.RandomCrop(205,205,p=0.8),
                        A.Resize(img_resize[0],img_resize[1]),
                        
                        A.Normalize(),
                        ToTensorV2()
                        ])
    
    transform_lowQulity_Img = A.Compose([
                        # A.Cutout(),
                        # 컷아웃 적용시 loss가 더 올라갑니다.
                        
                        A.RandomBrightness(),
                        A.RGBShift(),
                        # A.RandomRotate90(),
                        # 나중에 rotate도 
                        A.HorizontalFlip(),
                        A.Blur(p=0.8),

                        A.Resize(img_resize[0],img_resize[1]),
                        A.RandomCrop(205,205,p=0.8),
                        A.Resize(img_resize[0],img_resize[1]),
                        
                        A.Normalize(),
                        ToTensorV2()
                        ])

# 다음 augmentation에서는 BLUR도 넣어야합니다.
# 화질낮은 이미지에서는 감지가 잘 안되서 그렇습니다.


    transform_val = A.Compose([A.Resize(img_resize[0],img_resize[1]),
                               A.Normalize(),
                                ToTensorV2()])

# 이거 그래프랑 똑같이 나오면 padding 집어넣고 다시 해보기 ㅎㅎ
# padding 집어넣었을때 성능향상 효과가 있는지 . 

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)
    # np.random.seed(random_seed)

set_seed(42)

def subplotImg(img1, img2):

    fig = plt.figure()
    rows = 1
    cols = 2

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.axis('off')

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.axis('off')

    plt.show()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=CFG.model_pretrained, num_classes=CFG.model_num_class)

    def forward(self,x):
        output = self.model(x)
        return output


class ImageDataset(Dataset):
    def __init__(self, file_list, transform):
        self.transform = transform
        self.file_list = []
        self.labels_list = []
        self.labels = deque([])

        for path in os.listdir(file_list):
            full_path1 = os.path.join(file_list, path)
            for path2 in os.listdir(full_path1):
                self.file_list.append(full_path1+'/'+path2)
                self.labels.append(path)

                #  real label name
                # train 안에 폴더, val안에 폴더만 들고오면 되게만들어놨습니다.
                #  self.labels.append(cnt)
                # cnt+=1   cnt로 이용하여도 문제는 없음을 확인했습니다.

        for _ in range(len(self.labels)):
            i = self.labels.popleft()
            if i == 'mask':
                self.labels_list.append(0)
            elif i == 'nomask':
                self.labels_list.append(1)
            elif i == 'wrong':
                self.labels_list.append(2)
            elif i == 'blind':
                self.labels_list.append(3)


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        label = self.labels_list[index]

        image_ = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # image_ = transform_keep_ratio(image_)

        # 이거해도 오리지널보다 loss 안낮아지더랑
        
        if self.transform:
            if image_.shape[0]>50 and image_.shape[1]>50:
                transformed = CFG.transform_lowQulity_Img(image=image_)
                image = transformed['image']

            else:
                transformed = CFG.transform(image=image_)
            image = transformed['image']
        else:
            pass
        
        # 디버깅용
        # image = image.permute(1,2,0).numpy()
        # cv2.imshow('kk', image)
        # cv2.waitKey(0)
    
        return image, label


# 비율작은거 없애기 
# 일정 크기이하 
# list

# w/image.shape h/image
# 세로 가로 0.08:1  and  이하 list


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, mean=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.mean = mean

    def forward(self, inputs, targets):

        CELoss = nn.CrossEntropyLoss()(inputs, targets)
        # CELoss = -ln(pt) 이니깐 pt = e^(-CELoss)
        pt = torch.exp(-CELoss)
        FocalLoss = self.alpha*((1-pt)**self.gamma) * CELoss
        
        if self.mean:
            return torch.mean(FocalLoss)
        else:
            return FocalLoss

def transform_keep_ratio(image):
    h, w = image.shape[0], image.shape[1]
    if h == w:
        return image
    else:
        max_value = max(h,w)
        ones = np.ones((max_value, max_value, 3), dtype=np.uint8)
        
        # green image
        ones[:,:,1] = ones[:,:,1]*255
        diff_value = abs(h-w)//2
        if h > w:
            ones[:,diff_value:diff_value+w,:] = image
        else:
            ones[diff_value:diff_value+h,:,:] = image
        return ones

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
        print(f'{val_loss_arr.index(min(val_loss_arr))+1}epoch 에서 loss : {min(val_loss_arr)} 입니다.')

if __name__ == "__main__":

    train_loss_arr = []
    val_loss_arr = []

    model = Model().to(CFG.device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)

    train_dataset = ImageDataset(CFG.train_img_path, transform=CFG.transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CFG.batch_size)

    val_dataset = ImageDataset(CFG.val_img_path, transform=CFG.transform_val)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=CFG.batch_size)

    # 이거 dataset random split 해서 평가할까 흠 . . .
    # -> 그렇게 해도 seed 고정되어 있어서 일정하게 될거같기는 합니다.
    
    print(train_dataset)

    for epoch in range(1, CFG.epochs+1):
        train_one_epoch(model, optimizer, train_loader, epoch, train_loss_arr, CFG.device)
        val_one_epoch(model, optimizer, val_loader, epoch, val_loss_arr, CFG.device)
        # 가중치저장하는 제안된방식으로 바꿉니다.
        # torch.save(model.state_dict(), CFG.weight_save_path+f'07-01_size224_{CFG.model_name}_epoch_{epoch}.pt')
        torch.save(model.state_dict(), CFG.weight_save_path+f'07-14_size256_{CFG.model_name}_epoch_{epoch}.pt')

        print(train_loss_arr)
        print(val_loss_arr)

    # 여기서 albumentations augmentation 기법 적용해보기.
    # -> 나중에 도움된다. -> 완료
 