from cProfile import label
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


import torchvision.transforms as transforms

from PIL import Image

import timm
from tqdm import tqdm
import os


class CFG:
    batch_size = 5
    epochs = 20
    lr = 3e-4
    device = 'cuda'
    model_name = 'efficientnet_b0'
    model_pretrained = True
    model_classifier = 2

    train_img_path = './dataset/train/'
    val_img_path = './dataset/val/'
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((512,512))
                                    ])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=CFG.model_pretrained)
        self.model.classifier = nn.Linear(1280, CFG.model_classifier)

    def forward(self,x):
        output = self.model(x)
        return output


class ImageDataset(Dataset):
    def __init__(self, file_list, transform):
        self.transform = transform
        self.file_list = []
        self.labels = []

        cnt = 0
        for path in os.listdir(file_list):
            full_path1 = os.path.join(file_list, path)
            for path2 in os.listdir(full_path1):
                self.file_list.append(full_path1+'/'+path2)
                # self.labels.append(path) -> real label name
                self.labels.append(cnt)
            cnt+=1

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        label = self.labels[index]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        else:
            pass

        return image, label


def train_one_epoch(model, optimizer, dataloader, epoch, device):

    model.train()
    dataset_size = 0
    running_loss = 0

    # optimizer = optimizer(model.parameters(), lr=lr)
    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, data in bar:
        # print(f"data:  {data[0]}")
        # image tensor
        # print(f"data:  {data[1]}")
        # label tensor

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

def val_one_epoch(model, optimizer, dataloader, epoch, device):
    global data, images

    model.eval()
    dataset_size = 0
    running_loss = 0
    # optimizer = optimizer(model.parameters(), lr=lr)
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        # print(f"data:  {data[0]}")
        # image tensor
        # print(f"data:  {data[1]}")
        # label tensor
        images = data[0].to(device, dtype=torch.float)
        labels = data[1].to(device, dtype=torch.long)

        batch_size = images.size(0)
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        running_loss += loss.item()*batch_size
        dataset_size += batch_size
        epoch_loss = running_loss/dataset_size

        bar.set_postfix(EPOCH=epoch, VAL_LOSS=epoch_loss)

if __name__ == "__main__":

    model = Model().to(CFG.device)
    optimizer = optim.Adam(model.parameters(), lr = CFG.lr)

    train_dataset = ImageDataset(CFG.train_img_path, transform=CFG.transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)

    val_dataset = ImageDataset(CFG.val_img_path, transform=CFG.transform)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1)

    for epoch in range(1, CFG.epochs+1):
        train_one_epoch(model, optimizer, train_loader, epoch, CFG.device)
        val_one_epoch(model, optimizer, val_loader, epoch, CFG.device)

