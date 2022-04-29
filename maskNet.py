import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from PIL import Image

import timm
from tqdm import tqdm


class CFG:
    batch_size = 5
    epochs = 2
    ANYTHING = 1
    YOU = 2
    CAN = 3
    ADD = 4


class efficientnet_b0(nn.Module):
    def __init__(self):
        super(efficientnet_b0, self).__init__()
        self.model = timm.create_model('efficientnet_b0')
        self.model.classifier = nn.Linear(1280,3)

    def forward(self,images):
        features = self.model(images)

        return features

class ImageDataset(Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        return img_transformed


def train_one_epoch(model, optimizer, dataloader, device, epoch, lr):
    model.train()
    dataset_size = 0
    running_loss = 0
    
    optimizer = optimizer(model.parameters(), lr=lr)
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)
        batch_size = images.size(0)

        output = model(images)
        label = labels
        loss = nn.CrossEntropyLoss()(output, label)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()*batch_size
        dataset_size += batch_size
        epoch_loss = running_loss/dataset_size

        bar.set_postfix(EPOCH=epoch, TRAIN_LOSS=epoch_loss)


dataloader = DataLoader()