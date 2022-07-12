import timm
import torch
import torch.nn as nn

from collections import deque

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import os

import time

class CFG:

    device = 'cuda'

    model_name = 'efficientnet_b0'
    model_pretrained = True
    model_num_class = 4

    img_resize = (256, 256)

    weight_path = './weights/07-01_size224_efficientnet_b0_epoch_3.pt'

    transformed = A.Compose([
                            # A.
                            A.Resize(img_resize[0], img_resize[1]),
                            # A.Normalize(),
                            ToTensorV2()
                            ])

    # testset_data_path = './dataset3/test/'
    testset_data_path = './dataset4/val/'

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model(CFG.model_name, checkpoint_path=CFG.weight_path, num_classes=CFG.model_num_class)
        
        # print(f'Model_structure: {self.model}')

    def forward(self,x):
        output = self.model(x)
        return output

model = Model()
# model.load_state_dict(torch.load(CFG.weight_path))
model.to(CFG.device)

print(model)
# torch.randn()