from glob import glob
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from collections import deque

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import os

import time

import numpy as np

class CFG:

    device = 'cuda'
    model_name = 'efficientnet_b0'
    model_pretrained = True
    model_num_class = 3

    img_resize = (64, 64)
    weight_path = './weights/dataset4_efficientnet_b0_epoch_116.pt'

    transformed = A.Compose([A.Resize(img_resize[0], img_resize[1]),
                        # A.Normalize(),
                        ToTensorV2()
                        ])

    testset_data_path = './dataset4/val/nomask/*.jpg'

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=CFG.model_pretrained, num_classes=CFG.model_num_class)

    def forward(self,x):
        output = self.model(x)
        return output

model = Model()
model.load_state_dict(torch.load(CFG.weight_path))
model.to(CFG.device)

# 어차피 얼굴의 이미지들은 좌표값으로 나오므로 그 해당값들을
# numpy로 읽은다음에, tensor로 변경하여 처리하면 될 듯합니다.
# 흠... 어케 해야할까

# 그냥 한 파일에 있는거 전체 읽는게 편하지 않나요?
# 그렇게 배치로 테스트하고, numpy image들을 어떻게 batch로 넣을지가 중요할듯합니다.

# img_arr = np.array([])

img_arr = []
# 0.35~0.36s
with torch.no_grad():
    while True:
        model.eval()
        st = time.time()
        file_names = glob(CFG.testset_data_path)

        arr = []
        # img_cat = torch.ones(1,3,64,64).float().to(CFG.device)
        # img_cat = torch.zeros(1,3,64,64).to(CFG.device)
        # print(img_cat.size())

        for file_name in file_names[:500]:
            img = cv2.imread(file_name, cv2.IMREAD_COLOR)

            transformed = CFG.transformed(image=img)
            img = transformed['image'].unsqueeze(0).float().to(CFG.device)
            
            arr.append(img)
            # img_cat = torch.cat((img_cat, img), dim=0)
            # print(img_cat.size())

        img_cat = torch.cat(arr, 0)
        # img_cat = img_cat[1:]
        ed = time.time()
        print(img_cat.size())
        print(f'loading time:{ed-st}s')


        st = time.time()
        k = model(img_cat).size()
        ed = time.time()
        print(f'inference time{ed-st}s')

# tensor = model(img)
        
# print(tensor.size())
# ed = time.time()
# print(f"{ed-st}s passed")
