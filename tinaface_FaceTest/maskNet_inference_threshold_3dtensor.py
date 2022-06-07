import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    model_num_class = 3

    img_resize = (64, 64)

    weight_path = './weights/dataset4_efficientnet_b0_epoch_116.pt'

    transformed = A.Compose([A.Resize(img_resize[0], img_resize[1]),
                            # A.Normalize(),
                            ToTensorV2()
                            ])

    # testset_data_path = './dataset3/test/'
    testset_data_path = './faceImg/'

    test_save_path = './0.5threadhold_images/'
    confidenceThreshold = 0.5

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=CFG.model_pretrained, num_classes=CFG.model_num_class)
        # print(f'Model_structure: {self.model}')

    def forward(self,x):
        output = self.model(x)
        return output

model = Model()
model.load_state_dict(torch.load(CFG.weight_path))
model.to(CFG.device)

labels = ['mask', 'nomask', 'wrong']

with torch.no_grad():
    model.eval()
    # tensor_img = torch.Tensor(cv2.resize(img)) -> 이런방법도 있다고합니다.
    path = './faceImg/'
    img = os.listdir(path)
    img_list = []
    for i in img:
        img_list.append(path+i)
    cnt = 0
    for k in img_list:

        st = time.time()
        image = cv2.imread(k)

        transformed = CFG.transformed(image=image)
        transformed_img = transformed["image"].unsqueeze(0).float().to(CFG.device)

        output = model(transformed_img)

        # print(output.tolist()[0])
        print(output)
        f = open('./output.txt', 'a')
        f.write(f'{output.tolist()[0]}/')

        confidence = F.softmax(output).max()

        out_label = torch.argmax(output).item()

        if confidence<CFG.confidenceThreshold:
            imageName = k.split('/')[2]
            print(f"confidence:{confidence}")
            print(f"output    :{labels[out_label]}")

            cv2.imwrite(f"{CFG.test_save_path}pred_{labels[out_label]}_{imageName}", image)
            # cv2.imshow('kk',image)
            # cv2.waitKey(0)

        if out_label == 0:
            cv2.imwrite(f"./resultImg/mask/{k.split('/')[2]}",image)
        elif out_label == 1:
            cv2.imwrite(f"./resultImg/nomask/{k.split('/')[2]}",image)
        elif out_label == 2:
            cv2.imwrite(f"./resultImg/wrong/{k.split('/')[2]}",image)
        ed = time.time()
        # print(out_label)

# 1 -> camera 전원인가 충전이되고있는건지 몰겠다네융, 충전안되면곤란
# 2 -> 

# 5
# 55 185  
# 58 183
# 88 186
# 60 175
# 46 155          2 2 1 2 5
   