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
    model_num_class = 3

    img_resize = (64, 64)

    weight_path = './weights/dataset4_efficientnet_b0_epoch_113.pt'

    transformed = A.Compose([A.Resize(img_resize[0], img_resize[1]),
                        # A.Normalize(),
                        ToTensorV2()
                        ])

    # testset_data_path = './dataset3/test/'
    testset_data_path = './dataset4/val/'

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

with torch.no_grad():
    model.eval()
    # tensor_img = torch.Tensor(cv2.resize(img)) -> 이런방법도 있다고합니다.
    labels = deque([])
    image_list = []
    label_list = []
    for i in os.listdir(CFG.testset_data_path):
        path = CFG.testset_data_path+i
        for j in os.listdir(path):
            image_list.append(path+'/'+j)
            labels.append(i)

    for _ in range(len(labels)):
        i = labels.popleft()
        if i == 'mask':
            label_list.append(0)
        elif i == 'nomask':
            label_list.append(1)
        elif i == 'wrong':
            label_list.append(2)

    falseCount = 0
    rightCount = 0
    cnt = 0

    for k in image_list:
        st = time.time()
        image = cv2.imread(k)

        transformed = CFG.transformed(image=image)
        transformed_img = transformed["image"].unsqueeze(0).float().to(CFG.device)

        output = model(transformed_img)

        out_label = torch.argmax(output).item()

        # print(f'label:{label_list[cnt]}')
        # print(f'out:{out_label}')

        if label_list[cnt] == out_label:
            rightCount+=1
        else:
            falseCount+=1


        ed = time.time()

        cnt+=1

        print(f'accuracy : {round(rightCount/(rightCount+falseCount+1e-10)*100,2)}%')



    # accuracy == (TP+TN) / (TP+TN+FP+FN)


    # print(f'{ed-st}s passed \n')





