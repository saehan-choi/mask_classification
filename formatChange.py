import os
import shutil
# from PIL import Image
import cv2
from cv2 import waitKey

path_img = './dataset2/valid/images/'
path_label = './dataset2/valid/labels/'

# 0 -> Nomask
# 1 -> mask
# 2 -> wrong

result_path_nomask = './dataset3/val/nomask/'
result_path_mask = './dataset3/val/mask/'
result_path_wrong = './dataset3/val/wrong/'

# print(os.listdir(path_label))

for label in os.listdir(path_label):
    try:
        f = open(path_label+label,'r')
        
        img = path_img+label[:-4]+'.jpg'

        img = cv2.imread(img, cv2.IMREAD_COLOR)

        # cv2.imshow('img',img)
        # waitKey(0)

        height, width = img.shape[:2]

        cnt = 0
        # image name 중복을 피하기위해 cnt를 도입함
        for readline in f.readlines():
            line = readline.split()
            line[0] = int(line[0])
            line[1], line[2], line[3], line[4] = width*float(line[1]), height*float(line[2]), width*float(line[3]), height*float(line[4])
            line[1], line[2], line[3], line[4] = int(line[1]-int(line[3]/2)), int(line[2]-int(line[4]/2)), int(line[1]+int(line[3]/2)), int(line[2]+int(line[4]/2))

            img = img[line[2]:line[4], line[1]:line[3]]
            # print(f'{result_path_nomask+label[:-4]}_{cnt}'+'.jpg')

            if int(line[0]) == 0:
                # nomask
                cv2.imwrite(f'{result_path_nomask+label[:-4]}_{cnt}'+'.jpg', img)

            if int(line[0]) == 1:
                # mask
                cv2.imwrite(f'{result_path_mask+label[:-4]}_{cnt}'+'.jpg', img)

            if int(line[0]) == 2:
                # wrong
                cv2.imwrite(f'{result_path_wrong+label[:-4]}_{cnt}'+'.jpg', img)

            cnt+=1


        f.close()
    except:
        print(f'{path_label+label} ERROR 발생')


# x y w h
# x1 y1 x2 y2

# normalize --> x - min(x) / max(x) - min(x)
