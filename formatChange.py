import os
import shutil
# from PIL import Image
import cv2

path_img = './dataset2/test/images/'
path_label = './dataset2/test/labels/'

# 0 -> Nomask
# 1 -> mask
# 2 -> wrong

# 이거 지금 이상하게 받아옴 사진에 4명있는데도 한명것만 가져오니깐 다시해라.

result_path_nomask = './dataset3/test/nomask/'
result_path_mask = './dataset3/test/mask/'
result_path_wrong = './dataset3/test/wrong/'

# print(os.listdir(path_label))

# try except 하고, 없는것들은 append 해서 배열로 만들어서 나오게하면 되겠네.
arr = []

try:
    for label in os.listdir(path_label):
        f = open(path_label+label,'r')

        img = path_img+label[:-4]+'.jpg'
        if os.path.exists(img):
            # 파일있는지 없는지 판단
            print(img)
            img = cv2.imread(img, cv2.IMREAD_COLOR)

        elif os.path.exists(path_img+label[:-4]+'.png'):
            img = path_img+label[:-4]+'.png'
            print(img)
            img = cv2.imread(img, cv2.IMREAD_COLOR)
        
        else:
            arr.append(path_img+label[:-4])
            print()
            continue
        
        height, width = img.shape[:2]

        cnt = 0

        # image name 중복을 피하기위해 cnt를 도입함
        for readline in f.readlines():
            line = readline.split()
            line[0] = int(line[0])
            line[1], line[2], line[3], line[4] = width*float(line[1]), height*float(line[2]), width*float(line[3]), height*float(line[4])
            line[1], line[2], line[3], line[4] = int(line[1]-int(line[3]/2)), int(line[2]-int(line[4]/2)), int(line[1]+int(line[3]/2)), int(line[2]+int(line[4]/2))

            img_ = img[line[2]:line[4], line[1]:line[3]]
            # print(f'{result_path_nomask+label[:-4]}_{cnt}'+'.jpg')

            if int(line[0]) == 0:
                # nomask
                cv2.imwrite(f'{result_path_nomask+label[:-4]}_{cnt}'+'.jpg', img_)

            if int(line[0]) == 1:
                # mask
                cv2.imwrite(f'{result_path_mask+label[:-4]}_{cnt}'+'.jpg', img_)

            if int(line[0]) == 2:
                # wrong
                cv2.imwrite(f'{result_path_wrong+label[:-4]}_{cnt}'+'.jpg', img_)

            cnt+=1

        f.close()
except:
    print(f'except 구간입니다.')

# x y w h
# x1 y1 x2 y2

# normalize --> x - min(x) / max(x) - min(x)
