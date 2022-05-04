import os
import cv2

class CFG:
    arr = ['train', 'val', 'test']
    dataset_name = 'dataset3'

for dataset in CFG.arr:
    path_original = f'./{CFG.dataset_name}/{dataset}/'
    path = os.listdir(path_original)

    for i in path:
        for j in os.listdir(path_original+i):
            # print(path_original+i+'/'+j)
            img = cv2.imread(path_original+i+'/'+j, cv2.IMREAD_COLOR)
            height, width = img.shape[:2]
            # print(f'height width = {height} {width}')
            if width<64:
                os.remove(path_original+i+'/'+j)
                print(f"{path_original+i+'/'+j} removed")
