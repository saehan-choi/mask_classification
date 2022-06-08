import cv2
import os

path = './CRAWLING/crawling_dataset2/'

# print(os.listdir(path))
lid = os.listdir(path)
for i in lid:
    img = cv2.imread(path+i)
    print(f'path:{i}')
    print(img)