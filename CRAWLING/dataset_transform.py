import cv2
import os

path = './CRAWLING/crawling_dataset2/'
ldir = os.listdir(path)

# resize = (480,640)
resize = (640, 480)

for i in ldir:
    img = cv2.imread(path+i)
    img = cv2.resize(img, resize)
    cv2.imwrite(path+i,img)

