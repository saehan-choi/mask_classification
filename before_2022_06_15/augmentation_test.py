import cv2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

transform = A.Compose([
                    # A.Cutout(),
                    # # 컷아웃이 오히려 문제일 수 있다.
                    A.RandomBrightness(),
                    A.RGBShift(),
                    # A.RandomRotate90(),
                    # 나중에 rotate도 
                    A.HorizontalFlip(),
                    # A.Resize(img_resize[0],img_resize[1]),
                    # A.Normalize(),
                    # ToTensorV2()
                    ])



image = cv2.imread('./dataset4/train/nomask/2_01274.jpg')

transformed = transform(image=image)
image = transformed['image']

# plt써서 augmentation 안되었을때랑 되었을때 두개띄우기

cv2.imshow('kk', image)
cv2.waitKey(0)