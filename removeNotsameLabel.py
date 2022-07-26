import os

imgpath = './no/images/'
labelpath = './no/labels/'

img_list = os.listdir(imgpath)
label_list = os.listdir(labelpath)

imgArr = {}
for imgName in img_list:
    i_name = imgName.split('.')[0]
    
    imgArr[i_name] = 1
    
for labelName in label_list:
    l_name = labelName.split('.')[0]
    
    if imgArr[l_name]:
        imgArr[l_name] += 1
    else:
        pass
    

for img in imgArr.items():
    if img[1] == 1:
        print(img[0])
        try: 
            os.remove(imgpath+img[0]+'.jpg') 
        except:
            pass
        try:
            os.remove(labelpath+img[0]+'.txt')
        except:
            pass
    else:
        pass
    