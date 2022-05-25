# import matplotlib.pyplot as plt

# # plt.plot([1, 1, 1, 1], [2, 3, 5, 10])
# # plt.show()

# print(list(range(0, 100, 10)))


# import os

# path = './dataset5(mask_nomask)/val/aa/'
# k=os.listdir(path)

# for idx, i in enumerate(k):
#     print(f'i:{path+i}')
#     print(f"idx:{path+str(idx)+'.jpg'}")
    
#     os.rename(path+i, path+str(idx)+'.jpg')


import random

num= random.random()

print(str(num)[2:])