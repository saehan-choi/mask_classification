# from collections import deque
# import time
# arr = [1,2,3,4,5,6]


# print(min(arr))

# print(f'{arr.index(min(arr))+1}epoch에서 loss : {min(arr)} 입니다.')


# print([2 for i in range(500)])



# k = list(range(1,10000))



# m = deque(range(1,10000))

# # print(m)

# st = time.time()
# k.reverse()
# for a in range(len(k)):
#     k.pop()
# ed = time.time()
# print(f'{ed-st}s passed')

# st = time.time()
# for a in range(len(m)):
#     m.popleft()
# ed = time.time()
# print(f'{ed-st}s passed')

# 둘의 시간복잡도차이는 거의 없다 하지만 미세하게 deque가 빠르긴하다. popleft 할땐 reverse 후에 pop 해도 무관할듯 ㅎ

import torch
import torch.nn as nn

import math

input    = torch.tensor([0.1, 0.8, 0.1]).unsqueeze(0).float()
output = torch.tensor([1]).long()

# def softmax(input, target):
#     exponential_array = []
#     target = target.pop()
#     for i in input:
#         exponential_array.append(math.exp(i))
    
#     exp_target = exponential_array[target]
#     softmax = exp_target/sum(exponential_array)
#     # for instance) e^0.1 / ( e^0.1 + e^0.8 + e^0.1 )
#     return softmax

# class crossEntropyLoss():
#     def __init__(self, input, output):
#         self.softmax = softmax(input, output)
#         self.log = math.log
        
#     def __forward__(self):
#         return -self.log(self.softmax)


# class FocalLoss():
#     def __init__(self, input, output, alpha=0.25, gamma=2):
#         self.softmax = softmax(input, output)
#         self.log = math.log
#         self.weights = (1-self.softmax)
#         # print(f'self.weights:{self.weights}')
#         self.alpha = alpha
#         self.gamma = gamma

#     def __forward__(self):
#         return -self.alpha*((self.weights)**self.gamma)*self.log(self.softmax)

# input_array = [0.1, 0.8, 0.1]
# output_array = [1]

# loss = FocalLoss(input_array, output_array)
# print(f'FCLoss:{loss.__forward__()}')

# input_array = [0.1, 0.8, 0.1]
# output_array = [1]

# loss = crossEntropyLoss(input_array, output_array)
# print(f'CELoss:{loss.__forward__()}')


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, mean=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.mean = mean

    def forward(self, inputs, targets):

        CELoss = nn.CrossEntropyLoss()(inputs, targets)
        # CELoss = -ln(pt) 이니깐 pt = e^(-CELoss)
        pt = torch.exp(-CELoss)
        FocalLoss = self.alpha*((1-pt)**self.gamma) * CELoss
        
        if self.mean:
            return torch.mean(FocalLoss)
        else:
            return FocalLoss

input    = torch.tensor([0.1, 0.8, 0.1]).unsqueeze(0).float()
output = torch.tensor([1]).long()

loss = FocalLoss()(input, output)
print(f'FCLoss:{loss}')

loss = nn.CrossEntropyLoss()(input, output)
print(f'CELoss:{loss}')

