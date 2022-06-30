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


# import random

# num= random.random()

# print(str(num)[2:])

def solution(n, arr1, arr2):
    answer = []

    for a1, a2 in zip(arr1, arr2):
        cnt = 0
        result= ""
        for k in range(n):
            true = int(tenTotwo(a1, n)[cnt]) or int(tenTotwo(a2, n)[cnt])
            if true ==1:
                result+="#"
            else:
                result+=" "
            cnt+=1
        answer.append(result)
        print(answer)

    return answer

def tenTotwo(input, num):
    return bin(input)[2:].zfill(num)

# 정수를 저장한 배열, arr 에서 가장 작은 수를 제거한 배열을 리턴하는 함수, solution을 완성해주세요. 단, 리턴하려는 배열이 빈 배열인 경우엔 배열에 -1을 채워 리턴하세요. 예를들어 arr이 [4,3,2,1]인 경우는 [4,3,2]를 리턴 하고, [10]면 [-1]을 리턴 합니다.

# 제한 조건
# arr은 길이 1 이상인 배열입니다.
# 인덱스 i, j에 대해 i ≠ j이면 arr[i] ≠ arr[j] 입니다.

arr1 = [1,2,3,4,5]

arr2 = [5,2,6,2,3]

solution(5, arr1, arr2)