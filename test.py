from collections import deque
import time
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