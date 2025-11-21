# 집합 연산이 가능
import random
list_a = random.sample(range(10),6) #(1,2,3,4)
list_b = [1,5,4,1,2,1,1,1,1,5,7,3]
find_list = []
for a in list_a : 
    for b in list_b: 
        if a == b and a not in find_list:
            find_list.append(a)

print(list_a)
print(list_b)
print(find_list)

