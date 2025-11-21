list_a = [1,2,3]
list_b = [4,5,6]
last_name = '홍'
first_name = '길동'
# 리스트 연산
print(f'list_a+list_b = {list_a+list_b}')
print(f'list_a*2 = {list_a*2}')

print(f'last_name + first_name = {last_name + first_name}')
print(f'last_name*2 = {last_name*2}')

list_c = [list_a[0]+list_b[0], list_a[1]+list_b[1], list_a[2]+list_b[2]]
print(f'list_c = {list_c}')

print()

list_a.append(4)
list_a.append(5)
print(f'list_a = {list_a}')
print()

list_a.insert(0, 10)
print(f'list_a = {list_a}')

print()
list_a = [1,2,3]
list_a.extend([4,5,6])
print(f'list_a = {list_a}')

list_a = [1,2,3]
list_a += [4,5,6]
print(f'list_a = {list_a}')
