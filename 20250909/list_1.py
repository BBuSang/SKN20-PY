age = 25
array = [273, 23.3, '안녕하세요', True, age]
print(array)
array2 = [100, array]
print(array2)
print(array2[1][2])

temp = [[1,2],   # row 행
        [10,20], # temp[1][1] 
        [30,40]
        ]

list_a = [1,[1,2,[1,2,3,[1,2,3,4,[1,2,3,4,5,[1,2,3,4,5,6]]]]]]
print(list_a[1][2][3][4][5][2])

list_b = [1,2,'문자열', True, False]
print(list_b[2][2])
print(list_b[:])
print(list_b[:3])
print(list_b[3:])
print(list_b[-2:])
# start index : end index : -1 : 1
print(list_b[::2])
print(list_b[::-1])