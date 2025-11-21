def change(obj):
    obj[0] = 100

data = [1,2,3]
change(data)
print(data)
print('_'*10)
a=10
b=a
b=1000
print(f'a={a} b={b}')
print('_'*30)

list_a=[1,2,3]
list_b=list_a.copy()
list_b[0] = 100
print(f'list_a = {list_a} list_b ={list_b}')