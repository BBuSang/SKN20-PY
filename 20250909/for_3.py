# remove
list_a = [1, 2, 1,2]
# list_a.remove(1)
# print(list_a)

x = []
for i in list_a:
    if i != 1:
        x.append(i)
print(x)


list_a = [1, 2, 1,2]
for i in list_a:
    if i == 1:
        list_a.remove(i)
print(list_a)

