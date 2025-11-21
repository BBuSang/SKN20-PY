# 클래스 생성
class MyData():
    def __init__(self):
        pass

# 클래서 객체 생성
m1 = MyData()

li = [1,2,3,4]
x = li.pop()
li.append(x)
print(li)
print(li.index(li[0]))

print(x in li)
