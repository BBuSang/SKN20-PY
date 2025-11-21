# 람다가 필요없는 상황


# def minus(a,b):
#     return a-b

def calc(func,a,b):
    return func(a,b)

print(calc(lambda a,b : a+b, 1,2))
print(calc(lambda x,y : x*y, 10,20))

