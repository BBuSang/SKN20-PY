# 매개변수 o, 반환값 o
def plus(num1, num2):
    return num1 + num2
# 매개변수 o, 반환값 x
def minus(num1, num2):
    print(num1 - num2)
# 매개변수 x, 반환값 o
def home():
    return "집가고 싶다"
# 매개변수 x, 반환값 x
def callhome():
    print(home())

print(plus(10, 20))
minus(20, 10)
print(home())
callhome()



def doorman(age, money):
    if age < 20 :
        return "입장불가"
    elif money < 10000 :
        return "거지"
    else :
        return "입장가능"
    
print(doorman(25, 20000))
print(doorman(18, 20000))


