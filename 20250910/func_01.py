# import random
# result = random.randint(1, 5)

# 함수정의 def 키워드 사용
# 매개변수 (parameter) : 함수가 전달받는 값
# 인자 (argument) : 함수를 호출할때 전달하는 값
# 반환값 (return value) : 함수가 작업을 마치고 호출한 곳으로 돌려주는 값
    #   return 키워드 사용

# 함수의 구성요소
def myCalc(num1, num2):
    '''
    두 개의 값을 받아서 더하는 기능
    num1는 숫자
    num2는 숫자
    '''
    result = num1 + num2
    return result

# 1. 매개변수와 반환값이 없는 함수
def SayHello():
    print("안녕하세요...")

# 2. 매개변수가 있고 반환값이 없는 함수
def SayHelloName(name):
    print(f"{name}님 안녕하세요... ")

# 3. 매개변수가 없고 반환값이 있는 함수
import datetime
def getCurrentTime():
    return datetime.datetime.now()

print(myCalc(10, 20))
SayHello()
SayHelloName("홍길동")
print(getCurrentTime())

