# 가변 매개변수
    # 함수정의 할때, 매개변수의 개수를 지정하지 않습니다.
    # 함수내부에서는 리스트로 간주합니다
    # 함수를 호출하는 쪽에서는 편안하게.. 1,2,3,4 or ,1,2,3,1,4,2,4

def myFunc1(*args):
    for i in args:
        print(i)
        

myFunc1(10,20,50,60)

def myFunc2(args):
    for i in args:
        print(i)

myFunc2([10,20,50,60])

