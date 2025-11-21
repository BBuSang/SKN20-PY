# 함수
# 함수명 add
# 파라메터는 2개 op1, op2
# 결과를 변환하다

def add(op1, op2):
    return op1 + op2

# 사용
print( add(10, 20) )
two_number_hab = add(10,30)
numbers = [add(10,2),add(100,20)]

# 매개변수 받아서 출력하는함수
# 함수명 : show_number
# 매개변수명 : data
def show_number(data):
    print(f'numbers = {data}')


show_number(add(10,2))


def lenFunc(word):
    return len(word)

print(lenFunc("안녕하세요"))




def firstword(word):
    return word[0]

print(firstword("안녕하세요"))

a=4 
b =7
print((a-1)^(b-1))

