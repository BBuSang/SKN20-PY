# 논리 연산자 사용
# 나이가 18이상(성인) 이면서 주민번호를 가진사람은 "입장가능" "입장불가"

try:
    has_id = True
    age = int(input('나이를 입력하세요 : '))
    if age >= 18 and has_id:
        print('입장가능')
    else:
        print('입장불가능')
except Exception as e:
    print('나이는 숫자로 입력하세요')
