# 오류를 피해가는 행위 --> 예외 처리
# num1 = int(input("숫자를 입력하세요 : "))
# num2 = int(input("숫자를 입력하세요 : "))

try:
    num1, num2 = map(int, input("공백을 기준으로 두개의 숫자를 입력 : ").split())
    calc_lists = [num1+num2, num1-num2, num1*num2, num1/num2]

    print('1. 더하기', end='\t')
    print('2. 빼기', end='\t')
    print('3. 곱하기', end='\t')
    print('4. 나누기')
    select = int(input('원하는 계산을 선택하세요(1~4) : '))
    print(f'결과는 {calc_lists[select-1]} 입니다.')
except ValueError as e:
    print('오류가 발생했습니다.', e)


