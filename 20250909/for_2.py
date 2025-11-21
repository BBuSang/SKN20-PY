# 0~100 사이의 랜덤한 숫자 10개를 리스트로 저장
import random
numbers = random.sample(range(100), 10)

# 짝수만 출력
print(numbers)

# numbers데이터중에 짝수만 찾아서 새로운 리스트에 저장
# 1. 리스트를 순환한다
# 2. 순환하면서 각 데이터가 짝수인지 판단한다
# 3, 짝수이면 미리 준비한 빈 리스트에 추가한다
# 4. 모든 순환이 끝나면(for문 끝나면) 준비한 리스트를 출력하고 len()이용해서 개수도 확인한다

x = []
for i in numbers:
    if i % 2 == 0:
        x.append(i)
print(x)
print(len(x))


