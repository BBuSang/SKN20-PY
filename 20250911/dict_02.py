# list, set, tuple, dict
result = dict([['name','age'],['홍길동',20]])
print(result)

# 두개의 리스트 한개는 키에 해당하는 값들의 집합
# 다른하나는 값에 해당하는 집합
# 이걸 dict구조로 만들려면
names = ['홍길동', '이순신', '강감찬']
scores = [100, 99, 98]
students = {}
# for name in names:
#     students[name] = scores[name.index]
# print(students)

for i in range(len(names)):
    students[names[i]] = scores[i]

print(names.index("홍길동"))
# 비어있는 dict 변수를 생성
# 변수['키'] = 값 형태로 생성 - 순환문을 통해서 