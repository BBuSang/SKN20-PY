# enumerate(), zip(), .items() .key() .values()
# map(), 정렬 --> 람다 함수를 적용
# 함수의 파라메터 - 키워드 파라메터, 가변 키워드 파라메터

list_a = ['사과', '포도', '딸기']
for i,j in enumerate(list_a):
    print(i)

# zip() 두개의 리스트 또는 집합을 각 원소별로 묶어준다
names = ['홍길동', '이순신']
ages = [25,35]
print(list(zip( names, ages )))
print(dict(zip(names,ages)))
for name, age in zip(names,ages):
    print(name , age)

# .items()
dict_1 = {'취미':"수영", '음식':"떡볶이"}
print(dict_1)
print(dict_1.keys())
print(dict_1.values())
print(dict_1.items())
