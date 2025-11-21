# 딕셔너리
    # .item() .key() .value()
# 정렬
    # sorted()
# max
    # max()
# enumerate
    # 순환문에서 리스트를 감싸면 (인덱스, 리스트의 값)
# zip()
    # 여러개의 iterable 들을 각 원소를 쌍으로 하는 집합
    # (1,2), ('사과','배')
    # [(1,'사과'),(2,'배')]
# map()
    # iterable한 객체의 각 요소에 특정 함수를 적용
    # map(int, ['1','2']) -> [1,2]

import collections
datas = [1,1,1,2,3,1,2,1,1]
print(collections.Counter(datas))

del datas[0]
print(datas)