# dict
# 키와 값의 쌍으로 구성 {key1:value1, key2:value2}
# 순서가 없다
# 키는 중복 안됨
# 키는 변하지 않는 자료형만 가능(문자열, 숫자, 튜플)
# CRUD 가능
# 각 요소에 접근할때는키 값을 접근(인덱스가 아님)
student = {
    "name" : "홍길동",
    "age" : 20,
    "major" : "컴퓨터"
}
# 읽기
print(student)
# 업데이트
student['name'] = '이순신'
print(student)
# 삭제
del student["name"]
# 추가
student["addr"] = "서울시"
print(student)