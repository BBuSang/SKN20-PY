# 파이썬 클래스에서 getter setter 사용법
class Person:
    def __init__(self, name, age):
        self._name = name  # 이름은 읽기 전용
        self._age = age    # 나이는 읽기/쓰기 가능
    
    # 데코레이터를 이용한 setter
    @property
    def name(self):
        return self._name

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        if value < 0:
            raise ValueError("나이는 음수가 될 수 없습니다.")
        self._age = value
    
p1 = Person("홍길동", 25)
print(p1.name)  # 홍길동
print(p1.age)   # 25
p1.age = 30
print(p1.age)   # 30