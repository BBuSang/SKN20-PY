# isinstance() 함수
# 객체가 특정 클래스으 인스턴스(객체)인지 확인 하는 데 사용됩니다.
# 사용하는 이유
# 1. 타입 확인 : 함수나 메서드가 특정 클래스의 인스턴스를 기대할때, 이를 확인하여
# 2. 다형성 지원 : 상속 관계에 있는 클래스들 간에 공통된 인터페이스를 제공할 떄, 
class Student :
    def study(self):
        return "공부하는중"
class Teacher :
    def teach(self):
        return "가르치는 중"
    
# 리스트에 어떤 객체가 있는지 모를때 특정 인스턴스만 기대할 수 없다
people = [Student(), Teacher(), Student()]
del people[0]
if isinstance(people[0], Student):
    print(people[0].study())
else : 
    print(people[0].teach())