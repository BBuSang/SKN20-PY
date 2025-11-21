# 직원 Employee - 아이디, 이름, 기본급
class Employee:
    def __init__(self, id, name, base_salary):
        self.id = id
        self.name = name
        self._base_salary  = base_salary
    
    @property
    def base_salary(self):
        return self._base_salary
    @base_salary.setter
    def base_salary(self, money):
        if money > 0:
            self._base_salary = money
        else : 
            raise ValueError('금액은 양수입니다. 마이너스 불가')

    def __str__(self):
        return f'{self.name} : {self.id}, {self.base_salary}'
    
    def emp(self):
        print('직원 클래스')
        
# 정규직 FullTimeEmployee- 보너스
class FullTimeEmployee(Employee):
    def __init__(self, id, name, base_salary, bonus):
        super().__init__(id, name, base_salary)
        self.bonus = bonus
    def __str__(self):
        return super().__str__() +f', {self.bonus}'
    def emp(self):
        print('정규직 클래스')
# 계약직 PartTimeEmployee- 시간당 급여, 기본급 없음
class PartTimeEmployee(Employee):
    def __init__(self, id, name, hour_pay, hour):
        super().__init__(id, name, 0)
        self.hour_pay = hour_pay
        self.hour = hour
    def __str__(self):
        return f'{self.name} : {self.id}, {self.hour_pay}, {self.hour}'
    def emp(self):
        print('계약직 클래스')
# 인턴 Intern- 고정 수당
class Intern(Employee) :
    def __init__(self, id, name, pay):
        super().__init__(id, name,0)
        self.pay = pay
    def __str__(self):
        return f'{self.name} : {self.id}, {self.pay}'
    def emp(self):
        print('인턴클래스')
# 정규직 직원, 계약직, 인턴을 모두 리스트에 섞어서 객체에 저장
# emp = []

#  emp에 들어가는 직원이 각각 어떤 클래스인지 순환을 이용해서 각 클래스의 int,pte 메소드 호출

emp = [
    FullTimeEmployee(0,"홍길동",1000000,200000),
    PartTimeEmployee(1,"이순신",10000,9),
    Intern(2,"이성계",300000),
    FullTimeEmployee(3,"김길동",1000000,200000),
    PartTimeEmployee(4,"김순신",10000,9),
    Intern(5,"김성계",300000)
]
for i in emp:
    i.emp()
    print(i)
    
