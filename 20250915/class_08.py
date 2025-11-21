# 학생 클래스 생성
# 인스턴스 변수  : 이름 국 영 수
# 인스턴스 메서드 : 총점, 평균, 학점, __str__
class Student :
    def __init__(self, name, kor, eng, mat):
        self.name = name
        self.kor = kor
        self.eng = eng
        self.mat = mat

        def total(self):
            return kor + eng + mat
        def avg(self):
            return total() / 3
        def grade(self):
            if avg() >= 90 :
                return 'A'
            elif avg() >= 80 :
                return 'B'
            elif avg() >= 70 :
                return 'C'
            elif avg() >= 60 :
                return 'D'
            else :
                return 'F'
            
        def to_string(self):
            return f'이름 : {self.name}, 총점 : {self.total()}, 평균 : {self.avg()}, 학점 : {self.grade()}'

# 인스턴스 생성
s1 = Student("홍길동", 90, 80, 70)
s2 = Student("이순신", 85, 95, 80)
print(s1)
print(s2)