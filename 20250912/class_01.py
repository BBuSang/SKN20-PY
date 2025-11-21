# 클래스 변수 VS 인스턴스 변수

class StudentMng() :
    name = '홍길동' # 클래스 변수
    def make_instance(self):
        self.age = 0
        self.addr = 0

s1 = StudentMng()
s2 = StudentMng()
s3 = StudentMng()

s1.name="이순신"
StudentMng.name = "강감찬"
print(s1.name, s2.name, s3.name)

# 클래스 변수는 모든 객체가 참조되는 변수
# but 객체가 변수를 재 할당 받으면해당 객체는 더이상 참조하지 않는다
