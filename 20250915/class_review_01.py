# 기본 클래스 생성
class Review :
    # 클래스 변수 생성
    review_count = 0
    # 생성자 메소드
    def __init__(self, name=""):
        self.name = name

# 인스턴스 생성
r1 = Review('홍길동')
r2 = Review('ㅇㅇㅇ')

# 인스턴스 변수 변경
r1.name = "첫번쨰 리뷰"
print(f'r1 인스턴스 변수 : {r1.name} / r2 인스턴스 변수 : {r2.name}')
print(f'클래스 변수 : {Review.review_count} / r1 클래스 변수 : {r1.review_count} / r2 클래스 변수 : {r2.review_count}')
