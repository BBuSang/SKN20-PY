# 사용자 입력 처리 함수
# 이름 get_data()
# 파라메터
    # start : 시작값
    # end : 종료값
    # input_str : 가이드 문구
# 사용자 입력은 input()
# return 정수로 변환된 입력값


# while True:
#     try:
#         h_num = input("1부터 100까지 수 중 한개 입력 : ") 
#         if not 0 <= h_num <=100:
#             raise ValueError("1~100 범위 초과")
#     except:
#         continue
#     else :
#         break


def get_data():
    try:
        h_num = int(input("1부터 100까지 수 중 한개 입력 : "))
        if not 0 <= h_num <=100:
            raise ValueError("1~100 범위 초과")
        return h_num
    except:
        print("다시 입력하세요")
        return get_data()
    
print(get_data())
