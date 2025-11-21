import time

count = 0 # 0부터 시작
is_continue = True # 계속할지 여부
while is_continue: # 무한반복
    count += 1 # 1씩 증가
    print(f'{count} 초') # 1,2,3,4,5...
    time.sleep(1) # 1초 대기
    # 1초 단위로 계속 출력
    
    # 5초 단위로 사용자한테 계속 할건지 물어본다.. "To be continued? (y/n): "
    if count % 5 == 0: # 5의 배수일때
        answer = input("To be continued? (y/n): ") # 사용자한테 물어본다
        if answer.lower() == 'n': # 소문자로 바꿔서 n인지 확인
            print("프로그램을 종료합니다.") 
            is_continue = False # while문(반복문) 종료
