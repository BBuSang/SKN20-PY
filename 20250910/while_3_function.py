import time
count = 0 # 0부터 시작
is_continue = True # 계속할지 여부

def timer(a):
    a += 1
    print(f'{a} 초')
    time.sleep(1)
    return a

def tobecontinued(count):
    if count % 5 == 0:
        answer = input("To be continued? (y/n): ")
        if answer.lower() == 'n':
            print("프로그램을 종료합니다.")
            return False
    return True

count = 0
while is_continue:
    count = timer(count)
    is_continue = tobecontinued(count)


    