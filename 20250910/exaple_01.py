# 가위 바위 보 게임 (컴퓨터 vs 휴먼)
# 가위 0 바위 1 보 2
# 규칙 : 컴퓨터가 임의로 숫자를 선택
# 인간이 숫자를 입력
# 승패를 기록
# 3번마다 계속할지 물어본다
import random
import time

result = ""
counting1 = 0
win = 0
lose = 0
def raises(sec, word=""):
    for i in range(sec):
        time.sleep(1)
        print(word)

def bettle(com, user):
    if com == 0 and user == 1:
        return "승리"
    elif com == 1 and user == 2:
        return "승리"
    elif com == 2 and user == 0:
        return "승리"
    elif com == 0 and user == 2:
        return "패배"
    elif com == 1 and user == 0:
        return "패배"
    elif com == 2 and user == 1:
        return "패배"
    else:
        return "비김"

def counting(result):
    global counting1, win, lose
    counting1 += 1
    if result == "승리":
        win += 1
        print(f"승리!\n승리: {win} / 패배: {lose} / 횟수: {counting1}")
    elif result == "패배":
        lose += 1
        print(f"패배!\n승리: {win} / 패배: {lose} / 횟수: {counting1}")
    else:
        print(f"비겼습니다.\n승리: {win} / 패배: {lose} / 횟수: {counting1}")


while True:
    print("\n\n---------가위 바위 보---------")
    computer = random.choice([0, 1, 2])
    try:
        user = int(input("0. 가위\n1. 바위\n2. 보\n입력 : "))
    except ValueError:
        print("숫자를 입력해주세요.")
        continue
    if user not in [0, 1, 2]:
        print("다시 선택해주세요 0 or 1 or 2")
        continue
    
    raises(3,".")

    print(f'결과! computer {computer} vs 유저 {user}')
    result = bettle(computer, user)
    counting(result)

    if counting1 % 3 == 0:
        answer = input("continue? (y:n) : ")
        if answer.lower() =="n":
            break
        else :
            continue

    raises(2)
