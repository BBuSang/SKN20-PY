# 가위바위보 게임을 클래스로 구현하기
# 사용자로부터 입력을받아 컴퓨터와 대결하는 간단한 가위 바위 보 게임을 구하기
# 사용자는 "가위", "바위", "보" 중 하나를 입력하고, 컴퓨터는 무작위로 선택
# 게임의 승패를판단하고 결과를출력합니다
# 가위는 1, 바위는 2, 보는 3으로 표현합니다
# 게임이 끝나면 계속할지 물어본다
import random
class RPSGame :
    def __init__(self):
        self.choices = {1: "가위", 2: "바위", 3: "보"}
    
    def get_user_choice(self):
        while True:
            try:
                user_input = int(input("가위(1), 바위(2), 보(3) 중 하나를 선택하세요: "))
                if user_input in self.choices:
                    return user_input
                else:
                    print("잘못된 입력입니다. 1, 2, 3 중 하나를 선택하세요.")
            except ValueError:
                print("숫자를 입력하세요.")
    
    def get_computer_choice(self):
        return random.randint(1, 3)
    
    def determine_winner(self, user_choice, computer_choice):
        if user_choice == computer_choice:
            return "무승부"
        elif (user_choice == 1 and computer_choice == 3) or (user_choice == 2 and computer_choice == 1) or (user_choice == 3 and computer_choice == 2):
            return "사용자 승리"
        else:
            return "컴퓨터 승리"
    def play(self):
        while True:
            user_choice = self.get_user_choice()
            computer_choice = self.get_computer_choice()
            print(f"사용자 선택: {self.choices[user_choice]}, 컴퓨터 선택: {self.choices[computer_choice]}")
            result = self.determine_winner(user_choice, computer_choice)
            print(f"결과: {result}")
            play_again = input("다시 하시겠습니까? (y/n): ")
            if play_again.lower() != 'y':
                print("게임을 종료합니다.")
                break

RPSGame().play()