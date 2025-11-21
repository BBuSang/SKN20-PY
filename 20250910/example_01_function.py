import games

for i in range(1,101):
    if i % 3 == 0:
        is_continue = input('계속하시겠습니까?(Y/y)').upper()
        if not is_continue == 'Y':
            break
    # 랜덤하게 선택한 컴퓨터의 값
    com_choince = games.get_com_num()
    # 사용자의 값
    human_choice = games.get_human_num()
    # 승패 확인
    # print(com_choince)   # 디버깅용.. 개발이 완료되면 삭제
    games.check_winner(com_choince,human_choice)