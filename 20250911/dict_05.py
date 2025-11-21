# 선거시스템
# 유권자 들은 기호로 투표를 ㅣㅈㄴ행
# ex 1, 2, 3
# 투표는 순환문을 이용해서 유권자가 10명이라면 10번순환하면서 후보자(1~5)번호 선택
# [1,1,2,3,4,1,5,1]
# 리스트에 있는 각 번호의 횟수를 구해서 당선자를 출력
cadidate = ["홍길동","이순신","강감찬","율곡", '신사임당']
vote = [0,0,0,0,0]
counting = {}
counts=10

for i in range(counts):
    choice = int(input('투표를 하세요 (1~5) : '))
    
    if choice == 1 :
        vote[0] += 1
    elif choice == 2:
        vote[1] +=1
    elif choice == 3:
        vote[2] +=1
    elif choice == 4:
        vote[3] +=1
    elif choice == 5:
        vote[4] +=1
    else :
        pass
    
    for j in range(len(cadidate)):
        counting[cadidate[j]] = vote[j]

    winner = vote.index(max(vote))
print(f'{counting}\n당선 : {cadidate[winner]}')

# 키의 값을 가져올때.. 딕셔너리변수['키값']
# 딕셔너리변수.get('키값')


