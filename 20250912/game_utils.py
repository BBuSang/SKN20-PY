log = {}
def is_right(human, computer, count):
    if human > computer:
        print("크다")
        log[human] = "크다"
        return True
    elif human < computer:
        print("작다")
        log[human] = "작다"
        return True
    else :
        log[human] = "정답"
        print(f"정답 count : {count}\n{log}")
        return False
    
def get_data(start,end,input_str='입력'):
    while True:
        try:
            h_num = int(input(f'{input_str}({start}~{end}) '))
            if not start <= h_num <= end:
                raise ValueError(f'{start} ~ {end} 범위 초과')        
        except Exception as e:
            print(f'오류 : {e}')
        else:
            return h_num