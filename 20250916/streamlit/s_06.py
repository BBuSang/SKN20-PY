import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="Streamlit 레이아웃 예제",
    page_icon=":heart_eyes:",
    layout="wide"  # 전체 페이지를 wide 모드로 설정
)

# 사이드바 메뉴 생성
with st.sidebar:
    st.title("게임")
    selected_menu = st.radio(
        "원하시는 게임을 선택하세요:",
        ["숫자맞추기", "가위바위보"]
    )

# 메인 컨텐츠 영역
import random
def show_game1():
    st.header("숫자맞추기")
    st.write("환영합니다! 이곳은 숫자맞추기 게임 페이지입니다.")

    if 'c_number' not in st.session_state:
        st.session_state.c_number = random.randint(1, 100)    
    c_num = st.session_state.c_number        
    h_number = st.number_input("1에서 100 사이의 숫자를 입력하세요:", 1, 100)
    
    if st.button("확인"):
        if h_number < c_num:
            st.warning("예측한 값이 낮습니다.")
        elif h_number > c_num:
            st.warning("예측한 값이 높습니다.")
        else:
            st.balloons()
            st.success(f"정답! {c_num}였습니다.")
            del st.session_state.c_number    


def show_game2():
    st.header("가위바위보")
    
    # 승패 카운트 초기화
    if 'win_count' not in st.session_state:
        st.session_state.win_count = 0
    if 'lose_count' not in st.session_state:
        st.session_state.lose_count = 0
        
    # 현재 승패 상황 표시
    st.write(f'승리: {st.session_state.win_count}번')
    st.write(f'패배: {st.session_state.lose_count}번')
    
    # 버튼 3개 가로로 배치
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scissors = st.button("가위")
    with col2:
        rock = st.button("바위")
    with col3:
        paper = st.button("보")
    
    # 버튼 클릭시 게임 진행
    if scissors or rock or paper:
        computer = random.choice(["가위", "바위", "보"])
        player = "가위" if scissors else "바위" if rock else "보"
        
        st.write(f"컴퓨터: {computer}")
        st.write(f"플레이어: {player}")
        
        # 승패 판정
        if player == computer:
            st.write("비겼습니다!")
        elif ((player == "가위" and computer == "보") or 
              (player == "바위" and computer == "가위") or 
              (player == "보" and computer == "바위")):
            st.write("이겼습니다! 🎉")
            st.session_state.win_count += 1
        else:
            st.write("졌습니다... 😢")
            st.session_state.lose_count += 1
# 선택된 메뉴에 따라 해당하는 컨텐츠 표시
if selected_menu == "숫자맞추기":
    show_game1()
elif selected_menu == "가위바위보":
    show_game2()
