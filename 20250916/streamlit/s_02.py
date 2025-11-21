import streamlit as st
import random as r

# 컴퓨터 숫자 선택
if 'com' not in st.session_state:
    st.session_state.com = r.randint(1,100)
# 숫자 입력 1~100 사이
h_num = st.number_input('숫자입력', 1, 100)
if st.button('결과 확인'):
    st.write(st.session_state.com)
    if st.session_state.com > h_num:
        st.write('up')
    elif st.session_state.com < h_num :
        st.write('down')
    else:
        st.write('컴퓨터 숫자 : ', st.session_state.com)
        st.write("정답")

