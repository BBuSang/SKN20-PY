# pip install streamlit - cmd에서 실행
import streamlit as st
st.title('타이틀')
st.header('헤더')
st.subheader('서브헤더')
st.button("버튼")
st.checkbox('체크박스')
st.radio("레디오박스",(1,2,3,4,5))
st.selectbox("셀렉트박스", ('일번','이번'))
st.slider('슬라이더',1,10,5)
st.text_area("텍스트")


name = st.text_input("이름을 입력하세요")
st.write(f'안녕하세요 {name}님!!')

