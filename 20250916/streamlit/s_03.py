import streamlit as st
if 'count' not in st.session_state:
    st.session_state.count = 0

if st.button("카운트 증가"):
    st.session_state.count+=1
st.write(st.session_state.count)
st.json(st.session_state)
