import streamlit as st
from backend.handlers import handle_user_query


if "messages" not in st.session_state:
    st.session_state.messages = []

st.image("logo.png")
container = st.container()

if prompt := st.chat_input("Enter the matrix..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        st.session_state.messages = handle_user_query(
            prompt, messages=st.session_state.messages
        )

    with st.chat_message("assistant"):
        st.markdown(st.session_state.messages[-1].content)

if st.session_state.messages:
    with container.expander("Show full conversation"):
        for message in st.session_state.messages:
            with st.chat_message(message.role):
                if message.content:
                    st.markdown(message.content)
                if message.tool_calls:
                    st.write([message.tool_calls])
