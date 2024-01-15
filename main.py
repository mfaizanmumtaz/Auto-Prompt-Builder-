
import streamlit as st

st.set_page_config(page_title="Auto Prompt Builder", page_icon=":rocket:")

st.title("AutoPrompt Builder :rocket:")

user_input = st.text_input("Enter your objective:")
submit_button = st.button("Submit")

if submit_button:
    with st.spinner("Generating prompt..."):

        from app import retrieval_chain
        output = retrieval_chain.invoke({"objective":user_input})

        st.success("Prompt generated!")

        with st.expander("Prompt"):

            st.write(output)

