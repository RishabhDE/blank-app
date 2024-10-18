from openai import OpenAI
import streamlit as st

with st.sidebar:
    "[Documentation Link](https://www.google.com)"

st.title("💬 Cornell Movie Dialouges")
st.caption("🚀 A Streamlit chatbot powered by GPT 4o Mini")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
import os

openai_api_key = os.environ["OPENAIKEY"]
print(openai_api_key)  # Use your secret as needed
if prompt := st.chat_input():
    client = OpenAI(api_key=openai_api_key)

    result = client.fine_tuning.jobs.list()

    # Retrieve the fine tuned model
    fine_tuned_model = result.data[0].fine_tuned_model
    print(fine_tuned_model)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model=fine_tuned_model, messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

