from openai import OpenAI
import streamlit as st

with st.sidebar:
    "[Documentation Link](https://www.google.com)"

st.title("💬 Cornell Movie Dialouges")
st.caption("🚀 Fine Tuned Gpt4o chatbot on Cornell Movie Dialouges")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi There! I'm trained on movie dialogues. Let's have a chat!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
import os

openai_api_key = os.environ["OPENAIKEY"]
if prompt := st.chat_input():
    client = OpenAI(api_key=openai_api_key)

    result = client.fine_tuning.jobs.list()

    # Retrieve the fine tuned model
    fine_tuned_model_checkpoint = "ft:gpt-4o-mini-2024-07-18:personal::AJWkdxWX:ckpt-step-66426"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model=fine_tuned_model_checkpoint, messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

