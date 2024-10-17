import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned DialoGPT model and tokenizer
model_path = "/workspaces/blank-app/fine-tuned-dialoGPT"  # Replace with the path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

with st.sidebar:
    "[Documentation Link](https://www.google.com)"

st.title("💬 Cornell Movie Dialogues")
st.caption("🚀 A Streamlit chatbot powered by your fine-tuned DialoGPT")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to generate a response using your DialoGPT model
def generate_response(prompt):
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response_text

# Chat input handling
if prompt := st.chat_input():
    # Add the user message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate a response using the fine-tuned model
    response = generate_response(prompt)

    # Add the model's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
