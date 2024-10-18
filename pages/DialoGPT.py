import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Force the use of CPU
device = "cpu"

# Load your fine-tuned model and tokenizer
model_path = "fine-tuned-dialoGPT"  # Replace with the actual path to your model
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model using standard precision for CPU
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
model.to(device)

# Streamlit UI
st.title("💬 Fine-tuned DialoGPT Chatbot")
st.caption("🚀 Chat with your fine-tuned model!")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages from history on the Streamlit app
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to generate response from the model
def generate_response(prompt):
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt").to(device)
    
    # Generate model response
    response_ids = model.generate(
        input_ids,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
    )
    response_text = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response_text

# Chat input box
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Get response from the model
    bot_response = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.chat_message("assistant").write(bot_response)
