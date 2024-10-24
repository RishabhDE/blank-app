import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Force the use of CPU
device = torch.device("cpu")

# Load your fine-tuned model from Hugging Face
model_path = "mniazm/t5cornel150k"  # Replace with your model's path or Hugging Face repo
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.to(device)

# Streamlit UI
st.title("💬 Hugging Face Fine-tuned T5 Chatbot")
st.caption("🚀 Chat with your Hugging Face fine-tuned T5 model!")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you today?"}]

# Display chat messages from history on the Streamlit app
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to generate response from the model
def generate_response_with_history(conversation_history, max_length=50,
                                   temperature=0.95, num_beams=5, top_k=50, top_p=0.85, rep_penalty=5.0):

    # Concatenate the dialogue exchanges without speaker tag
    input_text = " ".join([text.split(": ")[1] for text in conversation_history])

    # Tokenize the concatenated input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate a response without gradient calculation
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            do_sample=(temperature != 1.0)
        )

    # Decode output tokens to return human readable string without special token
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Chat input box
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Get conversation history
    conversation_history = [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages if msg['role'] == "user"]

    # Generate response from the model
    bot_response = generate_response_with_history(conversation_history)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.chat_message("assistant").write(bot_response)
