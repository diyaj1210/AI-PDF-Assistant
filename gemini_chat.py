from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os 
import google.generativeai as genai

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

## function ot load gemini pro model and get response 

model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat()

def get_gemini_response(question):
    response = chat.send_message(question)
    return response.text


# initialise our streamlit app

st.set_page_config(page_title = "Gemini Q&A")
st.title("Ask Gemini Based Chatbot Anything")

# initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask your Question: ")
submit = st.button("Submit")

if submit and question:
    reply = get_gemini_response(question)

    # Add user query and response to session chat history
    st.session_state.chat_history.append(("You", question))
    st.session_state.chat_history.append(("Gemini", reply))

    st.markdown("**Gemini says:**")
    st.write(reply)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("**CHAT HISTORY**")
for speaker, msg in st.session_state.chat_history:
    st.write(f"**{speaker}**: {msg}")
    