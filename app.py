import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Handle API Key (Works locally and on the cloud)
load_dotenv() # Looks for .env locally
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.error("API Key not found. Please set GROQ_API_KEY in secrets or .env")
    st.stop()

# --- UI Configuration ---
st.set_page_config(page_title="Witty Groq Bot", page_icon="🤖")
st.title("🤖 Groq Chatbot")
st.caption("Smart, fast, and remarkably sarcastic.")

# --- Initialize Model ---
@st.cache_resource
def get_model(api_key):
    return ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.9, 
        groq_api_key=api_key
    )

model = get_model(api_key)

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant that answers questions with sensible humor and a touch of sarcasm. You have a witty personality and enjoy making jokes while providing information. Your responses should be concise, informative, and entertaining. Always try to include a clever remark or pun related to the topic at hand.")
    ]

# --- Display Chat History ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="✨"):
            st.markdown(msg.content)

# --- User Input ---
if prompt := st.chat_input("Ask me something..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Processing..."):
            try:
                response = model.invoke(st.session_state.messages)
                st.markdown(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"Error: {e}")