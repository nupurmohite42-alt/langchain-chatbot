import os

import streamlit as st
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------
load_dotenv()

# First try local .env
api_key = os.getenv("NVIDIA_API_KEY")

# If deployed on Streamlit Cloud, use Secrets
if not api_key:
    api_key = st.secrets.get("NVIDIA_API_KEY")

# ----------------------------------------------------
# Page Settings
# ----------------------------------------------------
st.set_page_config(
    page_title="LangChain Chatbot",
    page_icon="🤖",
)

st.title("🤖 LangChain Chatbot")
st.caption("Built using LangChain + NVIDIA NIM + Streamlit")

# ----------------------------------------------------
# Check API Key
# ----------------------------------------------------
if not api_key:
    st.error(
        "❌ NVIDIA API Key not found.\n\n"
        "Local: create a .env file.\n"
        "Streamlit Cloud: add NVIDIA_API_KEY in Secrets."
    )
    st.stop()

# ----------------------------------------------------
# LLM
# ----------------------------------------------------
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=api_key,
)

# ----------------------------------------------------
# Prompt Template (LCEL)
# ----------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a friendly AI assistant.

Give clear, accurate and concise answers.

Explain code in simple language whenever possible.
""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm

# ----------------------------------------------------
# Memory
# ----------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------
# Display Previous Messages
# ----------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------------------------------
# Chat Input
# ----------------------------------------------------
question = st.chat_input("Ask me anything...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        try:

            for chunk in chain.stream(
                {
                    "history": st.session_state.history,
                    "question": question,
                }
            ):

                if chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except Exception as e:

            st.error(f"Error: {e}")
            st.stop()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
        }
    )

    st.session_state.history.append(
        HumanMessage(content=question)
    )

    st.session_state.history.append(
        AIMessage(content=full_response)
    )