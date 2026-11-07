import streamlit as st
from dotenv import load_dotenv
import os

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

load_dotenv()

# ------------------------
# Page Settings
# ------------------------

st.set_page_config(
    page_title="LangChain Chatbot",
    page_icon="🤖",
)

st.title("🤖 LangChain Chatbot")

# ------------------------
# LLM
# ------------------------

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a friendly AI assistant.

Give clear, accurate and helpful answers.
Explain code in simple language.
""",
        ),

        MessagesPlaceholder(variable_name="history"),

        ("human", "{question}"),
    ]
)

chain = prompt | llm

# ------------------------
# Memory
# ------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------
# Show Chat History
# ------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------
# User Input
# ------------------------

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