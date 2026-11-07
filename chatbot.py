import os
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

# ----------------------------------------------------
# Create LLM
# ----------------------------------------------------

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=api_key,
)

# ----------------------------------------------------
# Prompt Template
# ----------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a friendly AI assistant.

Give accurate, concise and helpful answers.

If the user asks coding questions,
explain them in very simple language.
""",
        ),

        MessagesPlaceholder(variable_name="history"),

        (
            "human",
            "{question}",
        ),
    ]
)

# ----------------------------------------------------
# LCEL Chain
# ----------------------------------------------------

chain = prompt | llm

# ----------------------------------------------------
# Conversation Memory
# ----------------------------------------------------

conversation = []

# ----------------------------------------------------
# Chat Loop
# ----------------------------------------------------

print("=" * 50)
print("🤖 LangChain Chatbot (Version 3)")
print("Type 'exit' to quit.")
print("=" * 50)

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("\nGoodbye 👋")
        break

    print("\nAI: ", end="", flush=True)

    full_response = ""

    try:

        for chunk in chain.stream(
            {
                "history": conversation,
                "question": question,
            }
        ):

            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content

        print()

        conversation.append(
            HumanMessage(content=question)
        )

        conversation.append(
            AIMessage(content=full_response)
        )

    except Exception as e:

        print(f"\n\nError : {e}")