import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {role}."),
        ("human", "{question}")
    ]
)

messages = prompt.invoke(
    {
        "role": "Doctor",
        "question": "What is constipation?."
    }
)

response = llm.invoke(messages)

print(response.content)