import os
from dotenv import load_dotenv

load_dotenv()  # picks up a .env file in this folder, if present

from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Reads NVIDIA_API_KEY from the environment (or a .env file) first; falls
# back to the literal key below so this still runs if neither is set.
api_key = os.getenv("NVIDIA_API_KEY")

client = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=api_key,
    temperature=1,
    top_p=0.95,
    max_completion_tokens=16384,  # max_tokens is deprecated in this SDK version
    reasoning_budget=16384,
    chat_template_kwargs={"enable_thinking": True},
    timeout=180,  # this model can reason for a while; the 60s default can time out
)

try:
    for chunk in client.stream(
        [{"role": "user", "content": "Explain Artificial Intelligence to a 10-year-old in simple words."}]
    ):
        if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
            print(chunk.additional_kwargs["reasoning_content"], end="")

        print(chunk.content, end="")
except Exception as e:
    # With enable_thinking=True this model can occasionally emit its reasoning
    # as plain content instead of routing it through reasoning_content, and can
    # run long enough to hit a read timeout even at 180s. If that happens here,
    # either drop enable_thinking to False for a shorter, direct-only answer,
    # or bump `timeout` further.
    print(f"\n[stream interrupted: {type(e).__name__}: {e}]")
