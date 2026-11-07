"""
lcel_explore.py
================
A hands-on tour of LangChain's LCEL (LangChain Expression Language).

LCEL lets you compose LangChain building blocks (prompts, models, parsers,
retrievers, ...) with the `|` pipe operator, the same way you'd pipe shell
commands together. Every piece is a "Runnable", and piping two Runnables
together builds a new Runnable you can `.invoke()`, `.batch()`, or
`.stream()`.

This file is meant to be read top-to-bottom, then run and poked at. Each
section is a self-contained demo function; run them one at a time from
`main()` at the bottom (or just call them from a REPL).

SETUP
-----
    pip install langchain langchain-core langchain-nvidia-ai-endpoints python-dotenv

This demo is wired to NVIDIA's hosted Nemotron model (build.nvidia.com),
matching the connection already used in nvidia.py in this folder. The key
is read from the NVIDIA_API_KEY environment variable via a .env file — put
    NVIDIA_API_KEY=nvapi-...
in a .env file next to this script (same folder as nvidia.py). Never hardcode
the key directly in a .py file, especially one you might share or commit.

Swap in a different provider by changing `get_model()` below — see the
comments there for OpenAI / Anthropic / Azure OpenAI / local Ollama
alternatives.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # picks up a .env file in this folder, if present

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 0. THE MODEL
# ---------------------------------------------------------------------------
def get_model(temperature: float = 0.7):
    """
    Returns a chat model Runnable. Swap the implementation to change provider
    without touching any of the chains below — that's the point of LCEL:
    every piece is interchangeable as long as it's a Runnable.

    Default: NVIDIA's hosted Nemotron model, same one used in nvidia.py.
    """
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    # Hardcoded fallback so this runs even without a .env file / env var set.
    # (Same key used in nvidia.py. Fine for local scratch work — just don't
    # commit this file to git or hand it to anyone else with the key left in.)
    api_key = os.getenv("NVIDIA_API_KEY")

    return ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        api_key=api_key,
        temperature=temperature,
        top_p=0.95,
        max_completion_tokens=1024,  # plenty for these short demos; keeps latency down
        timeout=180,  # this model can run long on reasoning; default (60s) is too tight
        # This Nemotron model reasons by default and can leak its chain-of-thought
        # ("Here's a thinking process: ...") straight into the answer text, which
        # breaks the JSON parser demo. Turn reasoning off for clean, direct output.
        model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )

    # --- OpenAI -----------------------------------------------------------
    # from langchain_openai import ChatOpenAI
    # return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    # --- Anthropic ----------------------------------------------------------
    # pip install langchain-anthropic
    # from langchain_anthropic import ChatAnthropic
    # return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature)

    # --- Azure OpenAI -------------------------------------------------
    # from langchain_openai import AzureChatOpenAI
    # return AzureChatOpenAI(azure_deployment="your-deployment", api_version="2024-05-01-preview")

    # --- Local model via Ollama (no API key needed) -----------------------
    # pip install langchain-ollama
    # from langchain_ollama import ChatOllama
    # return ChatOllama(model="llama3.1", temperature=temperature)


# ---------------------------------------------------------------------------
# 1. THE SIMPLEST LCEL CHAIN: prompt | model | parser
# ---------------------------------------------------------------------------
def demo_basic_chain():
    """
    The canonical three-link LCEL chain:
      - PromptTemplate fills in a template with variables
      - the model turns the prompt into a response
      - StrOutputParser extracts the plain string from the model's message
    """
    print("\n=== 1. Basic LCEL chain ===")

    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in exactly two sentences, for a beginner."
    )
    model = get_model()
    parser = StrOutputParser()

    chain = prompt | model | parser  # <-- this pipe IS LCEL

    result = chain.invoke({"topic": "LangChain's LCEL"})
    print(result)
    return chain


# ---------------------------------------------------------------------------
# 2. EXPLORING PROMPTS: templates, roles, few-shot, partials
# ---------------------------------------------------------------------------
def demo_prompt_variations():
    """
    A few different ways to build prompts before piping them into a model.
    """
    print("\n=== 2. Prompt variations ===")

    # (a) Plain string template (single input)
    simple = PromptTemplate.from_template("Give me a fun fact about {animal}.")
    print("(a) rendered ->", simple.invoke({"animal": "octopuses"}).to_string())

    # (b) Chat-style template with system + human roles
    chat = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a terse {persona}. Answer in one short sentence."),
            ("human", "{question}"),
        ]
    )
    print(
        "(b) rendered ->",
        chat.invoke(
            {"persona": "senior Python engineer", "question": "What is a decorator?"}
        ).to_messages(),
    )

    # (c) Partial variables — bind some inputs now, fill the rest later
    partial = chat.partial(persona="patient teacher")
    print(
        "(c) partial + invoke ->",
        partial.invoke({"question": "What is LCEL?"}).to_messages(),
    )

    return chat


# ---------------------------------------------------------------------------
# 3. RunnableParallel: run several chains at once, fan-in the results
# ---------------------------------------------------------------------------
def demo_parallel_chain():
    """
    RunnableParallel runs multiple Runnables against the same input and
    returns a dict of their results. Handy for "generate several things
    about X in one call" or for building context before a final prompt.
    """
    print("\n=== 3. RunnableParallel ===")

    model = get_model()
    parser = StrOutputParser()

    summary_chain = (
        ChatPromptTemplate.from_template("Summarize {topic} in one sentence.")
        | model
        | parser
    )
    question_chain = (
        ChatPromptTemplate.from_template("Write one quiz question about {topic}.")
        | model
        | parser
    )

    parallel = RunnableParallel(summary=summary_chain, question=question_chain)
    result = parallel.invoke({"topic": "vector databases"})
    print(result)
    return parallel


# ---------------------------------------------------------------------------
# 4. Multi-step LCEL: chain the output of one chain into the next
# ---------------------------------------------------------------------------
def demo_sequential_chain():
    """
    Real pipelines are often more than one model call: draft, then critique,
    then revise. RunnablePassthrough lets you carry the original input
    forward alongside a step's output.
    """
    print("\n=== 4. Sequential / multi-step chain ===")

    model = get_model()
    parser = StrOutputParser()

    draft_prompt = ChatPromptTemplate.from_template(
        "Write a one-sentence tagline for a startup that does: {idea}"
    )
    critique_prompt = ChatPromptTemplate.from_template(
        "Critique this tagline in one short sentence: {tagline}"
    )

    draft_chain = draft_prompt | model | parser

    full_chain = (
        {"tagline": draft_chain, "idea": RunnablePassthrough()}
        | RunnableLambda(lambda x: {"tagline": x["tagline"]})
        | critique_prompt
        | model
        | parser
    )

    result = full_chain.invoke({"idea": "an AI that reviews pull requests"})
    print(result)
    return full_chain


# ---------------------------------------------------------------------------
# 5. Structured output: force the model's answer into a schema
# ---------------------------------------------------------------------------
class MovieRecommendation(BaseModel):
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    reason: str = Field(description="One-sentence reason for the recommendation")


def demo_structured_output():
    """
    JsonOutputParser (or `.with_structured_output()` on newer chat models)
    lets you get typed, parseable data back instead of free text — useful
    once you want to feed the result into other code.
    """
    print("\n=== 5. Structured output ===")

    model = get_model(temperature=0.2)
    parser = JsonOutputParser(pydantic_object=MovieRecommendation)

    prompt = ChatPromptTemplate.from_template(
        "Recommend one movie about {theme}.\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser
    result = chain.invoke({"theme": "time travel"})
    print(result)
    return chain


# ---------------------------------------------------------------------------
# 6. Streaming: watch tokens arrive instead of waiting for the full answer
# ---------------------------------------------------------------------------
def demo_streaming():
    print("\n=== 6. Streaming ===")

    chain = (
        ChatPromptTemplate.from_template("List 3 use cases for {topic}, briefly.")
        | get_model()
        | StrOutputParser()
    )
    for chunk in chain.stream({"topic": "RAG (retrieval augmented generation)"}):
        print(chunk, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# MAIN — comment/uncomment the demos you want to run
# ---------------------------------------------------------------------------
def main():
    demo_basic_chain()
    demo_prompt_variations()
    demo_parallel_chain()
    demo_sequential_chain()
    demo_structured_output()
    demo_streaming()


if __name__ == "__main__":
    main()
