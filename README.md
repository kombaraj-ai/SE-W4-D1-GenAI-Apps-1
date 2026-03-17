# Week 4 -> Day 1 -> Gen AI Apps - 1
---

## Table of Contents

1. [Simple GenAI Application using LangChain](#simple-genai-application-using-langchain)
    *   [Legacy Approach](#legacy-approach)
    *   [Modern Approach (LangChain v1+ with LCEL)](#modern-approach-langchain-v1-with-lcel)
2. [GenAI App - Code Assistant](#genai-app---code-assistant)
    *   [Legacy Approach](#legacy-approach---code-assistant)
    *   [Modern Approach](#modern-approach---code-assistant)
        *   [Run Locally using Ollama](#run-locally-using-ollama-llama321b)
3. [GenAI App - Smart Email Writer](#genai-app---smart-email-writer)

## Simple GenAI Application using LangChain

### Legacy Approach

```python
from langchain import OpenAI, LLMChain, PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)

prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are a helpful AI assistant.
User says: {user_input}
Your response:
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

if __name__ == "__main__":
    user_input = input("Ask me anything: ")
    response = chain.run({"user_input": user_input})
    print("AI says:", response)
```

---

### Modern Approach (LangChain v1+ with LCEL)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Use ChatOpenAI with gpt-4o-mini instead of the legacy completion-based OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

# PromptTemplate — same structure as before, just imported from langchain_core
prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""You are a helpful AI assistant.
User says: {user_input}
Your response:"""
)

# LCEL: compose the chain using the pipe operator
chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    user_input = input("Ask me anything: ")
    response = chain.invoke({"user_input": user_input})
    print("AI says:", response)
```

---

## What Changed & Why

### 1. Import Paths — Package Restructuring

| Legacy | Modern |
|---|---|
| `from langchain import OpenAI` | `from langchain_openai import ChatOpenAI` |
| `from langchain import PromptTemplate` | `from langchain_core.prompts import PromptTemplate` |
| `from langchain import LLMChain` | *(removed — replaced by LCEL pipe operator)* |

LangChain v0.1+ split the monolithic `langchain` package into focused sub-packages:
- `langchain_core` — stable, low-level primitives (prompts, parsers, runnables)
- `langchain_openai` — OpenAI-specific integrations
- `langchain_community` — third-party integrations

This makes dependencies lighter and upgrades more predictable.

---

### 2. `OpenAI` → `ChatOpenAI` with `model="gpt-4o-mini"`

```python
# Legacy — completion model, no explicit model name needed
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)

# Modern — chat model with explicit model selection
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)
```

The legacy `OpenAI` class targeted completion endpoints (now deprecated by OpenAI). `ChatOpenAI` targets the `/chat/completions` endpoint. Explicitly setting `model="gpt-4o-mini"` is best practice — it avoids relying on library defaults that may change between releases. `gpt-4o-mini` is a fast, cost-efficient model well suited for conversational tasks like this one.

---

### 3. `PromptTemplate` — Same Structure, New Import Path

```python
# Legacy — imported from the top-level langchain package
from langchain import PromptTemplate

# Modern — imported from langchain_core
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""You are a helpful AI assistant.
User says: {user_input}
Your response:"""
)
```

`PromptTemplate` works exactly as before — same `input_variables` and `template` syntax. The only change is the **import path**. It now lives in `langchain_core.prompts`, the stable lightweight primitives layer. The old `from langchain import PromptTemplate` is deprecated and will be removed in a future version.

---

### 4. `LLMChain` → LCEL Pipe Operator (`|`)

```python
# Legacy — opaque wrapper class
chain = LLMChain(llm=llm, prompt=prompt)

# Modern — explicit pipe composition
chain = prompt | llm | StrOutputParser()
```

**LCEL (LangChain Expression Language)** composes runnables using `|`, similar to Unix pipes. Each component (`prompt`, `llm`, `StrOutputParser`) is a `Runnable` with a consistent interface. This approach:

- Makes the data flow **explicit and readable**
- Enables **streaming** with `.stream()`
- Supports **async** via `.ainvoke()` and `.astream()`
- Allows **batching** via `.batch()`
- Is fully compatible with **LangSmith** tracing out of the box

---

### 5. `StrOutputParser` — Extracting the String

```python
chain = prompt | llm | StrOutputParser()
```

`ChatOpenAI` returns an `AIMessage` object, not a plain string. `StrOutputParser` unwraps `.content` from that message automatically, so `chain.invoke(...)` returns a clean string — no manual `.content` access needed.

---

### 6. `.run()` → `.invoke()`

```python
# Legacy
response = chain.run({"user_input": user_input})

# Modern
response = chain.invoke({"user_input": user_input})
```

`.run()` was a convenience shortcut on `LLMChain`. `.invoke()` is the **standard LCEL interface**, consistent across all runnables. It also unlocks `.stream()`, `.batch()`, and `.ainvoke()` on the same chain object.

---

## Bonus: Async & Streaming (LCEL Advantage)

One of the key benefits of LCEL is built-in async and streaming with no extra boilerplate:

```python
# Streaming — print tokens as they arrive
for chunk in chain.stream({"user_input": "Tell me a joke"}):
    print(chunk, end="", flush=True)

# Async invoke
import asyncio

async def main():
    response = await chain.ainvoke({"user_input": "Hello!"})
    print(response)

asyncio.run(main())
```

These capabilities are not available on the legacy `LLMChain.run()` pattern.

---

## Dependencies

Install the required packages:

```bash
pip install langchain-core langchain-openai python-dotenv
```

> **Note:** The legacy `langchain` package (v0.0.x) and `LLMChain` are deprecated as of LangChain v0.1 and will be removed in future versions. Migrating to LCEL ensures forward compatibility.


## GenAI App - Code Assistant

### Legacy Approach - Code Assistant

```python
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    temperature=0.5
)

# Define the prompt
prompt = PromptTemplate(
    input_variables=["code_task"],
    template="""
You are a professional coding assistant. Help the user with the following task:
{code_task}
Provide clean, well-commented code and explanations if needed.
"""
)

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.title("Code Assistant")
code_task = st.text_area("Describe your coding task:")

if st.button("Generate Code"):
    if code_task.strip() == "":
        st.warning("Please enter a task description.")
    else:
        response = chain.run({"code_task": code_task})
        st.subheader("Assistant Response")
        st.code(response, language='python')
```

---

## Modern Approach - Code Assistant

```python
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5
)

# Define the prompt
prompt = PromptTemplate(
    input_variables=["code_task"],
    template="""
You are a professional coding assistant. Help the user with the following task:
{code_task}
Provide clean, well-commented code and explanations if needed.
"""
)

# LCEL: compose the chain using the pipe operator
chain = prompt | llm | StrOutputParser()

# Streamlit UI
st.title("Code Assistant")
code_task = st.text_area("Describe your coding task:")

if st.button("Generate Code"):
    if code_task.strip() == "":
        st.warning("Please enter a task description.")
    else:
        response = chain.invoke({"code_task": code_task})
        st.subheader("Assistant Response")
        st.code(response, language='python')
```

---

## What Changed & Why

### 1. Import Paths — Package Restructuring

| Legacy | Modern |
|---|---|
| `from langchain.chains import LLMChain` | *(removed — replaced by LCEL pipe operator)* |
| `from langchain.prompts import PromptTemplate` | `from langchain_core.prompts import PromptTemplate` |
| *(not present)* | `from langchain_core.output_parsers import StrOutputParser` |

`langchain.chains` and `langchain.prompts` are the old monolithic import paths. In v1+, prompts and parsers live in `langchain_core` — the stable, lightweight primitives layer that rarely changes. Moving to `langchain_core` imports protects your code from future deprecation removals.

---

### 2. `model_name` → `model`

```python
# Legacy — used model_name parameter
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    temperature=0.5
)

# Modern — use model parameter
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5
)
```

`model_name` was the legacy parameter name on `ChatOpenAI`. In LangChain v1+, it is standardised to `model` across all LLM classes. Using `model_name` will still work for now but triggers a deprecation warning.

---

### 3. `LLMChain` → LCEL Pipe Operator (`|`)

```python
# Legacy — opaque wrapper class
chain = LLMChain(llm=llm, prompt=prompt)

# Modern — explicit pipe composition
chain = prompt | llm | StrOutputParser()
```

**LCEL (LangChain Expression Language)** composes runnables using `|`, similar to Unix pipes. Each component (`prompt`, `llm`, `StrOutputParser`) is a `Runnable` with a consistent interface. This approach:

- Makes the data flow **explicit and readable**
- Enables **streaming** with `.stream()`
- Supports **async** via `.ainvoke()` and `.astream()`
- Allows **batching** via `.batch()`
- Is fully compatible with **LangSmith** tracing out of the box

---

### 4. `StrOutputParser` — Extracting the String

```python
chain = prompt | llm | StrOutputParser()
```

`ChatOpenAI` returns an `AIMessage` object, not a plain string. `StrOutputParser` unwraps `.content` from that message automatically, so `chain.invoke(...)` delivers a clean string directly to `st.code()` — no manual `.content` access needed.

---

### 5. `.run()` → `.invoke()`

```python
# Legacy
response = chain.run({"code_task": code_task})

# Modern
response = chain.invoke({"code_task": code_task})
```

`.run()` was a convenience shortcut on `LLMChain`. `.invoke()` is the **standard LCEL interface**, consistent across all runnables. The Streamlit UI logic (`st.warning`, `st.code`) is otherwise untouched.

---

## Bonus: Streaming Responses in Streamlit (LCEL Advantage)

LCEL makes it trivial to stream tokens directly into the Streamlit UI — great for longer code generation responses:

```python
if st.button("Generate Code"):
    if code_task.strip() == "":
        st.warning("Please enter a task description.")
    else:
        st.subheader("Assistant Response")
        with st.empty():
            full_response = ""
            for chunk in chain.stream({"code_task": code_task}):
                full_response += chunk
                st.code(full_response, language="python")
```

This streams each token into `st.code()` as it arrives, giving users immediate visual feedback — not possible with the legacy `LLMChain.run()` pattern.

---

## Dependencies

Install the required packages:

```bash
pip install langchain-core langchain-openai streamlit python-dotenv
```

Run the app:

```bash
streamlit run app.py
```

> **Note:** `LLMChain` and the `langchain.prompts` / `langchain.chains` import paths are deprecated as of LangChain v0.1 and will be removed in future versions. Migrating to LCEL ensures forward compatibility.


## Run Locally using Ollama (llama3.2:1b)

### Code

```python
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the LLM — local Ollama model, no API key needed
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.5
)

# Define the prompt
prompt = PromptTemplate(
    input_variables=["code_task"],
    template="""
You are a professional coding assistant. Help the user with the following task:
{code_task}
Provide clean, well-commented code and explanations if needed.
"""
)

# LCEL: compose the chain using the pipe operator
chain = prompt | llm | StrOutputParser()

# Streamlit UI
st.title("Code Assistant (Ollama)")
code_task = st.text_area("Describe your coding task:")

if st.button("Generate Code"):
    if code_task.strip() == "":
        st.warning("Please enter a task description.")
    else:
        response = chain.invoke({"code_task": code_task})
        st.subheader("Assistant Response")
        st.code(response, language='python')
```

---

## What Changed & Why

### 1. Import — `ChatOpenAI` → `ChatOllama`

```python
# Before
from langchain_openai import ChatOpenAI

# After
from langchain_ollama import ChatOllama
```

`ChatOllama` is the LangChain integration class for locally running Ollama models. It lives in the `langchain_ollama` package and implements the same `Runnable` interface as `ChatOpenAI`, so it slots into the LCEL chain without any other changes.

---

### 2. LLM Initialisation — No API Key Needed

```python
# Before — cloud model, requires API key
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5
)

# After — local model, no API key or .env file required
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.5
)
```

Ollama runs entirely on your local machine. There is no API key, no network call to an external service, and no usage cost. The `dotenv` import and `load_dotenv()` call are no longer needed and have been removed.

---

### 3. Everything Else — Unchanged

The `PromptTemplate`, the LCEL pipe chain, and the entire Streamlit UI are **identical**. This is the core benefit of the LCEL `Runnable` interface — swapping the underlying LLM provider requires changing only the import and the LLM object. The chain composition `prompt | llm | StrOutputParser()` works the same regardless of provider.

---

## Ollama Setup (Prerequisites)

Before running the app, make sure Ollama is installed and the model is pulled locally:

```bash
# 1. Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the llama3.2:1b model (~1.3 GB)
ollama pull llama3.2:1b

# 3. Verify Ollama is running (starts automatically on most systems)
ollama list
```

Ollama serves the model at `http://localhost:11434` by default. `ChatOllama` connects to this endpoint automatically — no configuration required.

---

## Bonus: Streaming with Ollama in Streamlit

Ollama supports streaming out of the box via LCEL's `.stream()` — great for watching code generate token by token:

```python
if st.button("Generate Code"):
    if code_task.strip() == "":
        st.warning("Please enter a task description.")
    else:
        st.subheader("Assistant Response")
        full_response = ""
        placeholder = st.empty()
        for chunk in chain.stream({"code_task": code_task}):
            full_response += chunk
            placeholder.code(full_response, language="python")
```

---

## Dependencies

```bash
pip install langchain-core langchain-ollama streamlit
```

> **Note:** `langchain-ollama` is the official, actively maintained package for Ollama integration. The older `langchain-community` Ollama classes (`ChatOllama` from `langchain_community.chat_models`) are deprecated — always use `from langchain_ollama import ChatOllama`.

Run the app:

```bash
streamlit run app.py
```

## GenAI App - Smart Email Writer

### Code 

```python
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5
)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["bullet_points"],
    template="""
You are an expert email writer. Using the following bullet points, draft a professional, friendly email:
{bullet_points}

Make sure the email has a greeting, clear structure, and a closing.
"""
)

# LCEL: compose the chain using the pipe operator
chain = prompt | llm | StrOutputParser()

# Streamlit UI
st.title("Smart Email Writer")

st.write("Enter key bullet points for your email below:")

bullet_points = st.text_area("Bullet Points", height=200)

if st.button("Generate Email"):
    if bullet_points.strip() == "":
        st.warning("Please enter some bullet points.")
    else:
        email = chain.invoke({"bullet_points": bullet_points})
        st.subheader("Drafted Email")
        st.write(email)
```

---


### Bonus: Streaming the Email in Streamlit (LCEL Advantage)

LCEL makes it trivial to stream the generated email token-by-token into the UI:

```python
if st.button("Generate Email"):
    if bullet_points.strip() == "":
        st.warning("Please enter some bullet points.")
    else:
        st.subheader("Drafted Email")
        full_email = ""
        placeholder = st.empty()
        for chunk in chain.stream({"bullet_points": bullet_points}):
            full_email += chunk
            placeholder.write(full_email)
```

This gives users immediate visual feedback as the email is drafted — not possible with the legacy `LLMChain.run()` pattern.

---

### Dependencies

```bash
pip install langchain-core langchain-openai streamlit python-dotenv
```

Run the app:

```bash
streamlit run app.py
```

> **Note:** `LLMChain` and the `langchain.prompts` / `langchain.chains` import paths are deprecated as of LangChain v0.1 and will be removed in future versions. Migrating to LCEL ensures forward compatibility.
