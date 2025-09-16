# 🎙️ Customer Support Voice Agent

A **Streamlit application** that delivers voice-powered answers to questions about your documentation.\
It uses **Firecrawl** to crawl docs, **Qdrant** to store embeddings, **FastEmbed** for vectorization, **Ollama** for LLM reasoning, and **gTTS** for speech synthesis.

Upload a documentation URL, ask questions in plain English, and get back **both text and audio responses**.

## ✨ Features

### Knowledge Base Creation

-Crawl documentation pages with **Firecrawl API**

-Convert docs into **semantic embeddings** using **FastEmbed**

-Store searchable embeddings in **Qdrant vector database**

### AI-Powered Querying

-Uses **Ollama** locally (e.g., `llama3.2:latest`) to answer questions

-Context-aware responses citing relevant docs

-Clear and concise answers optimized for support

### Voice Responses

-Generates **audio output** with **gTTS (Google Text-to-Speech)**

-Supports multiple selectable voice styles (default, male, female)

-Integrated audio player + download option

### Interactive Streamlit UI

-Sidebar for API key setup and configuration

-Enter a documentation URL → initialize system → ask queries

-Get responses with sources, audio, and text side-by-side

## ⚙️ Setup

### 1\. Clone the repository

```bash
`git clone https://github.com/your-username/customer-support-voice-agent.git
cd customer-support-voice-agent`
```

### 2\. Install dependencies

```bash
`pip install -r requirements.txt`
```

**Dependencies include**:

-`streamlit` (UI)

-`firecrawl-py` (documentation crawling)

-`qdrant-client` (vector DB)

-`fastembed` (embeddings)

-`ollama` (local LLM)

-`gtts` (text-to-speech)

### 3\. Install and run Ollama (for local LLM)

-[Download Ollama](https://ollama.com/download)

-Pull a model, e.g.:

```bash
`ollama pull llama3.2`
```

### 4\. Set up required API keys

You'll need:

-**Firecrawl API key** → [Get free key](https://firecrawl.dev)

-**Qdrant Cloud API key + URL** → Sign up

Save them in a `.env` file at the root:

```bash         
`QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your_qdrant_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key`
```

### 5\. Run the app

```bash
`streamlit run app.py`
```

## 🖥️ How to Use

1.Open the Streamlit app in your browser.

2.Enter **Qdrant URL**, **Qdrant API Key**, **Firecrawl API Key**, and the **documentation URL** in the sidebar.

3.Click **Initialize System** to crawl, embed, and index the docs.

4.Ask a question about the documentation in the text box.

5.Get:

- **📖 Text Response** (AI-generated answer)
- **🔊 Audio Response** (play or download)
- **📚 Sources** (links to docs)

## 🛠️ Example Queries

- "How do I authenticate API requests?"

- "Explain how rate limiting works in this API."

- "Summarize the steps for integrating the SDK."

## 🔮 Future Improvements

- Add support for Whisper STT → voice input queries

- Replace gTTS with **Coqui TTS** or other higher-quality offline TTS

- Cache embeddings for faster reloading

- Multi-page doc crawling with deeper Firecrawl integration