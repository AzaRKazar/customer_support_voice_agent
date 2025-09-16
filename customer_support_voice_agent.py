import os
import uuid
import time
import tempfile
from datetime import datetime
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from firecrawl import Firecrawl
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from fastembed import TextEmbedding
from ollama import chat
from gtts import gTTS

# ======================
# Setup
# ======================
load_dotenv()


def init_session_state():
    defaults = {
        "initialized": False,
        "qdrant_url": "",
        "qdrant_api_key": "",
        "firecrawl_api_key": "",
        "doc_url": "",
        "setup_complete": False,
        "client": None,
        "embedding_model": None,
        "selected_voice": "default"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sidebar_config():
    with st.sidebar:
        st.title("🔑 Configuration")
        st.markdown("---")

        st.session_state.qdrant_url = st.text_input("Qdrant URL", value=st.session_state.qdrant_url)
        st.session_state.qdrant_api_key = st.text_input("Qdrant API Key", value=st.session_state.qdrant_api_key, type="password")
        st.session_state.firecrawl_api_key = st.text_input("Firecrawl API Key", value=st.session_state.firecrawl_api_key, type="password")
        st.session_state.doc_url = st.text_input("Documentation URL", value=st.session_state.doc_url, placeholder="https://docs.example.com")

        st.markdown("---")
        st.session_state.selected_voice = st.selectbox(
            "Select Voice",
            options=["default", "female", "male"],
            index=0,
            help="Choose the voice for the audio response (gTTS)"
        )

        if st.button("Initialize System", type="primary"):
            if all([st.session_state.qdrant_url, st.session_state.qdrant_api_key, st.session_state.firecrawl_api_key, st.session_state.doc_url]):
                try:
                    st.markdown("🔄 Setting up Qdrant connection...")
                    client, embedding_model = setup_qdrant_collection(
                        st.session_state.qdrant_url, st.session_state.qdrant_api_key
                    )
                    st.session_state.client = client
                    st.session_state.embedding_model = embedding_model
                    st.success("✅ Qdrant setup complete!")

                    st.markdown("🔄 Crawling documentation pages...")
                    pages = crawl_documentation(st.session_state.firecrawl_api_key, st.session_state.doc_url)
                    st.success(f"✅ Crawled {len(pages)} documentation pages!")

                    store_embeddings(client, embedding_model, pages, "docs_embeddings")
                    st.session_state.setup_complete = True
                    st.success("✅ System initialized successfully!")

                except Exception as e:
                    st.error(f"Error during setup: {str(e)}")
            else:
                st.error("Please fill in all the required fields!")


def setup_qdrant_collection(qdrant_url: str, qdrant_api_key: str, collection_name: str = "docs_embeddings"):
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embedding_model = TextEmbedding()
    test_embedding = list(embedding_model.embed(["test"]))[0]
    embedding_dim = len(test_embedding)

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise e

    return client, embedding_model


def crawl_documentation(firecrawl_api_key: str, url: str, limit: int = 5):
    firecrawl = Firecrawl(api_key=firecrawl_api_key)

    # Start crawl job
    job = firecrawl.crawl(url=url, limit=limit)

    # Handle the case where crawl returns immediately completed results
    if hasattr(job, 'status') and job.status == 'completed':
        # Job is already completed, extract data directly
        result = job
    else:
        # Handle both return types for job ID: dict or CrawlJob
        if isinstance(job, dict):
            job_id = job.get("id")
        else:
            job_id = getattr(job, "id", None)

        if not job_id:
            raise ValueError(f"Could not extract crawl job ID. Got: {job}")

        # Poll until finished
        while True:
            result = firecrawl.get_crawl_status(job_id)
            status = getattr(result, "status", None) if not isinstance(result, dict) else result.get("status")

            if status == "completed":
                break
            elif status == "failed":
                raise RuntimeError(f"Crawl failed: {result}")
            else:
                time.sleep(2)  # wait before retrying

    # Extract pages
    pages = []
    data = getattr(result, "data", None) if not isinstance(result, dict) else result.get("data", [])
    
    # Handle case where data might be None
    if data is None:
        # Try to get data from other possible attributes
        if hasattr(result, 'results'):
            data = result.results
        elif hasattr(result, 'pages'):
            data = result.pages
        else:
            data = []
    
    for page in data:
        # Handle different page data structures
        if isinstance(page, dict):
            content = page.get("markdown") or page.get("html", "")
            metadata = page.get("metadata", {})
            source_url = metadata.get("sourceURL") or page.get("url", url)
        else:
            # Handle object-like page structure
            content = getattr(page, "markdown", None) or getattr(page, "html", "")
            metadata = getattr(page, "metadata", {})
            if isinstance(metadata, dict):
                source_url = metadata.get("sourceURL") or getattr(page, "url", url)
            else:
                source_url = getattr(page, "url", url)
        
        if content:
            pages.append({
                "content": content,
                "url": source_url,
                "metadata": metadata if isinstance(metadata, dict) else {}
            })

    return pages


def store_embeddings(client: QdrantClient, embedding_model: TextEmbedding, pages: List[Dict], collection_name: str):
    for page in pages:
        embedding = list(embedding_model.embed([page["content"]]))[0]
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": page["content"],
                        "url": page["url"],
                        **page["metadata"]
                    }
                )
            ]
        )


def query_knowledge_base(query: str, client: QdrantClient, embedding_model: TextEmbedding, collection_name: str = "docs_embeddings"):
    query_embedding = list(embedding_model.embed([query]))[0]
    search_response = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=3,
        with_payload=True
    )
    results = search_response.points if hasattr(search_response, "points") else []
    return results


def process_query(query: str, client: QdrantClient, embedding_model: TextEmbedding, doc_url: str):
    results = query_knowledge_base(query, client, embedding_model)
    if not results:
        return {"status": "error", "error": "No relevant documents found"}

    context = "Based on the following documentation:\n\n"
    for result in results:
        payload = result.payload
        if not payload:
            continue
        url = payload.get("url", doc_url)
        content = payload.get("content", "")
        context += f"From {url}:\n{content}\n\n"

    context += f"\nUser Question: {query}\n\nPlease provide a clear, concise answer."

    response = chat(
        model="llama3.2:latest",
        messages=[
            {"role": "system", "content": "You are a helpful documentation assistant."},
            {"role": "user", "content": context}
        ]
    )

    text_response = response["message"]["content"]

    # Generate speech with gTTS
    tts = gTTS(text=text_response, lang="en")
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, f"response_{uuid.uuid4()}.mp3")
    tts.save(audio_path)

    return {"status": "success", "text_response": text_response, "audio_path": audio_path, "sources": [r.payload.get("url") for r in results if r.payload]}


def run_streamlit():
    st.set_page_config(page_title="Customer Support Voice Agent", page_icon="🎙️", layout="wide")
    init_session_state()
    sidebar_config()

    st.title("🎙️ Customer Support Voice Agent")
    st.markdown("Ask questions about documentation and get both text + voice responses!")

    query = st.text_input("What would you like to know?", placeholder="e.g., How do I authenticate API requests?", disabled=not st.session_state.setup_complete)

    if query and st.session_state.setup_complete:
        with st.spinner("Processing your query..."):
            result = process_query(query, st.session_state.client, st.session_state.embedding_model, st.session_state.doc_url)

            if result["status"] == "success":
                st.markdown("### 📖 Text Response")
                st.write(result["text_response"])

                st.markdown("### 🔊 Audio Response")
                st.audio(result["audio_path"], format="audio/mp3")

                with open(result["audio_path"], "rb") as audio_file:
                    st.download_button(
                        label="📥 Download Audio Response",
                        data=audio_file,
                        file_name="voice_response.mp3",
                        mime="audio/mp3"
                    )

                st.markdown("### 📚 Sources")
                for src in result["sources"]:
                    st.markdown(f"- {src}")
            else:
                st.error(f"Error: {result['error']}")


if __name__ == "__main__":
    run_streamlit()