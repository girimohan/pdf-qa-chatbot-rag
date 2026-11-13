import logging
import os
from collections.abc import Sequence
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams

from rag import (
    CHUNK_SIZE,
    MAX_CHUNKS_PER_SESSION,
    DocumentChunk,
    answer_question,
    chunk_pages,
    create_qdrant_client,
    create_session_documents,
    create_session_retriever,
    delete_session_documents,
    load_document_pages,
    source_label,
    source_text,
    summarize_document,
)

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "pdf_docs")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
FLAN_MODEL = os.getenv("HF_FLAN_MODEL", "google/flan-t5-base")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
IS_HF_SPACE = bool(os.getenv("SPACE_ID"))
LLM_BACKEND = os.getenv(
    "LLM_BACKEND", "flan_t5" if IS_HF_SPACE else "ollama"
).lower()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_MODE = os.getenv("QDRANT_MODE", "memory").strip().lower()
MAX_TOTAL_UPLOAD_BYTES = 25 * 1024 * 1024

st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide",
    menu_items={"About": "Ask questions about uploaded PDF and text documents."},
)


@st.cache_resource
def get_embeddings(model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def get_qdrant_resources(
    collection_name: str,
    embedding_model: str,
    qdrant_mode: str,
    qdrant_url: str,
    qdrant_api_key: str,
):
    embeddings = get_embeddings(embedding_model)
    client = create_qdrant_client(
        qdrant_mode,
        url=qdrant_url or None,
        api_key=qdrant_api_key or None,
    )

    if not client.collection_exists(collection_name):
        vector_size = len(embeddings.embed_query("dimension check"))
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return client, vector_store


@st.cache_resource
def load_flan_t5(model_name: str):
    from langchain_huggingface import HuggingFacePipeline

    return HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text2text-generation",
        pipeline_kwargs={
            "max_new_tokens": 128,
            "do_sample": False,
            "truncation": True,
        },
    )


@st.cache_resource
def load_ollama(model_name: str, base_url: str):
    from langchain_ollama import OllamaLLM

    return OllamaLLM(
        model=model_name,
        base_url=base_url,
        temperature=0.1,
    )


def get_llm():
    if LLM_BACKEND == "flan_t5":
        return load_flan_t5(FLAN_MODEL)
    if LLM_BACKEND == "ollama":
        return load_ollama(OLLAMA_MODEL, OLLAMA_BASE_URL)
    raise ValueError("LLM_BACKEND must be either 'flan_t5' or 'ollama'.")


def initialize_session_state() -> None:
    defaults = {
        "document_chunks": [],
        "document_names": [],
        "messages": [],
        "session_id": uuid4().hex,
        "index_id": None,
        "uploader_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_messages(messages: Sequence[dict]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            if message.get("error"):
                st.error(message["content"])
            else:
                st.markdown(message["content"])
            sources = message.get("sources", [])
            if sources:
                with st.expander(f"Source passages ({len(sources)})"):
                    for source in sources[:20]:
                        st.caption(source["label"])
                        st.write(source["text"])
                    if len(sources) > 20:
                        st.caption(f"{len(sources) - 20} more passages omitted.")


def run_chat_action(prompt: str, summarize: bool = False) -> None:
    try:
        llm = get_llm()
        if summarize:
            answer, source_chunks = summarize_document(
                st.session_state.document_chunks,
                llm,
            )
        else:
            _, vector_store = get_qdrant_resources(
                COLLECTION_NAME,
                EMBEDDING_MODEL,
                QDRANT_MODE,
                QDRANT_URL or "",
                QDRANT_API_KEY or "",
            )
            retriever = create_session_retriever(
                vector_store,
                st.session_state.session_id,
                st.session_state.index_id,
            )
            answer, source_chunks = answer_question(
                prompt,
                retriever,
                llm,
                st.session_state.messages,
            )
        sources = [
            {"label": source_label(chunk), "text": source_text(chunk)}
            for chunk in source_chunks
        ]
        st.session_state.messages.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer, "sources": sources},
            ]
        )
    except Exception as exc:
        logger.exception("Document question failed")
        st.session_state.messages.extend(
            [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": (
                        f"Could not complete the request: {exc}. Check the selected "
                        "model configuration and service logs."
                    ),
                    "error": True,
                },
            ]
        )
    st.rerun()


initialize_session_state()

st.title("📚 PDF Q&A Chat")
st.caption("Upload documents, get a document-wide summary, and ask questions with citations.")
st.info(
    "Text-based PDFs and TXT files only. Documents are session-scoped. "
    "The first model use may take longer while models are downloaded."
)

with st.sidebar:
    st.header("Documents")
    if LLM_BACKEND == "flan_t5":
        st.success(f"Hugging Face Space model: {FLAN_MODEL}")
        st.caption("FLAN-T5 runs inside this Space on CPU; first load can be slow.")
    elif LLM_BACKEND == "ollama":
        st.success(f"Local Ollama model: {OLLAMA_MODEL}")
        st.caption(f"Ollama endpoint: {OLLAMA_BASE_URL}")
    else:
        st.error("Unsupported LLM_BACKEND configuration.")

    if QDRANT_MODE == "cloud":
        st.caption("Vector store: configured Qdrant server")
        st.warning(
            "Source text is stored in the configured Qdrant server until the session "
            "is cleared. Avoid sensitive documents."
        )
    elif QDRANT_MODE == "memory":
        st.caption("Vector store: session-isolated in-memory Qdrant")
    else:
        st.error("QDRANT_MODE must be either 'memory' or 'cloud'.")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"document_uploader_{st.session_state.uploader_key}",
        help="Text-based PDFs are supported. Scanned PDFs require OCR.",
    )

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        total_size = sum(file.size for file in uploaded_files or [])
        if total_size > MAX_TOTAL_UPLOAD_BYTES:
            st.error("The combined upload size cannot exceed 25 MB.")
        else:
            pages = []
            document_names = []
            errors = []
            with st.spinner("Extracting text and creating searchable chunks..."):
                for uploaded_file in uploaded_files or []:
                    try:
                        pages.extend(
                            load_document_pages(
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                            )
                        )
                        document_names.append(uploaded_file.name)
                    except ValueError as exc:
                        errors.append(f"{uploaded_file.name}: {exc}")

            for error in errors:
                st.error(error)

            if pages:
                chunks = chunk_pages(pages)
                if len(chunks) > MAX_CHUNKS_PER_SESSION:
                    st.error(
                        f"Documents create {len(chunks)} chunks; the session limit "
                        f"is {MAX_CHUNKS_PER_SESSION}. Upload fewer or shorter documents."
                    )
                else:
                    try:
                        with st.spinner("Loading embeddings and saving to Qdrant..."):
                            client, vector_store = get_qdrant_resources(
                                COLLECTION_NAME,
                                EMBEDDING_MODEL,
                                QDRANT_MODE,
                                QDRANT_URL or "",
                                QDRANT_API_KEY or "",
                            )
                            new_index_id = uuid4().hex
                            documents = create_session_documents(
                                chunks,
                                st.session_state.session_id,
                                new_index_id,
                            )
                            vector_store.add_documents(documents)

                        old_index_id = st.session_state.index_id
                        st.session_state.document_chunks = chunks
                        st.session_state.document_names = document_names
                        st.session_state.index_id = new_index_id
                        st.session_state.messages = []
                        if old_index_id:
                            try:
                                delete_session_documents(
                                    client,
                                    COLLECTION_NAME,
                                    st.session_state.session_id,
                                    old_index_id,
                                )
                            except Exception:
                                logger.exception("Could not remove the previous session index")
                                st.warning(
                                    "New documents are ready, but cleanup of the prior "
                                    "session index failed. Use Clear session to remove it."
                                )
                        st.success(
                            f"Ready: {len(document_names)} document(s), "
                            f"{len(chunks)} searchable chunks."
                        )
                    except Exception as exc:
                        logger.exception("Document indexing failed")
                        st.error(f"Could not index documents in Qdrant: {exc}")
            elif not errors:
                st.error("No readable document content was found.")

    if st.session_state.document_chunks:
        st.success(
            f"{len(st.session_state.document_names)} document(s) active in this session"
        )
        if st.button("Clear session"):
            try:
                client, _ = get_qdrant_resources(
                    COLLECTION_NAME,
                    EMBEDDING_MODEL,
                    QDRANT_MODE,
                    QDRANT_URL or "",
                    QDRANT_API_KEY or "",
                )
                delete_session_documents(
                    client,
                    COLLECTION_NAME,
                    st.session_state.session_id,
                )
                st.session_state.document_chunks = []
                st.session_state.document_names = []
                st.session_state.messages = []
                st.session_state.index_id = None
                st.session_state.uploader_key += 1
                st.rerun()
            except Exception as exc:
                logger.exception("Could not clear the session index")
                st.error(f"Could not clear this session's Qdrant data: {exc}")
        st.caption("Clearing removes only this session's vectors.")

st.markdown("### Document chat")
if not st.session_state.document_chunks:
    st.info("Upload and process documents to start.")
else:
    if st.button("Summarize all uploaded documents"):
        with st.spinner("Summarizing all document chunks..."):
            run_chat_action("Summarize all uploaded documents.", summarize=True)

    render_messages(st.session_state.messages)
    prompt = st.chat_input("Ask a question about your documents")
    if prompt:
        with st.spinner("Retrieving passages and generating an answer..."):
            run_chat_action(prompt)
