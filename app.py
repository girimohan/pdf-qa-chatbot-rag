import os
import tempfile
import logging
from typing import List, Dict, Any
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant  # Updated import
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import OpenAI
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import pytesseract

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurations
COLLECTION_NAME = "pdf_docs"
VECTOR_SIZE = 384  # For 'all-MiniLM-L6-v2', change if using a different model
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL")  # For Qdrant Cloud
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # For Qdrant Cloud
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# HuggingFace Spaces optimized configuration
HUGGINGFACE_API_KEY = (
    os.getenv("HF_TOKEN") or  # HF Spaces automatically provides this
    os.getenv("HUGGINGFACE_API_KEY") or  # Fallback for other deployments
    os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Legacy name
)

# Detect if running on HuggingFace Spaces
IS_HF_SPACES = "SPACE_ID" in os.environ or "HF_TOKEN" in os.environ

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Ollama model configurations
OLLAMA_MODELS = {
    "general": "mistral:latest",  # Good balance of performance and speed
    "code": "mistral:latest",    # Use mistral for code as well
    "large": "mistral:latest"    # Use mistral for complex reasoning
}
DEFAULT_MODEL = "mistral:latest"  # Ensure the default model is mistral

# Qdrant client setup
try:
    if QDRANT_URL and QDRANT_API_KEY:
        # Qdrant Cloud configuration
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        logger.info("Connecting to Qdrant Cloud...")
    else:
        # Local Qdrant configuration
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info(f"Connecting to local Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    
    # Test connection
    qdrant_client.get_collections()
    QDRANT_CONNECTED = True
    logger.info("✅ Qdrant connection successful!")
except Exception as e:
    logger.warning(f"Qdrant server not available: {e}")
    qdrant_client = QdrantClient(":memory:")  # Fallback to in-memory
    QDRANT_CONNECTED = False

# Streamlit session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "docs" not in st.session_state:
    st.session_state.docs = []
if "db" not in st.session_state:
    st.session_state.db = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "progress" not in st.session_state:
    st.session_state.progress = 0

# Helper functions

def load_documents(files: List[Any]) -> List[Any]:
    """Load documents from uploaded files."""
    docs = []
    for file in files:
        ext = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        try:
            if ext == "pdf":
                # Try different PDF loading strategies
                st.info(f"Processing {file.name}...")
                
                # 1. Try PyPDFLoader first (fastest)
                try:
                    loader = PyPDFLoader(tmp_path)
                    pdf_docs = loader.load()
                    if pdf_docs and any(len(doc.page_content.strip()) > 0 for doc in pdf_docs):
                        st.success(f"✅ Successfully extracted text from {file.name}")
                        docs.extend(pdf_docs)
                        continue
                    else:
                        st.warning(f"⚠️ No text content found in {file.name}, trying OCR...")
                except Exception as e:
                    st.warning(f"⚠️ Standard extraction failed for {file.name}: {str(e)}")
                
                # 2. Try UnstructuredPDFLoader with OCR
                try:
                    st.info(f"🔍 Attempting OCR extraction for {file.name}...")
                    loader = UnstructuredPDFLoader(
                        tmp_path,
                        mode="elements",
                        strategy="ocr_only",  # Force OCR for scanned documents
                        languages=["eng"],
                        ocr_language="eng"
                    )
                    pdf_docs = loader.load()
                    if pdf_docs and any(len(doc.page_content.strip()) > 0 for doc in pdf_docs):
                        st.success(f"✅ Successfully extracted text from {file.name} using OCR")
                        docs.extend(pdf_docs)
                    else:
                        st.error(f"❌ Could not extract any text from {file.name}. Please verify the PDF contains readable content.")
                except Exception as e:
                    st.error(f"❌ OCR extraction failed for {file.name}: {str(e)}")
            elif ext == "txt":
                loader = TextLoader(tmp_path)
            elif ext in ["docx", "doc"]:
                loader = UnstructuredWordDocumentLoader(tmp_path)
            else:
                st.error(f"Unsupported file type: {ext}")
                continue
            doc_objs = loader.load()
            docs.extend(doc_objs)
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")
            logger.error(f"Error loading {file.name}: {e}")
        finally:
            os.remove(tmp_path)
    return docs

def chunk_and_embed(docs: List[Any]) -> List[Dict]:
    """Chunk documents and generate embeddings."""
    if not docs:
        st.error("No documents were loaded successfully.")
        return [], None
        
    # Log the content of the first document for debugging
    if docs:
        st.info(f"📄 Processing {len(docs)} document pages (Total characters: {sum(len(doc.page_content) for doc in docs):,})")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_documents(docs)
    
    if not chunks:
        st.error("❌ No chunks were generated from the documents.")
        return [], None
        
    st.success(f"✅ Generated {len(chunks)} text chunks for optimal retrieval")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return chunks, embeddings

def index_to_qdrant(chunks: List[Any], embeddings: Any):
    """Index chunks to Qdrant with persistent storage."""
    try:
        # Check if collection exists, if not create it
        collections = qdrant_client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        if not collection_exists:
            from qdrant_client.models import Distance, VectorParams
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            st.info(f"🆕 Created new collection: {COLLECTION_NAME}")
        else:
            st.info(f"📊 Using existing collection: {COLLECTION_NAME}")
        
        # Create Qdrant vector store with persistent connection
        db = Qdrant(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings
        )
        
        # Add documents to the collection
        with st.spinner("🔄 Indexing documents to vector database..."):
            db.add_documents(chunks)
        st.success(f"✅ Successfully indexed {len(chunks)} chunks to Qdrant Cloud")
        
        return db
        
    except Exception as e:
        st.error(f"Error indexing to Qdrant: {str(e)}")
        logger.error(f"Error indexing to Qdrant: {str(e)}")
        # Fallback to in-memory if Qdrant server is not available
        st.warning("Falling back to in-memory storage")
        return Qdrant.from_documents(
            chunks,
            embeddings,
            location=":memory:",
            collection_name=COLLECTION_NAME,
            prefer_grpc=False
        )

def detect_content_type(docs):
    """Detect the type of content in the documents."""
    # Join all document content
    all_content = " ".join(doc.page_content for doc in docs).lower()
    
    # Check for code-related content
    code_indicators = ["def ", "class ", "function", "import ", "console.log", 
                      "public class", "private void", "var ", "let ", "const "]
    if any(indicator in all_content for indicator in code_indicators):
        return "code"
    
    # Add more content type detection as needed
    return "general"

def get_llm():
    """Get the LLM based on environment: Ollama (local) or HuggingFace (production)."""
    try:
        # Detect environment - Local development vs Cloud deployment
        is_local_dev = os.getenv("ENVIRONMENT") == "local" or os.path.exists("C:/Users")  # Windows dev
        is_cloud_deployment = (
            os.getenv("ENVIRONMENT") == "production" or
            "streamlit" in os.getenv("USER", "").lower() or  # Streamlit Cloud
            os.getenv("DYNO") or  # Heroku
            os.getenv("RAILWAY_ENVIRONMENT") or  # Railway
            os.getenv("RENDER")  # Render
        )
        
        st.sidebar.markdown("### 🤖 AI Model Status")
        if is_local_dev and not is_cloud_deployment:
            st.sidebar.info("🏠 **Local Development Mode**")
            st.sidebar.caption("Using Ollama/Mistral for privacy")
        else:
            st.sidebar.info("☁️ **Production Mode**") 
            st.sidebar.caption("Using HuggingFace (free tier)")
        
        # PRODUCTION: Try HuggingFace (FREE) for cloud deployment
        if is_cloud_deployment or HUGGINGFACE_API_KEY:
            if HUGGINGFACE_API_KEY:
                try:
                    from langchain_community.llms import HuggingFaceEndpoint
                    
                    # Use Google's FLAN-T5 - excellent for Q&A tasks
                    llm = HuggingFaceEndpoint(
                        repo_id="google/flan-t5-base",  # Best free model for Q&A
                        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                        temperature=0.3,  # Lower for more focused answers
                        max_length=256,   # Optimal for Q&A responses
                        top_k=10,
                        top_p=0.95
                    )
                    
                    # Test with a simple query
                    response = llm.invoke("What is AI?")
                    if response:
                        st.sidebar.success("✅ Google FLAN-T5 Connected")
                        st.sidebar.caption("� Free tier • Cloud optimized")
                        return llm
                except Exception as hf_error:
                    st.sidebar.error(f"❌ HuggingFace error: {str(hf_error)}")
        
        # LOCAL: Try Ollama for local development
        if is_local_dev and not is_cloud_deployment:
                            temperature=0.7,
                            max_length=512
                        )
                        response = llm.invoke("Hi")
                        if response:
                            st.sidebar.success("✅ Using DialoGPT (FREE)")
                            return llm
                    except Exception as fallback_error:
                        st.sidebar.error(f"❌ Fallback model failed: {str(fallback_error)}")
            else:
                st.sidebar.warning("🌐 Cloud deployment detected")
                st.sidebar.info("💡 Add HUGGINGFACE_API_KEY for FREE cloud LLM")
        
        # Try OpenAI as fallback (if user wants to pay)
        if OPENAI_API_KEY:
            try:
                from langchain_community.llms import OpenAI
                llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)
                # Test with a simple query
                response = llm.invoke("Hi")
                if response:
                    st.sidebar.success("✅ Using OpenAI GPT")
                    return llm
            except Exception as openai_error:
                st.sidebar.error(f"❌ OpenAI error: {str(openai_error)}")
        
        # Try Ollama for local development
        if not is_cloud_deployment:
            try:
                # Get content type if documents are loaded
                if hasattr(st.session_state, 'docs') and st.session_state.docs:
                    content_type = detect_content_type(st.session_state.docs)
                    model = OLLAMA_MODELS.get(content_type, DEFAULT_MODEL)
                else:
                    model = DEFAULT_MODEL
                
                # Let user override model choice
                selected_model = st.sidebar.selectbox(
                    "Select LLM Model",
                    options=list(OLLAMA_MODELS.values()),
                    index=list(OLLAMA_MODELS.values()).index(model),
                    help="Choose the best model for your use case"
                )
                
                # Test Ollama connection
                try:
                    # Initialize Ollama with simplified configuration
                    llm = OllamaLLM(
                        model=selected_model,
                        temperature=0.7,
                        base_url="http://localhost:11434"
                    )
                    # Log the initialization details
                    logger.info(f"Initialized OllamaLLM with model: {selected_model}")
                    # Simple test query with more informative error handling
                    response = llm.invoke("Hi")
                    logger.info(f"Ollama response: {response}")
                    if response:
                        st.sidebar.success(f"✅ Using Ollama with {selected_model}")
                        return llm
                    else:
                        st.sidebar.error("❌ Ollama test response was empty")
                        logger.error("Ollama test response was empty.")
                except Exception as query_error:
                    st.sidebar.error(f"❌ Ollama test query failed: {str(query_error)}")
                    logger.error(f"Ollama test query failed: {str(query_error)}")
                raise RuntimeError("Failed to initialize Ollama. Check logs for details.")
            except Exception as e:
                st.sidebar.warning("⚠️ Ollama not available locally")
                logger.error(f"Ollama error: {str(e)}")
        
        # If everything fails
        st.error("❌ No LLM available. Please add one of these FREE options:")
        st.error("1. **🤗 Hugging Face** (FREE): Get API key from https://huggingface.co/settings/tokens")
        st.error("2. **🤖 Local Ollama**: Install from https://ollama.ai")
        st.error("3. **💳 OpenAI** (Paid): Set OPENAI_API_KEY")
        st.info("💡 For portfolio demos, Hugging Face is completely FREE!")
        return None
        
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def query_rag(query: str, retriever: Any, chat_history: List[Dict]) -> str:
    """Query the RAG pipeline."""
    llm = get_llm()
    if llm is None:
        return "LLM not available. Please check the setup instructions.", ""
        
    try:
        # Convert chat history from dict format to tuple format expected by ConversationalRetrievalChain
        formatted_history = []
        for i in range(0, len(chat_history) - 1, 2):  # Process pairs of user/assistant messages
            if i + 1 < len(chat_history):
                user_msg = chat_history[i]
                assistant_msg = chat_history[i + 1]
                if user_msg.get('role') == 'user' and assistant_msg.get('role') == 'assistant':
                    formatted_history.append((user_msg['content'], assistant_msg['content']))
        
        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever,
            return_source_documents=True,
            chain_type="stuff"
        )
        result = chain.invoke({"question": query, "chat_history": formatted_history})
        answer = result["answer"]
        sources = result.get("source_documents", [])
    except Exception as e:
        st.error(f"Error during query: {str(e)}")
        logger.error(f"Error during query: {str(e)}")
        return "Error processing your question. Please try again.", ""
    
    citations = []
    for src in sources:
        meta = src.metadata
        page = meta.get("page", "?")
        fname = meta.get("source", "?")
        citations.append(f"{fname} (page {page})")
    citation_str = ", ".join(citations)
    return answer, citation_str
    citations = []
    for src in sources:
        meta = src.metadata
        page = meta.get("page", "?")
        fname = meta.get("source", "?")
        citations.append(f"{fname} (page {page})")
    citation_str = ", ".join(citations)
    return answer, citation_str

def clear_qdrant():
    """Clear the Qdrant collection."""
    try:
        # Check if collection exists before trying to delete
        collections = qdrant_client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        if collection_exists:
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
            st.success("Qdrant collection cleared successfully.")
        else:
            st.info("No collection to clear.")
            
        # Reset session state
        st.session_state.indexed = False
        st.session_state.docs = []
        st.session_state.db = None
        st.session_state.retriever = None
        st.session_state.chain = None
        st.session_state.chat_history = []
        
    except Exception as e:
        st.error(f"Failed to clear Qdrant: {e}")
        logger.error(f"Failed to clear Qdrant: {e}")
        # If server connection fails, just reset session state
        st.session_state.indexed = False
        st.session_state.docs = []
        st.session_state.db = None
        st.session_state.retriever = None
        st.session_state.chain = None
        st.session_state.chat_history = []

# Streamlit UI
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide",
    menu_items={
        'About': "🤗 Optimized for HuggingFace Spaces deployment"
    }
)

# Main title with environment indicator
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📚 PDF Q&A Chatbot with RAG")
with col2:
    if IS_HF_SPACES:
        st.success("🤗 HF Spaces")
    else:
        st.info("💻 Local/Cloud")

st.sidebar.title("📚 Document Manager")

# Display Qdrant connection status
if QDRANT_CONNECTED:
    if QDRANT_URL:
        st.sidebar.success("✅ Qdrant Cloud Connected")
    else:
        st.sidebar.success("✅ Local Qdrant Connected")
else:
    st.sidebar.warning("⚠️ Using In-Memory Storage")
    st.sidebar.info("💡 Setup Options:")
    st.sidebar.markdown("• **Qdrant Cloud**: cloud.qdrant.io (recommended)")
    st.sidebar.markdown("• **Docker**: `docker-compose up -d`")
    st.sidebar.markdown("• **Binary**: Download from qdrant.tech")

st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "📁 Upload Documents",
    type=["pdf", "txt", "docx", "doc"],
    accept_multiple_files=True,
    help="Supported formats: PDF, TXT, DOCX, DOC"
)

if st.sidebar.button("🚀 Index Documents", type="primary"):
    if not uploaded_files:
        st.sidebar.error("❌ No files uploaded.")
    else:
        with st.spinner("📊 Processing documents..."):
            docs = load_documents(uploaded_files)
            if not docs:
                st.sidebar.error("❌ No valid documents loaded.")
            else:
                st.session_state.docs = docs
                chunks, embeddings = chunk_and_embed(docs)
                if chunks and embeddings:  # Only proceed if we have valid chunks and embeddings
                    try:
                        st.session_state.db = index_to_qdrant(chunks, embeddings)
                        st.session_state.retriever = st.session_state.db.as_retriever(
                            search_kwargs={"filter": None}
                        )
                        st.session_state.indexed = True
                        st.balloons()  # Celebration animation!
                        
                        # Show summary
                        total_chunks = len(chunks)
                        total_files = len(uploaded_files)
                        st.success(f"🎉 Indexing Complete!")
                        st.info(f"📊 **Summary**: {total_files} file(s) → {total_chunks} searchable chunks")
                        st.info("💬 You can now ask questions about your documents below!")
                    except Exception as e:
                        st.error(f"❌ Error during indexing: {str(e)}")
                else:
                    st.error("❌ Could not process documents. Please check if they contain extractable text.")

if st.sidebar.button("🗑️ Clear Database"):
    clear_qdrant()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 App Status")
if st.session_state.indexed:
    st.sidebar.success(f"✅ {len(st.session_state.docs)} documents indexed")
    st.sidebar.info(f"💬 {len(st.session_state.chat_history)} chat messages")
else:
    st.sidebar.info("📤 Ready to upload documents")

# Add model information
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI Models")
st.sidebar.markdown("""
**💬 Chat Model**: Google FLAN-T5 (Free)
**🔍 Embeddings**: all-MiniLM-L6-v2
**💾 Vector DB**: Qdrant Cloud
**⚡ Platform**: Streamlit Community Cloud

*All models are completely FREE!*
""")

# Add helpful tips
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
• **Ask specific questions** for better answers
• **Use natural language** - no special syntax needed
• **Try different phrasings** if results aren't perfect
• **Check sources** to verify information
""")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Links")
st.sidebar.markdown("""
[📖 Documentation](https://github.com/your-repo)
[🐛 Report Bug](https://github.com/your-repo/issues)
[⭐ Star Project](https://github.com/your-repo)
""")

st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ using Streamlit & AI")

st.title("🤖 AI Document Assistant")
st.markdown("*Powered by RAG (Retrieval-Augmented Generation) • Qdrant Cloud • Free AI Models*")
st.markdown("---")

if st.session_state.indexed:
    # Create two columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Chat with your documents")
    
    with col2:
        if st.button("🔄 New Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Custom CSS for better chat appearance
    st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ff6b6b;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #ff6b6b;
    }
    .assistant-message {
        background-color: #e8f4f8;
        border-left-color: #1f77b4;
    }
    .source-citation {
        font-size: 0.85em;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display chat history with improved formatting
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"**You asked:** {msg['content']}")
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    # Parse assistant message to separate answer and sources
                    content = msg["content"]
                    if "📎 **Sources:**" in content:
                        answer_part, sources_part = content.split("📎 **Sources:**", 1)
                        st.markdown(answer_part.replace("**Assistant:** ", ""))
                        st.markdown(f'<div class="source-citation">📎 <strong>Sources:</strong> {sources_part.strip()}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(content.replace("**Assistant:** ", ""))
    
    # Improved chat input with suggestions
    st.markdown("---")
    
    # Quick question buttons
    st.markdown("**💡 Quick questions:**")
    quick_questions = [
        "📋 What are the main topics?",
        "📊 Can you summarize this?", 
        "🔍 What are the key requirements?",
        "❓ What should I know about this?"
    ]
    
    cols = st.columns(len(quick_questions))
    for i, question in enumerate(quick_questions):
        with cols[i]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                # Remove emoji and process the question
                clean_question = question.split(" ", 1)[1]
                user_query = clean_question
                # Process the question immediately
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with st.spinner("🤔 Analyzing documents..."):
                    answer, citations = query_rag(
                        user_query,
                        st.session_state.retriever,
                        st.session_state.chat_history
                    )
                    response_text = f"{answer}"
                    if citations:
                        response_text += f"\n\n📎 **Sources:** {citations}"
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                st.rerun()
    
    # Custom chat input
    user_query = st.chat_input("💭 Ask anything about your documents... (e.g., 'What are the main requirements?')")
    
    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Show thinking animation
        with st.spinner("🤔 Analyzing documents and generating response..."):
            answer, citations = query_rag(
                user_query,
                st.session_state.retriever,
                st.session_state.chat_history
            )
            
            # Format response with sources
            response_text = f"{answer}"
            if citations:
                response_text += f"\n\n📎 **Sources:** {citations}"
            
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        
        # Rerun to show the new messages
        st.rerun()
        
else:
    # Enhanced welcome screen
    st.markdown("### 🚀 Get Started")
    
    # Create attractive info boxes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>📁 Upload</h3>
            <p>PDF, DOCX, TXT files</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>🔄 Process</h3>
            <p>AI extracts & indexes content</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>💬 Chat</h3>
            <p>Ask questions & get answers</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature highlights
    st.markdown("### ✨ Features")
    feature_cols = st.columns(2)
    
    with feature_cols[0]:
        st.markdown("""
        **🎯 Smart Q&A**
        - Natural language questions
        - Context-aware responses
        - Source citations included
        
        **🔍 Advanced Search**
        - Semantic similarity matching
        - Multi-document support
        - Persistent storage
        """)
    
    with feature_cols[1]:
        st.markdown("""
        **� Powered By**
        - Google FLAN-T5 (Free AI)
        - Qdrant Vector Database
        - Advanced RAG Pipeline
        
        **💡 Perfect For**
        - Research documents
        - Policy manuals
        - Educational content
        """)
    
    # Instructions with emojis
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **� Upload** your documents using the sidebar
    2. **� Click "Index Documents"** to process them
    3. **💬 Ask questions** about your content
    4. **📚 Get intelligent answers** with source references
    
    **Example questions:**
    - *"What are the main points discussed?"*
    - *"Can you explain the requirements in simple terms?"*
    - *"What should I focus on first?"*
    - *"Are there any important deadlines mentioned?"*
    """)

# Sample run command
# streamlit run app.py

# For containerization, see Dockerfile in project root.

# Inline comments throughout code explain key sections.
