import os
import tempfile
import logging
from typing import List, Dict, Any
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv

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
                # Use PyPDFLoader for reliable PDF text extraction
                st.info(f"Processing {file.name}...")
                try:
                    loader = PyPDFLoader(tmp_path)
                    pdf_docs = loader.load()
                    if pdf_docs and any(len(doc.page_content.strip()) > 0 for doc in pdf_docs):
                        st.success(f"✅ Successfully extracted text from {file.name}")
                        docs.extend(pdf_docs)
                    else:
                        st.warning(f"⚠️ No readable text found in {file.name}. Please ensure PDF contains text (not just images).")
                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {str(e)}")
                    logger.error(f"Error processing PDF {file.name}: {str(e)}")
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

@st.cache_resource
def get_embeddings():
    """Get embeddings model with caching to avoid reloading."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

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
    embeddings = get_embeddings()  # Use cached embeddings
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

@st.cache_resource
def get_llm():
    """Get the LLM based on environment: Ollama (local) or HuggingFace (production).
    Cached to avoid reloading on every interaction."""
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
                    
                    # Use a smaller, faster model first for quicker startup
                    model_options = [
                        "google/flan-t5-small",    # Fastest startup
                        "google/flan-t5-base",     # Good balance
                        "microsoft/DialoGPT-small" # Alternative fast option
                    ]
                    
                    for model_name in model_options:
                        try:
                            st.sidebar.info(f"🚀 Loading {model_name.split('/')[-1]}...")
                            llm = HuggingFaceEndpoint(
                                repo_id=model_name,
                                huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                                temperature=0.3,
                                max_length=200,  # Reduced for faster response
                                top_k=5,         # Reduced for speed
                                top_p=0.9
                            )
                            
                            # Skip test call for faster startup, just return if model loads
                            st.sidebar.success(f"✅ {model_name.split('/')[-1]} Ready")
                            st.sidebar.caption("🚀 Fast startup • Cloud optimized")
                            return llm
                        except Exception as model_error:
                            st.sidebar.warning(f"⚠️ {model_name} failed, trying next...")
                            continue
                            
                except Exception as hf_error:
                    st.sidebar.error(f"❌ HuggingFace error: {str(hf_error)}")
            else:
                # Try free HuggingFace Inference API without auth (limited requests)
                try:
                    st.sidebar.info("🔄 Trying free inference API...")
                    from langchain_community.llms import HuggingFaceEndpoint
                    
                    # Use small model for free tier
                    llm = HuggingFaceEndpoint(
                        repo_id="google/flan-t5-small",
                        temperature=0.3,
                        max_length=150
                    )
                    
                    response = llm.invoke("Hi")
                    if response:
                        st.sidebar.success("✅ Using Free Inference API")
                        st.sidebar.caption("⚡ Limited requests • Add API key for unlimited")
                        return llm
                except Exception as free_error:
                    st.sidebar.warning(f"⚠️ Free API failed: {str(free_error)}")
                    st.sidebar.info("💡 Add HUGGINGFACE_API_KEY for reliable access")
        
        # LOCAL: Try Ollama for local development
        if is_local_dev and not is_cloud_deployment:
            try:
                from langchain_community.llms import Ollama
                llm = Ollama(model="mistral", temperature=0.7)
                response = llm.invoke("Hi")
                if response:
                    st.sidebar.success("✅ Using Ollama/Mistral (FREE)")
                    return llm
            except Exception as ollama_error:
                st.sidebar.warning(f"⚠️ Ollama not available: {str(ollama_error)}")
                # Try alternative local model
                try:
                    from transformers import pipeline
                    from langchain_community.llms import HuggingFacePipeline
                    
                    pipe = pipeline("text-generation", model="microsoft/DialoGPT-medium", 
                                  tokenizer="microsoft/DialoGPT-medium",
                                  max_length=200,
                                  temperature=0.7)
                    llm = HuggingFacePipeline(pipeline=pipe)
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
                    from langchain_community.llms import Ollama
                    llm = Ollama(
                        model=selected_model,
                        temperature=0.7,
                        base_url="http://localhost:11434"
                    )
                    # Log the initialization details
                    logger.info(f"Initialized Ollama with model: {selected_model}")
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
    """Query the RAG pipeline or general chat."""
    llm = get_llm()
    if llm is None:
        return "LLM not available. Please check the setup instructions.", ""
        
    try:
        # If retriever is available, use RAG
        if retriever:
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
            
            citations = []
            for src in sources:
                meta = src.metadata
                page = meta.get("page", "?")
                fname = meta.get("source", "?")
                citations.append(f"{fname} (page {page})")
            citation_str = ", ".join(citations)
            return answer, citation_str
        else:
            # General chat mode without documents
            response = llm.invoke(query)
            return response, ""
            
    except Exception as e:
        st.error(f"Error during query: {str(e)}")
        logger.error(f"Error during query: {str(e)}")
        return "Error processing your question. Please try again.", ""

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
st.title("� PDF Q&A Chat")
st.caption("Upload documents and chat with them using AI • Powered by RAG & HuggingFace")

# Sidebar - Simplified
st.sidebar.title("� Document Manager")

# Connection status - simplified
if QDRANT_CONNECTED:
    if QDRANT_URL:
        st.sidebar.success("✅ Qdrant Cloud")
    else:
        st.sidebar.success("✅ Local Qdrant")
else:
    st.sidebar.warning("⚠️ Memory Mode")

st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "📁 Choose Files",
    type=["pdf", "txt", "docx", "doc"],
    accept_multiple_files=True,
    help="Upload PDF, TXT, DOCX files"
)

# Show selected files
if uploaded_files:
    st.sidebar.write(f"📄 **{len(uploaded_files)} file(s) selected:**")
    for file in uploaded_files:
        st.sidebar.write(f"• {file.name}")

# Process button
if st.sidebar.button("🚀 Process Documents", type="primary", disabled=not uploaded_files):
    if uploaded_files:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            # Load documents with progress
            status_text.text("Loading documents...")
            progress_bar.progress(0.2)
            docs = load_documents(uploaded_files)
            
            if not docs:
                st.sidebar.error("❌ No valid documents loaded.")
            else:
                # Process documents
                status_text.text("Processing documents...")
                progress_bar.progress(0.5)
                st.session_state.docs = docs
                
                chunks, embeddings = chunk_and_embed(docs)
                progress_bar.progress(0.7)
                
                if chunks and embeddings:
                    # Index to database
                    status_text.text("Creating vector database...")
                    st.session_state.db = index_to_qdrant(chunks, embeddings)
                    st.session_state.retriever = st.session_state.db.as_retriever()
                    st.session_state.indexed = True
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Complete!")
                    
                    # Success message
                    st.sidebar.success(f"🎉 Processed {len(uploaded_files)} files!")
                    st.sidebar.info(f"📊 Created {len(chunks)} searchable chunks")
                    st.balloons()
                    
                else:
                    st.sidebar.error("❌ Processing failed")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

if st.sidebar.button("🗑️ Clear All", disabled=not st.session_state.indexed):
    clear_qdrant()
    st.sidebar.success("✅ Cleared!")
    st.rerun()

st.sidebar.markdown("---")

# Status - simplified
if st.session_state.indexed:
    st.sidebar.success(f"✅ **{len(st.session_state.docs)}** docs indexed")
    st.sidebar.info(f"💬 **{len(st.session_state.chat_history)}** messages")
else:
    st.sidebar.info("📤 Ready for documents")

# Model info - minimal
st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 AI Model**")
st.sidebar.caption("Google FLAN-T5 (Free)")
st.sidebar.caption("Qdrant Vector DB")

# Main Content Area - Always available chat
st.markdown("### 💬 AI Assistant")

# Show current mode
if st.session_state.indexed:
    st.info(f"📚 **Document Mode**: Chatting with {len(st.session_state.docs)} uploaded documents")
else:
    st.info("🤖 **General Mode**: Ask me anything! Upload documents for document-specific Q&A")

# New chat button - top right
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔄 New Chat", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

# Display chat history - always show
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            content = msg["content"]
            if "📎 **Sources:**" in content:
                answer_part, sources_part = content.split("📎 **Sources:**", 1)
                st.write(answer_part)
                st.caption(f"📎 Sources: {sources_part.strip()}")
            else:
                st.write(content)
    
# Quick questions - adaptive based on mode
if not st.session_state.chat_history:
    if st.session_state.indexed:
        st.markdown("**Document questions:**")
        quick_questions = [
            "Summarize main points",
            "Key requirements?", 
            "Important dates?",
            "Main topics?"
        ]
    else:
        st.markdown("**General questions:**")
        quick_questions = [
            "What is artificial intelligence?",
            "Explain machine learning", 
            "How does RAG work?",
            "What is Python?"
        ]
    
    cols = st.columns(4)
    for i, (col, question) in enumerate(zip(cols, quick_questions)):
        with col:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                # Process question
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.spinner("Thinking..."):
                    if st.session_state.indexed:
                        # Document-specific Q&A using RAG
                        answer, citations = query_rag(
                            question,
                            st.session_state.retriever,
                            st.session_state.chat_history
                        )
                        response_text = answer
                        if citations:
                            response_text += f"\n\n📎 **Sources:** {citations}"
                    else:
                        # General Q&A using LLM only
                        llm = get_llm()
                        if llm:
                            try:
                                answer = llm.invoke(question)
                                response_text = answer
                            except Exception as e:
                                response_text = f"Sorry, I encountered an error: {str(e)}"
                        else:
                            response_text = "AI model not available. Please check your setup."
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                st.rerun()

# Chat input - adaptive placeholder
if st.session_state.indexed:
    placeholder_text = "Ask about your documents..."
else:
    placeholder_text = "Ask me anything..."

user_input = st.chat_input(placeholder_text)

if user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Get AI response
    with st.spinner("Generating response..."):
        if st.session_state.indexed:
            # Document-specific Q&A using RAG
            answer, citations = query_rag(
                user_input,
                st.session_state.retriever,
                st.session_state.chat_history
            )
            
            response_text = answer
            if citations:
                response_text += f"\n\n📎 **Sources:** {citations}"
        else:
            # General Q&A using LLM only
            llm = get_llm()
            if llm:
                try:
                    answer = llm.invoke(user_input)
                    response_text = answer
                except Exception as e:
                    response_text = f"Sorry, I encountered an error: {str(e)}"
            else:
                response_text = "AI model not available. Please check your setup."
        
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        st.rerun()

# Tips section - always visible
st.markdown("---")
st.markdown("### 💡 How to Use")

if st.session_state.indexed:
    st.markdown("""
    **📄 Document Mode Active:**
    • Ask specific questions about your uploaded documents
    • Questions will search through your document content
    • Check source citations to verify information
    """)
else:
    st.markdown("""
    **🤖 General AI Mode:**
    • Ask me about any topic - technology, science, etc.
    • Upload documents above to switch to document-specific Q&A
    • I can explain concepts, provide definitions, and answer questions
    """)

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit, LangChain & HuggingFace")
