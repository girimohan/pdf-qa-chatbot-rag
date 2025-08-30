# 📚 PDF Q&A Chatbot with RAG

An intelligent document question-answering system using **Retrieval-Augmented Generation (RAG)** architecture. Upload your documents and chat with them using AI!

## 🎯 What it does

- Upload PDF documents and ask questions about their content
- Smart document retrieval using vector embeddings
- AI-powered responses with source citations
- Conversation memory for natural chat experience

## 🛠️ Tech Stack & Models

### **Core Framework**
- **Frontend**: Streamlit
- **RAG Pipeline**: LangChain
- **Vector Database**: Qdrant Cloud

### **AI Models Used**
- **🏠 Local Development**: Ollama + Mistral 7B
- **☁️ Cloud Deployment**: HuggingFace FLAN-T5
- **📄 Document Processing**: Sentence Transformers (all-MiniLM-L6-v2)

## 🚀 Live Demo

Deploy instantly on HuggingFace Spaces - automatically syncs with this GitHub repo!

## ⚡ Quick Start

1. **Clone & Install**:
   ```bash
   git clone https://github.com/girimohan/pdf-qa-chatbot-rag.git
   cd pdf-qa-chatbot-rag
   pip install -r requirements.txt
   ```

2. **Setup Environment** (copy `.env.template` to `.env`):
   ```
   QDRANT_URL=your_qdrant_cloud_url
   QDRANT_API_KEY=your_api_key
   HUGGINGFACE_API_KEY=your_hf_token  # For cloud deployment
   ```

3. **Run**:
   ```bash
   streamlit run app.py
   ```

## 🌐 Deployment

- **HuggingFace Spaces**: Connect this repo for auto-deployment
- **Streamlit Cloud**: Link GitHub repo and add environment secrets

---

*Built to demonstrate RAG architecture, vector databases, and modern AI integration* 🤖
