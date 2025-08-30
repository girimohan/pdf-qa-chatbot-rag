---
title: PDF Q&A Chatbot with RAG
emoji: ""
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

#  PDF Q&A Chatbot with RAG

An intelligent document question-answering system using **Retrieval-Augmented Generation (RAG)** architecture. Upload your documents and chat with them using AI!

##  What it does

- Upload PDF documents and ask questions about their content
- Smart document retrieval using vector embeddings
- AI-powered responses with source citations
- Conversation memory for natural chat experience

##  Tech Stack & Models

### **Core Framework**
- **Frontend**: Streamlit
- **RAG Pipeline**: LangChain
- **Vector Database**: Qdrant Cloud

### **AI Models Used**
- ** Local Development**: Ollama + Mistral 7B
- ** Cloud Deployment**: HuggingFace FLAN-T5
- ** Document Processing**: Sentence Transformers (all-MiniLM-L6-v2)

##  Configuration

This Space requires Qdrant Cloud credentials to be set in the Space settings:

- QDRANT_URL: Your Qdrant Cloud cluster URL
- QDRANT_API_KEY: Your Qdrant API key

Get free Qdrant Cloud access at [cloud.qdrant.io](https://cloud.qdrant.io)

---

*Built to demonstrate RAG architecture, vector databases, and modern AI integration* 
