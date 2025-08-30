# 📚 PDF Q&A Chatbot with RAG

An intelligent document question-answering system built with **Retrieval-Augmented Generation (RAG)** architecture. Upload your documents and chat with them using AI!

[![Deploy to HuggingFace Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/deploy-to-spaces-md.svg)](https://huggingface.co/spaces)

## ✨ What it does

- **Upload documents** (PDF, TXT, DOCX) and ask questions about their content
- **Smart retrieval** using vector search to find relevant document sections  
- **Dual AI setup**: Local privacy (Ollama) + Cloud deployment (HuggingFace)
- **Professional vector storage** with Qdrant Cloud

## 🛠️ Tech Stack

- **Frontend**: Streamlit with custom UI
- **Vector Database**: Qdrant Cloud (persistent storage)
- **AI Models**: 
  - 🏠 **Local**: Ollama/Mistral (private, offline)
  - ☁️ **Production**: HuggingFace FLAN-T5 (free, cloud)
- **Framework**: LangChain for RAG pipeline

## 🚀 Deployment Options

### 🤗 HuggingFace Spaces (Recommended)

**Automatic deployment from this GitHub repo:**

1. **Fork this repository**
2. **Create HF Space** and connect to your forked repo
3. **Add secrets** in Space settings:
   ```
   QDRANT_URL=your_qdrant_cloud_url
   QDRANT_API_KEY=your_api_key
   ```
4. **Auto-sync**: Push to GitHub → Automatically deploys to HF Spaces!

### ☁️ Streamlit Community Cloud

1. **Connect this GitHub repo** to Streamlit Cloud
2. **Add secrets** in app settings:
   ```
   QDRANT_URL=your_qdrant_cloud_url
   QDRANT_API_KEY=your_api_key
   HUGGINGFACE_API_KEY=your_hf_token
   ```

## 💡 Quick Start (Local Development)

### 🏠 Local with Ollama (Private)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Ollama**:
   ```bash
   # Install from https://ollama.ai
   ollama pull mistral:latest
   ollama serve
   ```

3. **Configure environment** (copy `.env.template` to `.env`):
   ```
   ENVIRONMENT=local
   QDRANT_URL=your_qdrant_cloud_url
   QDRANT_API_KEY=your_api_key
   ```

4. **Run**:
   ```bash
   streamlit run app.py
   ```

### ☁️ Cloud Development

1. **Get HuggingFace token**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **Set environment**:
   ```
   ENVIRONMENT=production
   HUGGINGFACE_API_KEY=your_hf_token
   QDRANT_URL=your_qdrant_cloud_url
   QDRANT_API_KEY=your_api_key
   ```

## 🌐 Live Demo

- **HuggingFace Spaces**: [View Demo](#) *(Add your HF Spaces URL)*
- **Streamlit Cloud**: [View Demo](#) *(Add your Streamlit URL)*

## 🔧 Features

- **📄 Multi-format support**: PDF, TXT, DOCX
- **🔍 Semantic search**: Find relevant document sections
- **💬 Chat interface**: Natural conversation with your documents
- **📝 Source citations**: Know which documents provided answers
- **🔄 Environment switching**: Local privacy or cloud deployment
- **⚡ Auto-sync**: GitHub → HF Spaces deployment

## 📂 Project Structure

```
rag-project/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.template      # Environment variables template
├── .streamlit/        # Streamlit configuration
├── Procfile          # Deployment configuration
└── README.md         # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

MIT License - feel free to use for personal and commercial projects.

---

**🎯 Built for Portfolio Demonstration**

*Showcasing RAG architecture, vector databases, environment management, and modern AI integration with professional deployment workflows.*

**🔗 Connect:**
- GitHub: [Your GitHub Profile](#)
- LinkedIn: [Your LinkedIn](#)
- Portfolio: [Your Portfolio Site](#)
