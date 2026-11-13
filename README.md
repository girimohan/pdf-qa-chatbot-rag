---
title: PDF Q&A Chatbot with RAG
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# PDF Q&A Chatbot

A retrieval-augmented document chatbot built with LangChain and Qdrant. Upload
text-based PDFs or TXT files to get a document-wide summary or ask questions.
Answers cite filenames and page numbers, and the UI shows the retrieved source
passages.

## Model behavior

- **Local development:** Ollama serves `mistral:7b` on the developer's machine.
- **Hugging Face Space:** `google/flan-t5-base` runs inside the Space's Docker
  container with Transformers. The model is downloaded on first use; expect a
  slower first response on CPU hardware.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` runs locally in the
  app process.

The Space does not call Ollama on the developer's computer or depend on a hosted
FLAN-T5 inference provider. The model weights are public; no Hugging Face
Inference Providers token is required.

## Hugging Face Space setup

The Space uses process-local, in-memory Qdrant by default, even if old Cloud
secrets are still configured. No database secrets are required for a demo. To use
Qdrant Cloud, explicitly set `QDRANT_MODE=cloud` and configure `QDRANT_URL` and
`QDRANT_API_KEY` as Space secrets.

Vectors and source text are filtered by a random Streamlit session ID. Clearing a
session deletes only that session's points. In-memory data is lost when the Space
restarts. In Cloud mode, explicitly clear sessions to remove persisted data.

## Local development

1. Install Ollama and pull the model:

   ```powershell
   ollama pull mistral:7b
   ```

2. Install Python dependencies and start Streamlit:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   Copy-Item .env.template .env
   streamlit run app.py
   ```

To run FLAN-T5 locally instead, set `LLM_BACKEND=flan_t5` in `.env`. To use
Qdrant Cloud locally, set `QDRANT_MODE=cloud`, `QDRANT_URL`, and `QDRANT_API_KEY`.

## Supported files and limits

- PDF with selectable text and UTF-8 TXT. Scanned PDFs require OCR and are not
  supported.
- Maximum combined upload size per processing run: 25 MB.
- Maximum indexed size: 200 chunks per session.
- Chat history and the default in-memory Qdrant index are session-scoped.

## Tests

Run the document, citation, summarization, and Qdrant-isolation tests with:

```powershell
python -m unittest discover -s tests -v
```
