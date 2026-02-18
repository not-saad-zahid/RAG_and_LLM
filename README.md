# RAG Project

A modular Retrieval-Augmented Generation (RAG) pipeline using LangChain, OpenAI, HuggingFace, and ChromaDB. This project enables you to build powerful question-answering and document retrieval systems over your own PDF/text data, with configurable LLMs, embeddings, and retrievers.

## Features
- **Plug-and-play LLMs**: Use OpenAI or HuggingFace models
- **Flexible Embeddings**: OpenAI or HuggingFace
- **ChromaDB Vector Store**: Persistent, fast document search
- **PDF Loader**: Easily index your own PDFs
- **Retriever Options**: Similarity or MMR (Maximal Marginal Relevance)
- **Streamlit UI**: Simple web interface for chat and QA

## Quickstart

### 1. Clone & Install
```bash
# Clone the repo
cd Rag_project
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. Configure
Edit `config.yaml` to set your LLM, embeddings, and data paths. Example:
```yaml
llm:
  provider: openai   # huggingface | openai
  model_name: model-name
  temperature: 0.2
  max_tokens: 512
embeddings:
  provider: openai   # huggingface | openai
  model_name: model-name
  input_path: data/docs  # Path to your PDFs
vector_store:
  type: Chroma
  persist_directory: data/docs
retriever:
  type: mmr
  top_k: 5
  fetch_k: 20
  lambda_mult: 0.7
```

### 3. Add Your Documents
Place your PDF files in the folder specified by `embeddings.input_path` (default: `data/docs`).

### 4. Run the App
```bash
streamlit run app.py
```

Or run the pipeline directly:
```bash
python main.py
```

## How It Works
- **app.py**: Streamlit web UI for chat and QA
- **main.py**: CLI pipeline for testing and debugging
- **rag_pipeline.py**: Builds the RAG chain (loads docs, splits, embeds, stores, retrieves, answers)
- **llm.py / embeddings.py**: Model selection logic
- **document_loader.py**: Loads PDFs from a folder
- **retriever.py**: Configurable retriever (similarity, MMR)

## Requirements
See `requirements.txt` for all dependencies. Key packages:
- langchain
- openai
- transformers
- chromadb
- streamlit
- pyyaml
- python-dotenv

## Customization
- Swap LLMs or embeddings by editing `config.yaml`
- Add more document loaders in `document_loader.py`
- Change retriever logic in `retriever.py`

**Author:** Saad Zahid
