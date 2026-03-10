# RAG Legal Search – Title 17 US Code

A Retrieval-Augmented Generation (RAG) system for querying **U.S. copyright law (Title 17)** using a hybrid search pipeline (BM25 + embeddings) and LLMs (Groq).

## Features
- Hybrid retrieval: semantic + keyword
- Chunked and embedded legal code
- LLM answers with relevant legal context
- Fully local preprocessing

## Installation

```bash
git clone https://github.com/StecStecStec/RAG-Legal-Search.git
cd RAG-Legal-Search
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Author
Michał Kryspin Stec