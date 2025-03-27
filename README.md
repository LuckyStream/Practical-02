# Practical 02: Local RAG System

## Overview

This project builds a local Retrieval-Augmented Generation (RAG) system。

- Ingests and indexes course notes from the semester
- Retrieves relevant context using different vector databases and embedding models
- Passes the context to a locally running LLM to generate answers
- Benchmarks the system across 162 different pipeline configurations

---

## Pipelines We Tested

To evaluate the RAG system, we varied the following key components:

- **Chunking Strategies**:
  - Chunk Sizes: 200, 500, 1000
  - Overlap Sizes: 0, 50, 100
- **Embedding Models**:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `sentence-transformers/all-mpnet-base-v2`
  - `InstructorXL`
- **Vector Databases**:
  - `Redis`
  - `ChromaDB`
  - `FAISS`
- **LLMs**:
  - `llama3.2:latest`
  - `mistral:instruct`

We evaluated each of the 162 pipelines on:
- Indexing time & memory
- Query time & memory
- Qualitative scoring of answers to 3 user questions

---

## Best Pipeline (Selected)

After three rounds of testing, we selected the following as the **best-performing pipeline**:

- **Chunk Size**: 500  
- **Overlap**: 50  
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`  
- **Vector DB**: `FAISS`  
- **LLM**: `llama3.2:latest`  

### Why this Pipeline?

1. **Balanced Performance**:  
   It achieved the best combination of low indexing/query time and memory usage, while still delivering high-quality answers.

2. **Output Quality**:  
   It consistently produced the most accurate and contextually relevant responses to our test questions, as judged manually.

---

## Usage

To run the full experiment and generate the results CSV:

```bash
python test_pipeline.py
```

## Team

Li Zou

