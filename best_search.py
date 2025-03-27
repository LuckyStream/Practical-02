import numpy as np
import time
import pickle
import argparse
from sentence_transformers import SentenceTransformer
import ollama

# --- Global Configuration ---
INDEX_DIM = 384
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index.pkl"
METADATA_PATH = "faiss_metadata.pkl"

# Load the saved FAISS index and metadata from disk
with open(FAISS_INDEX_PATH, "rb") as f:
    faiss_index = pickle.load(f)
with open(METADATA_PATH, "rb") as f:
    faiss_metadata = pickle.load(f)

# Load the embedding model
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def get_embedding(text: str):
    """
    Computes the embedding vector for the given text.
    """
    return embedding_model.encode(text, show_progress_bar=False)

def search_embeddings(query: str, top_k: int = 5):
    """
    Performs a vector similarity search using FAISS for the given query text.
    
    Returns:
        list of dict: Each result contains the file name, page number, text chunk, and similarity (distance).
    """
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding], dtype=np.float32)
    start_time = time.time()
    distances, indices = faiss_index.search(query_vector, top_k)
    top_results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(faiss_metadata):
            continue
        meta = faiss_metadata[idx]
        top_results.append({
            "file": meta["file"],
            "page": meta["page"],
            "chunk": meta["chunk"],
            "similarity": float(dist)
        })
    elapsed_time = time.time() - start_time
    print(f"Search time (FAISS): {elapsed_time:.4f} sec")
    return top_results

def generate_rag_response(query: str, context_results: list) -> str:
    """
    Generates a response using Retrieval-Augmented Generation (RAG).
    
    Constructs a prompt that includes the retrieved context and calls the local LLM (llama3.2:latest via Ollama)
    to generate a detailed answer.
    """
    context_str = "\n".join([
        f"File: {result['file']} (Page {result['page']}), Content: {result['chunk']} (Distance: {result['similarity']:.2f})"
        for result in context_results
    ])
    prompt = f"""You are a professional AI assistant.
Please provide a concise and focused answer based on the retrieved context.
If the context does not provide sufficient information, answer "I don't know".

Context:
{context_str}

User query:
{query}

Answer:
"""
    response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def interactive_search():
    """
    Provides an interactive search interface.
    """
    print("Interactive Search Mode (Type 'exit' to quit)")
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break
        context_results = search_embeddings(query)
        answer = generate_rag_response(query, context_results)
        print("\n--- AI Response ---")
        print(answer)

def main():
    parser = argparse.ArgumentParser(description="FAISS-based search using best_search configuration.")
    parser.add_argument("--query", type=str, help="Query string. If omitted, enters interactive mode.")
    args = parser.parse_args()
    
    if args.query:
        context_results = search_embeddings(args.query)
        answer = generate_rag_response(args.query, context_results)
        print("\n--- AI Response ---")
        print(answer)
    else:
        interactive_search()

if __name__ == "__main__":
    main()

