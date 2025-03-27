import sys
import redis
import chromadb
import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
import argparse

# === Global Configuration Variables ===
USE_DATABASE = "Redis"  # Options: "Redis", "ChromaDB", "FAISS"
EMBEDDING_BACKEND = "sentence-transformers"  # Options: "sentence-transformers", "ollama"

# === Redis Configuration ===
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

# === ChromaDB Configuration ===
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("embeddings")

# === FAISS Configuration ===
INDEX_DIM = 768
faiss_index = faiss.IndexFlatL2(INDEX_DIM)
faiss_vectors = []  # Should be populated during indexing
faiss_metadata = []  # Corresponding metadata for each vector

# === Embedding Model Global Variable and Cache ===
embedding_model = None
embedding_models_cache = {}  # Cache for SentenceTransformer models

def get_embedding_model():
    """
    Lazily loads the default SentenceTransformer model to avoid repeated loading.
    """
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

def get_embedding(text, model="nomic-embed-text"):
    """
    Computes the embedding vector for the given text using the selected backend.
    
    If the model is "nomic-embed-text", uses Ollama's API.
    Otherwise, uses SentenceTransformer to compute the embedding.
    
    Args:
        text (str): The input text.
        model (str): The name/identifier of the embedding model.
    
    Returns:
        list: The embedding vector.
    """
    if model == "nomic-embed-text":
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    else:
        # For SentenceTransformer models, check cache first.
        if model not in embedding_models_cache:
            print(f"Loading embedding model: {model}")
            embedding_models_cache[model] = SentenceTransformer(model)
        transformer = embedding_models_cache[model]
        return transformer.encode(text, show_progress_bar=False).tolist()

def search_embeddings(query, top_k=5):
    query_embedding = get_embedding(query)
    query_vector = np.array(query_embedding, dtype=np.float32)
    start_time = time.time()

    try:
        if USE_DATABASE == "Redis":
            q = Query("*=>[KNN {} @embedding $vec AS vector_distance]".format(top_k)) \
                .sort_by("vector_distance") \
                .return_fields("file", "page", "chunk", "vector_distance") \
                .dialect(2)
            results = redis_client.ft("embedding_index").search(q, query_params={"vec": query_vector.tobytes()})
            docs = results.docs if hasattr(results, "docs") else results
            top_results = [
                {
                    "file": get_field(doc, "file"),
                    "page": get_field(doc, "page"),
                    "chunk": get_field(doc, "chunk"),
                    "similarity": float(get_field(doc, "vector_distance") or 0)
                }
                for doc in docs
            ]
        elif USE_DATABASE == "ChromaDB":
            results = chroma_collection.query(query_embeddings=[query_embedding], n_results=top_k)
            top_results = [
                {"file": meta["file"], "page": meta["page"], "chunk": meta["chunk"],
                 "similarity": results["distances"][0][i]}
                for i, meta in enumerate(results["metadatas"][0])
            ]
        elif USE_DATABASE == "FAISS":
            distances, indices = faiss_index.search(np.array([query_vector]), top_k)
            top_results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < 0 or idx >= len(faiss_metadata):
                    continue
                top_results.append({
                    "file": faiss_metadata[idx]["file"],
                    "page": faiss_metadata[idx]["page"],
                    "chunk": faiss_metadata[idx]["chunk"],
                    "similarity": float(dist)
                })
        else:
            print(f"Unknown database: {USE_DATABASE}")
            return []
    except Exception as e:
        print(f"Search error: {e}")
        return []

    elapsed_time = time.time() - start_time
    print(f"Search time ({USE_DATABASE}): {elapsed_time:.4f} sec")
    return top_results

def get_field(doc, field_name):
    """
    Retrieves the value for field_name from doc.
    If doc is a dict, use doc.get(field_name).
    Otherwise, use getattr(doc, field_name, None).
    """
    if isinstance(doc, dict):
        return doc.get(field_name)
    else:
        return getattr(doc, field_name, None)


def generate_rag_response(query, context_results):
    """
    Generates a response using Retrieval-Augmented Generation (RAG).
    
    Args:
        query (str): User query.
        context_results (list): List of context items retrieved from search.
    
    Returns:
        str: The AI-generated answer.
    """
    context_str = "\n".join(
        [
            f"File: {result['file']} (Page {result['page']}), Content: {result['chunk']} (Similarity: {result['similarity']:.2f})"
            for result in context_results
        ]
    )
    prompt = f"""You are a professional AI assistant.
      Please provide a detailed, structured, and clear answer based on the retrieved context. 
      Ensure that your answer references relevant details from the context. If the context does not contain sufficient information, simply answer "I don't know".

Context:
{context_str}

User query:
{query}

Answer:
"""
    response = ollama.chat(model="mistral:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def interactive_search():
    """
    Provides an interactive query interface.
    """
    print(f"Using {USE_DATABASE} for search with embedding backend: {EMBEDDING_BACKEND}")
    print("Type 'exit' to quit")
    while True:
        query = input("\nPlease enter your query: ")
        if query.lower() == "exit":
            break
        context_results = search_embeddings(query)
        response = generate_rag_response(query, context_results)
        print("\n--- AI Response ---")
        print(response)

def main():
    parser = argparse.ArgumentParser(description="Search embeddings using various vector databases.")
    parser.add_argument("--db", type=str, default="Redis", choices=["Redis", "ChromaDB", "FAISS"],
                        help="Choose the vector database backend.")
    parser.add_argument("--embedding_backend", type=str, default="sentence-transformers", choices=["sentence-transformers", "ollama"],
                        help="Choose the embedding backend.")
    parser.add_argument("query", nargs="*", help="Query string (if not provided, interactive mode is used)")
    args = parser.parse_args()

    global USE_DATABASE, EMBEDDING_BACKEND
    USE_DATABASE = args.db
    EMBEDDING_BACKEND = args.embedding_backend

    if args.query:
        query = " ".join(args.query)
        context_results = search_embeddings(query)
        response = generate_rag_response(query, context_results)
        print("\n--- AI Response ---")
        print(response)
    else:
        interactive_search()

if __name__ == "__main__":
    main()

