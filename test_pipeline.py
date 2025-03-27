import time
import itertools
import csv
import os
import psutil
import ingest
import search
import ollama

# --- Define Pipeline Parameters ---
chunk_sizes = [200, 500, 1000]
chunk_overlaps = [0, 50, 100]
embedding_models = [
    "sentence-transformers/all-MiniLM-L6-v2", 
    "sentence-transformers/all-mpnet-base-v2", 
    "InstructorXL"
]
vector_dbs = ["Redis", "ChromaDB", "FAISS"]
llms = ["mistral:latest", "llama3.2:latest"]



# The set of user queries to test each pipeline
user_queries = [
    "What are the differences between Binary Search and Linear Search?",
    "What is the purpose of indexing in a database, and how does it improve search performance?",
    "Explain the structure and advantages of B-Trees in database systems."
]

# CSV file path and headers
csv_file_path = "/Users/zouli/Desktop/DS4300/week10/Practical 02/pipeline_results6.csv"
csv_headers = [
    "Chunk Size", "Overlap", "Embedding Model", "Vector DB", "LLM",
    "Index Time (s)", "Index Memory (MB)", "Avg Query Time (s)", "Avg Query Memory (MB)",
    "Query1", "Response1", "Query2", "Response2", "Query3", "Response3", "Score"
]

# List to store results from each pipeline configuration
results = []

def get_memory_usage_mb():
    """
    Returns the current process memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size in bytes
    return mem_bytes / (1024 * 1024)

def generate_rag_response_custom(query, context_results, llm_model):
    """
    Generates a response using Retrieval-Augmented Generation (RAG) with the specified LLM model.
    
    Args:
        query (str): User query.
        context_results (list): List of retrieved context items.
        llm_model (str): The LLM model to use.
        
    Returns:
        str: The generated answer.
    """
    # Build a context string from the search results
    context_str = "\n".join(
        [
            f"File: {result['file']} (Page {result['page']}), Content: {result['chunk']} (Similarity: {float(result['similarity']):.2f})"
            for result in context_results
        ]
    )
    # Construct the prompt with context and query
    prompt = f"""You are a professional AI assistant. Please answer the user's question based on the retrieved context.
If the context does not contain relevant information, answer "I don't know".

Context:
{context_str}

User query:
{query}

Answer:
"""
    # Call the LLM using the specified model
    response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# --- Begin Testing ---
# Iterate over all parameter combinations using itertools.product
for chunk_size, overlap, embedding_model in itertools.product(chunk_sizes, chunk_overlaps, embedding_models):
    for vector_db in vector_dbs:
        for llm_model in llms:
            # Build a unique identifier for the current pipeline configuration
            pipeline_id = f"CS{chunk_size}_OV{overlap}_{embedding_model}_{vector_db}_{llm_model}"
            print(f"\nTesting pipeline: {pipeline_id}")

            # Set the vector database to use in the search module by updating its global variable
            search.USE_DATABASE = vector_db

            # For Redis, clear the index and create a new one; for other databases, add corresponding logic if needed.
            if vector_db == "Redis":
                ingest.clear_redis_store()
                ingest.create_hnsw_index()
            
            # Record memory usage before indexing
            mem_before_index = get_memory_usage_mb()

            # Start indexing (ingest) process
            start_index = time.time()
            ingest.process_pdfs(
                data_dir="/Users/zouli/Desktop/DS4300/week10/Practical 02/data",
                chunk_size=chunk_size,
                overlap=overlap,
                embedding_model=embedding_model,
                preprocess=True,
                remove_stop_words=False
            )
            index_time = time.time() - start_index

            # Record memory usage after indexing and calculate difference
            mem_after_index = get_memory_usage_mb()
            index_memory = mem_after_index - mem_before_index

            print(f"Ingestion time: {index_time:.2f} sec, Memory used: {index_memory:.2f} MB")
            
            # For each query, record query time and memory usage
            query_times = []
            query_memories = []
            responses = []

            for query in user_queries:
                mem_before_query = get_memory_usage_mb()
                start_query = time.time()
                # Perform search for the given query (returns context results)
                context_results = search.search_embeddings(query=query, top_k=5)
                query_time = time.time() - start_query
                mem_after_query = get_memory_usage_mb()
                query_memory = mem_after_query - mem_before_query

                # Clamp negative memory changes to zero
                query_memory = max(0, query_memory)

                query_times.append(query_time)
                query_memories.append(query_memory)

                # Generate answer using the specified LLM
                response = generate_rag_response_custom(query, context_results, llm_model)
                responses.append(response)

                print(f"Query time: {query_time:.2f} sec, Memory change: {query_memory:.2f} MB")

            # Compute average query time and average query memory change
            avg_query_time = sum(query_times) / len(query_times)
            avg_query_memory = sum(query_memories) / len(query_memories)

            # Score can be manually assigned later; for now, set it as empty
            score = ""

            # Save the current configuration's results into the results list
            results.append({
                "Chunk Size": chunk_size,
                "Overlap": overlap,
                "Embedding Model": embedding_model,
                "Vector DB": vector_db,
                "LLM": llm_model,
                "Index Time (s)": index_time,
                "Index Memory (MB)": index_memory,
                "Avg Query Time (s)": f"{avg_query_time:.2f}",
                "Avg Query Memory (MB)": f"{avg_query_memory:.2f}",
                "Query1": user_queries[0],
                "Response1": responses[0],
                "Query2": user_queries[1],
                "Response2": responses[1],
                "Query3": user_queries[2],
                "Response3": responses[2],
                "Score": score
            })

# Write all results into the CSV file
with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("All results have been saved to CSV.")
