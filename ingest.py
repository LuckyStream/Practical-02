import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import re
import string
from sentence_transformers import SentenceTransformer

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Stop words list
STOP_WORDS = set([
    "a", "an", "the", "and", "or", "if", "in", "on", "with", "of", "at", "by", "for", 
    "to", "from", "is", "are", "was", "were", "be", "been", "being"
])

# Dictionary to cache loaded SentenceTransformer models
embedding_models_cache = {}

# Mapping for model identifiers
embedding_model_mapping = {
    "InstructorXL": "hkunlp/instructor-xl"  # Use the correct identifier for InstructorXL
}

def clear_redis_store():
    """Clears the existing Redis store."""
    redis_client.flushdb()

def create_hnsw_index():
    """Creates an HNSW index in Redis for vector search."""
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """
    Generates an embedding for the given text using the specified model.
    
    If the model is "nomic-embed-text", it uses Ollama's API.
    Otherwise, it assumes the model is a SentenceTransformer model and loads it accordingly.
    
    Args:
        text (str): The input text.
        model (str): The name of the embedding model.
    
    Returns:
        list: The embedding vector.
    """
    # If the model is "InstructorXL", update it to the proper Hugging Face ID.
    if model in embedding_model_mapping:
        model = embedding_model_mapping[model]
    
    if model == "nomic-embed-text":
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    else:
        # Use SentenceTransformer for other embedding models.
        if model not in embedding_models_cache:
            print(f"Loading embedding model: {model}")
            embedding_models_cache[model] = SentenceTransformer(model)
        transformer = embedding_models_cache[model]
        return transformer.encode(text, show_progress_bar=False).tolist()

def store_embedding(file: str, page: str, chunk_index: str, chunk_text: str, embedding: list):
    """
    Stores the embedding in Redis with the associated metadata.
    
    Args:
        file (str): The source file name.
        page (str): The page number.
        chunk_index (str): The index of the text chunk.
        chunk_text (str): The actual text chunk.
        embedding (list): The embedding vector.
    """
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk_index}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk_index": chunk_index,
            "text": chunk_text,  # Store original text for debugging and verification
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_path (str): The path to the PDF file.
    
    Returns:
        list: A list of tuples (page number, text) for each page.
    """
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

def preprocess_text(text: str, remove_stop_words: bool = False) -> str:
    """
    Preprocesses text by removing extra whitespace, punctuation, and optionally stop words.
    
    Args:
        text (str): The original text.
        remove_stop_words (bool): Whether to remove stop words.
        
    Returns:
        str: The preprocessed text.
    """
    # Remove newlines and extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    if remove_stop_words:
        # Simple stop words removal (convert to lowercase first)
        words = text.lower().split()
        words = [word for word in words if word not in STOP_WORDS]
        text = " ".join(words)
    
    return text

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """
    Splits the text into chunks of approximately 'chunk_size' words with specified overlap.
    
    Args:
        text (str): The preprocessed text.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between chunks.
        
    Returns:
        list: A list of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdfs(data_dir, chunk_size=300, overlap=50, embedding_model="nomic-embed-text", preprocess=False, remove_stop_words=False):
    """
    Processes all PDF files in the specified directory and stores their embeddings in the vector database.
    
    Args:
        data_dir (str): Directory containing PDF files.
        chunk_size (int): Number of words per text chunk.
        overlap (int): Number of overlapping words between chunks.
        embedding_model (str): The embedding model to use.
        preprocess (bool): Whether to preprocess the raw text.
        remove_stop_words (bool): Whether to remove stop words during preprocessing.
    """
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                # Preprocess text if required
                if preprocess:
                    text = preprocess_text(text, remove_stop_words)
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, model=embedding_model)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk_index=str(chunk_index),
                        chunk_text=chunk,
                        embedding=embedding,
                    )

def query_redis(query_text: str):
    """
    Performs a vector similarity search on Redis using the query text.
    
    Args:
        query_text (str): The query text.
    """
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    # Example query
    embedding = get_embedding(query_text)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")

def main():
    clear_redis_store()
    create_hnsw_index()
    # Process PDFs with options to preprocess text and remove stop words
    process_pdfs("/Users/zouli/Desktop/DS4300/week10/Practical 02/data", chunk_size=300, overlap=50, embedding_model="nomic-embed-text", preprocess=True, remove_stop_words=False)
    print("\n---Done processing PDFs---\n")
    query_redis("Efficient search in vector databases")

if __name__ == "__main__":
    main()
