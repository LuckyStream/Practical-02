import os
import re
import string
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# Configuration Parameters
INDEX_DIM = 384
CHUNK_SIZE = 500      # Use 500 words per chunk
OVERLAP = 50          # Overlap 50 words between chunks
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize FAISS index (using L2 distance)
faiss_index = faiss.IndexFlatL2(INDEX_DIM)
# Global metadata list to store information for each vector (for later retrieval)
faiss_metadata = []

# Load the SentenceTransformer model
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Stop words list
STOP_WORDS = set([
    "a", "an", "the", "and", "or", "if", "in", "on", "with", "of", "at", "by", "for",
    "to", "from", "is", "are", "was", "were", "be", "been", "being"
])

def preprocess_text(text: str, remove_stop_words: bool = False) -> str:
    """
    Preprocess the text by removing extra whitespace, punctuation, and optionally stop words.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    if remove_stop_words:
        words = text.lower().split()
        words = [word for word in words if word not in STOP_WORDS]
        text = " ".join(words)
    return text

def extract_text_from_pdf(pdf_path: str):
    """
    Extracts text from a PDF file.
    Returns a list of tuples (page number, text) for each page.
    """
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """
    Splits the text into chunks of approximately 'chunk_size' words with the specified overlap.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdfs(data_dir: str, preprocess_flag: bool = True, remove_stop_words: bool = False):
    """
    Processes all PDF files in the specified directory.
    For each PDF, it extracts text, optionally preprocesses it, splits it into chunks,
    computes embeddings for each chunk, and adds the vectors to the FAISS index while saving metadata.
    """
    for file_name in os.listdir(data_dir):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            print(f"Processing file: {file_name}")
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                if preprocess_flag:
                    text = preprocess_text(text, remove_stop_words)
                chunks = split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
                for chunk_index, chunk in enumerate(chunks):
                    # Compute the embedding vector for the text chunk
                    embedding = embedding_model.encode(chunk, show_progress_bar=False)
                    embedding_np = np.array([embedding], dtype=np.float32)
                    # Add the vector to the FAISS index
                    faiss_index.add(embedding_np)
                    # Save metadata (file name, page number, chunk index, and chunk text)
                    metadata = {
                        "file": file_name,
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "chunk": chunk
                    }
                    faiss_metadata.append(metadata)

def save_index_and_metadata(index_path="faiss_index.pkl", metadata_path="faiss_metadata.pkl"):
    """
    Saves the FAISS index and metadata to disk for later retrieval.
    """
    with open(index_path, "wb") as f:
        pickle.dump(faiss_index, f)
    with open(metadata_path, "wb") as f:
        pickle.dump(faiss_metadata, f)
    print(f"Saved FAISS index to {index_path} and metadata to {metadata_path}")

def main():
    data_dir = "/Users/zouli/Desktop/DS4300/week10/Practical 02/exam_data"  # Modify this path as needed
    process_pdfs(data_dir, preprocess_flag=True, remove_stop_words=False)
    print("Processing complete. Total vectors indexed:", faiss_index.ntotal)
    save_index_and_metadata()

if __name__ == "__main__":
    main()
