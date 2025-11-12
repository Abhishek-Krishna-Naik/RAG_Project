# embedder.py
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
from typing import List

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "index/faiss.index"
META_PATH = "index/meta.pkl"

def embed_and_store(chunks: List[str]):
    model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Create FAISS index
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    # Save metadata (texts)
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_index_and_metadata():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata
