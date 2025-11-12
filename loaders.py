# loaders.py

import os
import fitz  # PyMuPDF for PDF
import wikipedia
from typing import List
from docx import Document  # For .docx files

def load_all_documents(folder_path: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    chunks = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        ext = filename.lower().split(".")[-1]

        if ext == "pdf":
            chunks += load_pdf(filepath, chunk_size, overlap)
        elif ext == "txt":
            chunks += load_txt(filepath, chunk_size, overlap)
        elif ext == "docx":
            chunks += load_docx(filepath, chunk_size, overlap)
        elif ext == "md":
            chunks += load_txt(filepath, chunk_size, overlap)  # treat as text
        else:
            print(f"⚠️ Unsupported file format: {filename}")
    return chunks

def load_pdf(path: str, chunk_size: int, overlap: int) -> List[str]:
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return split_text(text, chunk_size, overlap)

def load_txt(path: str, chunk_size: int, overlap: int) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return split_text(text, chunk_size, overlap)

def load_docx(path: str, chunk_size: int, overlap: int) -> List[str]:
    doc = Document(path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return split_text(text, chunk_size, overlap)

def load_wikipedia_chunks(query: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    text = wikipedia.page(query).content
    return split_text(text, chunk_size, overlap)

def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
