#  RAG Project – Retrieval-Augmented Generation with Local Files

This project implements a simple yet powerful **Retrieval-Augmented Generation (RAG)** pipeline using Python. It allows you to ask questions about your own documents — including `.pdf`, `.txt`, `.docx`, and more — and get intelligent answers powered by open-source models.

---


## Features

-  Ingest & chunk documents (`.pdf`, `.txt`, `.docx`)
-  Embedding using `sentence-transformers`
-  Vector search with `FAISS`
-  Answer generation using `Flan-T5`
-  Summarization of long chunks before answering
-  Works offline & on CPU (optionally GPU)
-  Fully open source

---
## System Requirements

[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

* Python 3.9

* Optional GPU for faster performance

* 5GB disk space for models and vector store
---

##  Tech Stack

| Component           | Library / Model                           |
|---------------------|--------------------------------------------|
| Embeddings          | `sentence-transformers/all-MiniLM-L6-v2`  |
| Vector DB           | `FAISS`                                   |
| LLM (Q&A)           | `google/flan-t5-base`                     |
| Summarization       | `philschmid/bart-large-cnn-samsum-sft`   |
| File Parsing        | `PyMuPDF`, `python-docx`, `langchain`     |
| Orchestration       | Plain Python + `langchain` utils          |

---

##  Installation

1. **Clone this repo**

    ```bash
    git clone https://github.com/Abhishek-Krishna-Naik/rag-project.git
    cd rag-project
    ```

##  Installation & Setup

Follow these steps to get the project up and running on your local machine.

 2. **Create and Activate Virtual Environment**

    It's highly recommended to use a virtual environment. We'll use pip for windows for this example.

    ```bash
    py -3.9 -m venv rag-env-py3.9
    ```
    Activate the enviroment (windows).
    ```bash
    rag-env-py3.9\Scripts\activate
    ```
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
---

## Usage 

1. **Put your files inside the data/ folder**

    Supported: `.pdf`, `.txt`, `.docx`, `.md`, `.json` 

2. **Run the main script**
    ```bash
    python main.py
    ```

    ```bash 
    Ask your question: What is Machine Learning?
    ```

---

The system will:

* Embed and index your files

* Retrieve relevant chunks

* Summarize them (if needed)

* Generate an answer using an open-source LLM.

---
## Customization

* LLM: Swap `flan-t5-base` with another model (`mistral`, `llama`, etc.)

* Summarizer: You can switch to `pegasus-xsum` or others if using GPU

* Embeddings: Replace with `OpenAI`, `GTE`, or `Instructor XL`

* UI: Add Streamlit or Gradio for frontend

* Fallback: Add Wikipedia / web search if no chunks found
---
## Sample Use Cases

* Ask questions from research papers or technical docs

* Build a private chatbot over company policies

* Extract insights from meeting notes, logs, etc.
---

## Acknowledgements

- [Hugging Face](https://huggingface.co/) – Models and Transformers ecosystem
- [LangChain](https://www.langchain.com/) – Document loading and processing utilities
- [FAISS by Facebook](https://github.com/facebookresearch/faiss) – Vector search engine
- [Sentence-Transformers](https://www.sbert.net/) – Document embeddings
- [PyTorch](https://pytorch.org/) – Deep learning framework
- [Transformers Library](https://github.com/huggingface/transformers) – Model pipelines
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) – PDF processing
- [python-docx](https://github.com/python-openxml/python-docx) – DOCX file support
