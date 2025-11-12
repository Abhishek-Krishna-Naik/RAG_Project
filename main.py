# main.py
from loaders import load_all_documents, load_wikipedia_chunks
from embedder import embed_and_store, load_index_and_metadata
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np

# Load tokenizer and model once
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer=tokenizer, device=-1)

def search_and_answer(query: str, top_k: int = 1,  min_score: float = 0.5):
    index, metadata = load_index_and_metadata()
    model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    print("Using device:", model.device)

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)

    scores = D[0]
    if max(scores) < min_score:
        print("🤖 I'm not confident I can answer that based on the provided documents.")
        return

    # print("\n🔍 Top Retrieved Chunks:\n")
    # for i in I[0]:
    #     print("------")
    #     print(metadata[i])
    
    # Simple answer: concatenate top chunks
    print("\n🧠 Suggested Answer (Raw Context):\n")
    print(" ".join([metadata[i] for i in I[0]]))

    retrieved_chunks = [metadata[i] for i in I[0]]

    # print("\n🔍 Top Chunks Retrieved:\n")
    # for i, chunk in enumerate(retrieved_chunks, 1):
    #     print(f"\n-- Chunk {i} --\n{chunk[:300]}...\n")  # Print only first 300 chars

    print("\n🧠 Final Answer (LLM):\n")
    answer = generate_answer_with_llm(retrieved_chunks, query)
    print(answer)

def generate_answer_with_llm(chunks: list, query: str, max_input_tokens=512):
    # Start with all chunks, then reduce if needed
    while chunks:
        context = "\n".join(chunks)
        prompt = f"Answer this briefly:\n\nContext:\n{context}\n\nQuestion: {query}"

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"]
        if input_ids.shape[1] <= max_input_tokens:
            break
        # Remove the last chunk and try again
        chunks.pop()

    # If even one chunk is too long, truncate the input
    prompt = f"Answer this briefly:\n\nContext:\n{context}\n\nQuestion: {query}"
    inputs = tokenizer(prompt, truncation=True, max_length=max_input_tokens, return_tensors="pt")
    decoded_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    # Generate answer
    output = qa_pipeline(decoded_prompt, max_new_tokens=100)[0]["generated_text"]
    return output.strip()

if __name__ == "__main__":
    # Step 1: Load + chunk documents
    all_doc_chunks = load_all_documents("data/")
    wiki_chunks = load_wikipedia_chunks("Artificial intelligence")
    all_chunks = all_doc_chunks + wiki_chunks

    print(f"Total Chunks: {len(all_chunks)}")

    # Step 2: Embed + save to FAISS
    embed_and_store(all_chunks)

    # Step 3: Ask a question
    query = input("\n❓ Ask your question: ")
    search_and_answer(query)
