import json
import faiss
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Load chunked data with embeddings
with open("chunked_data_with_embeddings.json", "r", encoding="utf-8") as f:
    chunked_data = json.load(f)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_relevant_chunks(query, k=5):
    """Retrieve top-k most relevant chunks using FAISS."""
    query_embedding = embedding_model.encode(query).reshape(1, -1)  # Convert query to embedding
    distances, indices = index.search(query_embedding, k)  # Retrieve top-k chunks

    return [chunked_data[i] for i in indices[0]]  # Get original text chunks

def query_ollama_with_context(query):
    """Retrieve relevant context and query Ollama 3.2."""
    retrieved_chunks = retrieve_relevant_chunks(query)
    context = "\n".join([chunk["body"] for chunk in retrieved_chunks])  # Combine relevant chunks

    # Formulate prompt for LLaMA
    prompt = f"Context:\n{context}\n\nQuery: {query}\nAnswer:"

    # Query Ollama
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

if __name__ == "__main__":
    query = input("Enter your query: ")
    answer = query_ollama_with_context(query)
    print("\nOllama's Answer:", answer)
