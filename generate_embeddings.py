import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load chunked JSON data
with open("chunked_data.json", "r", encoding="utf-8") as f:
    chunked_data = json.load(f)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = np.array([embedding_model.encode(chunk["body"]) for chunk in chunked_data], dtype=np.float32)

# Store embeddings in chunked JSON
for i, chunk in enumerate(chunked_data):
    chunk["embedding"] = embeddings[i].tolist()

# Save updated JSON
with open("chunked_data_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunked_data, f, ensure_ascii=False, indent=4)

# Create and save FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)
faiss.write_index(index, "faiss_index.bin")

print("✅ Embeddings generated and FAISS index saved!")

