from transformers import pipeline
from sentence_transformers import SentenceTransformer
from config import RAG_CONFIG
import os
from PyPDF2 import PdfReader
import chromadb

# Initialize the embeddings model
embeddings_model = SentenceTransformer("intfloat/e5-large-v2")

# Create or get collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Initialize ChromaDB client
collection = chroma_client.get_or_create_collection(
    name="RagDocuments",
    metadata={
        "hnsw:space": "cosine"
    },  # cosine similarity will be used to measure the distance between vectors
)

def initRAG(device):
    # Initialize documents if collection is empty
    if collection.count() == 0:
        print("Loading documents into ChromaDB...")
        texts = load_pdfs(RAG_CONFIG["path"])
        all_chunks = []
        for text in texts:
            all_chunks.extend(chunk_text(text, chunk_size=100, overlap=5))

        # Generate embeddings and add to ChromaDB
        embeddings = embeddings_model.encode(all_chunks)
        collection.add(
            embeddings=embeddings.tolist(),
            documents=all_chunks,
            ids=[f"doc_{i}" for i in range(len(all_chunks))],
        )

def load_pdfs(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "rb") as file:
                pdf = PdfReader(file)
                for page in pdf.pages:
                    texts.append(page.extract_text())
    return texts


def chunk_text(text, chunk_size=200, overlap=0):
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        # Calculate end index for current chunk
        end = min(i + chunk_size, len(words))
        # Create chunk from words
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        # Move index forward by chunk_size - overlap
        i += chunk_size - overlap

        # If near the end and have leftover words that are less than overlap
        if i < len(words) and len(words) - i < overlap:
            break

    # Add final chunk if there are remaining words
    if i < len(words):
        chunks.append(" ".join(words[i:]))

    return chunks


def search_docs(query, top_k=3):
    query_embedding = embeddings_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()], n_results=top_k
    )
    
    formatted_results = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        distance = results["distances"][0][i] if "distances" in results else 0
        similarity = 1 - distance  # Convert distance to similarity score
        
        formatted_result = {
            "content": doc,
            "similarity_score": f"{similarity:.2f}",
        }
        formatted_results.append(formatted_result)
    
    return formatted_results
