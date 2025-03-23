from transformers import pipeline
from sentence_transformers import SentenceTransformer
from config import RAG_CONFIG
import os
from PyPDF2 import PdfReader
import chromadb
import docx


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
        pdf_texts = load_pdfs(RAG_CONFIG["path"])
        word_texts = load_word_docs(RAG_CONFIG["path"])
        all_chunks = []
        for text in pdf_texts:
            all_chunks.extend(chunk_text(text, chunk_size=100, overlap=5))
        # Chunk word documents by paragraphs
        for text in word_texts: 
            all_chunks.extend(text.split("\n\n"))
        # check for ''
        all_chunks = [chunk for chunk in all_chunks if chunk.strip()] 
        print(f"Total number of chunks: {len(all_chunks)}")
        print(all_chunks)

        # Generate embeddings and add to ChromaDB
        embeddings = embeddings_model.encode(all_chunks)
        collection.add(
            embeddings=embeddings.tolist(),
            documents=all_chunks,
            ids=[f"doc_{i}" for i in range(len(all_chunks))],
        )

### Load PDFs
def load_pdfs(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "rb") as file:
                    pdf = PdfReader(file)
                    document_text = ""  # Initialize for each file
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        # Normalize whitespace
                        page_text = " ".join(page_text.split())
                        document_text += f"{page_text} "
                    if page_text.strip():
                        texts.append(document_text)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return texts


### Load Word Documents
def load_word_docs(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".docx"):
            filepath = os.path.join(directory, filename)
            try:
                doc = docx.Document(filepath)
                document_text = "\n".join([para.text for para in doc.paragraphs])
                if document_text.strip():
                    texts.append(document_text)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    #  check for empty paragraphs
    return texts

### Chunk Text for PDF
def chunk_text(text, chunk_size, overlap=0):
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        # Calculate end index for current chunk
        end = min(i + chunk_size, len(words))
        # Create chunk from words
        chunk = " ".join(words[i:end])
        if chunk.strip():  # Ensure the chunk is not empty
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


### Search Documents in ChromaDB
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
