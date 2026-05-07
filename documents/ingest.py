import fitz
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"

def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page": page_num,
                "text": text.strip()
            })
    print(f"Extracted {len(pages)} pages")
    return pages

def chunk_pages(pages):
    print("Chunking text...")
    chunks = []
    for page in pages:
        text = page["text"]
        words = text.split()
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i:i + CHUNK_SIZE]
            if len(chunk_words) < 20:
                continue
            chunks.append({
                "chunk_id": len(chunks),
                "page": page["page"],
                "text": " ".join(chunk_words)
            })
    print(f"Created {len(chunks)} chunks")
    return chunks

def embed_chunks(chunks):
    print(f"Embedding chunks with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, model

def build_index(embeddings):
    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    print(f"Index built with {index.ntotal} vectors")
    return index

def save_artifacts(chunks, index, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    faiss.write_index(index, f"{output_dir}/index.faiss")
    print(f"Saved chunks and index to {output_dir}/")

def ingest(pdf_path):
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)
    embeddings, model = embed_chunks(chunks)
    index = build_index(embeddings)
    save_artifacts(chunks, index)
    return chunks, index, model

if __name__ == "__main__":
    ingest("documents/sample_model_doc.pdf")
