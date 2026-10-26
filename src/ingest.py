"""Load PDFs, chunk them, embed the chunks, and store them in ChromaDB.

Run from the project root:
    python -m src.ingest
    python -m src.ingest --chunk-size 400 --overlap 50 --collection docs_400
"""
import argparse
import glob
import os

import chromadb
try:
    import pymupdf as fitz  # PyMuPDF (new name)
except ImportError:
    import fitz
from sentence_transformers import SentenceTransformer

from . import config


def load_pdfs(folder=config.DATA_DIR):
    """Return a list of {text, source, page} dicts, one per non-empty page."""
    pages = []
    for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        with fitz.open(path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if text:
                    pages.append({"text": text,
                                  "source": os.path.basename(path),
                                  "page": page_num})
    return pages


def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Fixed-size character chunks with overlap."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def build_index(pages, collection=config.COLLECTION,
                chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP,
                embed_model=config.EMBED_MODEL, db_path=config.DB_PATH):
    """(Re)build a collection from scratch. Returns the number of chunks stored."""
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(collection)
    except Exception:
        pass
    col = client.create_collection(collection, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    for p in pages:
        for i, c in enumerate(chunk_text(p["text"], chunk_size, overlap)):
            ids.append(f'{p["source"]}-{p["page"]}-{i}')
            docs.append(c)
            metas.append({"source": p["source"], "page": p["page"]})

    if not docs:
        return 0

    embedder = SentenceTransformer(embed_model)
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=64).tolist()
    for i in range(0, len(ids), 500):  # Chroma limits batch size
        col.add(ids=ids[i:i + 500], documents=docs[i:i + 500],
                embeddings=embeddings[i:i + 500], metadatas=metas[i:i + 500])
    return len(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--collection", default=config.COLLECTION)
    ap.add_argument("--data", default=config.DATA_DIR)
    args = ap.parse_args()

    pages = load_pdfs(args.data)
    if not pages:
        raise SystemExit(f"No extractable text found in '{args.data}/'. "
                         "Add PDFs (scanned PDFs need OCR first).")
    print(f"Loaded {len(pages)} pages. Building index...")
    n = build_index(pages, args.collection, args.chunk_size, args.overlap)
    print(f"Done: stored {n} chunks in collection '{args.collection}'.")
