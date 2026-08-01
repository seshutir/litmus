"""Populate the local Chroma vectorstore that FactReasoner reads from.

FactReasoner's ``ChromaReader`` only *reads* an existing collection, using the
``all-MiniLM-L6-v2`` sentence-transformer embedding function. This script creates
(or refreshes) that collection so ``example_ai_incidents.py --context-source
chroma`` has evidence to reason over.

It ingests plain-text (``.txt``, ``.md``) and PDF (``.pdf``) files from a source
directory, splits them into overlapping chunks, and adds them to the collection
with the *same* embedding function ChromaReader expects.

Usage:
    conda run -n litmus_neurips python ingest_vectorstore.py \
        --source ./docs_evidence \
        --persist-dir ./vectorstore_sae_google \
        --collection mydocs

    # Wipe and rebuild the collection from scratch:
    conda run -n litmus_neurips python ingest_vectorstore.py \
        --source ./docs_evidence --reset
"""

import argparse
import glob
import os
from typing import List

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Must match FactReasoner's ChromaReader (fact_reasoner.core.retriever.EMBEDDING_MODEL).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    import PyPDF2

    text_parts: List[str] = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def load_documents(source_dir: str) -> List[dict]:
    """Return a list of {"path", "text"} for every supported file under source_dir."""
    docs = []
    patterns = ("**/*.txt", "**/*.md", "**/*.pdf")
    for pattern in patterns:
        for path in glob.glob(os.path.join(source_dir, pattern), recursive=True):
            try:
                text = read_pdf(path) if path.lower().endswith(".pdf") else read_txt(path)
            except Exception as e:  # noqa: BLE001 - report and skip unreadable files
                print(f"  ! skipping {path}: {type(e).__name__}: {e}")
                continue
            text = text.strip()
            if text:
                docs.append({"path": path, "text": text})
            else:
                print(f"  ! skipping {path}: no extractable text")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate the FactReasoner Chroma vectorstore."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Directory containing evidence files (.txt/.md/.pdf, searched recursively).",
    )
    parser.add_argument("--persist-dir", default="./vectorstore_sae_google")
    parser.add_argument("--collection", default="mydocs")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the collection first, then rebuild it from scratch.",
    )
    args = parser.parse_args()

    print(f"[ingest] Loading documents from {args.source} ...")
    documents = load_documents(args.source)
    if not documents:
        raise SystemExit(f"No readable .txt/.md/.pdf files found under {args.source!r}.")
    print(f"[ingest] Loaded {len(documents)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )

    chunks: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []
    for doc in documents:
        parts = splitter.split_text(doc["text"])
        base = os.path.relpath(doc["path"], args.source)
        for i, part in enumerate(parts):
            chunks.append(part)
            metadatas.append({"source": base, "chunk": i})
            ids.append(f"{base}::chunk{i}")
    print(f"[ingest] Split into {len(chunks)} chunk(s).")

    client = chromadb.PersistentClient(
        path=args.persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
    )

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"[ingest] Deleted existing collection '{args.collection}'.")
        except Exception:
            pass

    # Use the SAME embedding function ChromaReader uses on the read side.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=args.collection, embedding_function=embed_fn
    )

    # Add in batches (embeddings computed locally by the sentence-transformer).
    BATCH = 256
    for start in range(0, len(chunks), BATCH):
        end = start + BATCH
        collection.add(
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"[ingest] Added chunks {start}..{min(end, len(chunks)) - 1}")

    print(
        f"[ingest] Done. Collection '{args.collection}' now has "
        f"{collection.count()} chunk(s) in {args.persist_dir}."
    )


if __name__ == "__main__":
    main()
