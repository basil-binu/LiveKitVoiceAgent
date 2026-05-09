import os
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv(".env.local")

# ==================================================
# CONFIG  (all paths are absolute — no CWD issues)
# ==================================================

BASE_DIR = Path(__file__).parent            # .../src/rag_creation/

DOCS_DIR    = str(BASE_DIR / "docs")
UPLOADS_DIR = str(BASE_DIR.parent / "uploads")   # .../src/uploads
FAQ_FILE    = str(BASE_DIR / "docs" / "faq_document.json")
INDEX_PATH  = str(BASE_DIR.parent / "index")     # .../src/index

EMBED_MODEL   = "text-embedding-3-small"
CHUNK_SIZE    = 350
CHUNK_OVERLAP = 100



ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# Create uploads folder automatically
Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)

# ==================================================
# LOADERS
# ==================================================

def load_files(folder_path: str) -> List[Document]:
    docs = []
    folder = Path(folder_path)

    if not folder.exists():
        return docs

    loaders = {
        ".pdf":  PyPDFLoader,
        ".txt":  TextLoader,
        ".docx": Docx2txtLoader,
    }

    for file in folder.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() == ".json":
            continue
        loader_cls = loaders.get(file.suffix.lower())
        if not loader_cls:
            continue
        loader = loader_cls(str(file))
        loaded = loader.load()
        for d in loaded:
            d.metadata = {"source": file.name}
            docs.append(d)

    return docs


def load_faq_json(path: str) -> List[Document]:
    faq_docs = []
    faq_path = Path(path)

    if not faq_path.exists():
        return faq_docs

    with open(faq_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if not q or not a:
            continue
        faq_docs.append(
            Document(
                page_content=f"{q}\n{a}",
                metadata={"source": "faq"},
            )
        )

    return faq_docs

# ==================================================
# BUILD VECTORSTORE
# ==================================================

def build_vectorstore() -> FAISS:
    print(f"DOCS_DIR:    {DOCS_DIR}")
    print(f"UPLOADS_DIR: {UPLOADS_DIR}")
    print(f"INDEX_PATH:  {INDEX_PATH}")
    print(f"FAQ_FILE:    {FAQ_FILE}")
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    raw_docs = load_files(DOCS_DIR) + load_files(UPLOADS_DIR)
    docs = splitter.split_documents(raw_docs)

    for doc in docs:
        filename = doc.metadata.get("source", "unknown")
        doc.page_content = f"Source: {filename}\n\n{doc.page_content}"

    faq_docs = load_faq_json(FAQ_FILE)
    all_docs = docs + faq_docs

    print(f"Indexed {len(all_docs)} chunks total "
          f"({len(docs)} from files, {len(faq_docs)} from FAQ)")

    vs = FAISS.from_documents(all_docs, embeddings)
    vs.save_local(INDEX_PATH)
    return vs


# ==================================================
# RUN DIRECTLY TO REBUILD
# ==================================================

if __name__ == "__main__":
    vectorstore = build_vectorstore()
    results = vectorstore.similarity_search("is there a helipad", k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {doc.metadata['source']}")
        print(doc.page_content)