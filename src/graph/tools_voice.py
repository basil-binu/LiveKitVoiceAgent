import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv(".env.local")

# ==================================================
# CONFIG
# ==================================================

EMBED_MODEL = "text-embedding-3-small"
INDEX_PATH  = str(Path(__file__).parent.parent / "index")   # src/index

# ==================================================
# HELPERS
# ==================================================

def _get_retriever():
    """
    Reload FAISS index fresh on every call so newly
    uploaded files are immediately visible to the agent.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def _format_rag_docs(docs: list) -> str:
    grouped = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        grouped[source].append(doc.page_content)
    return "\n\n".join(
        f"{src}:\n" + "\n".join(chunks)
        for src, chunks in grouped.items()
    )

# ==================================================
# RAG TOOL
# ==================================================

@tool
def rag_tool(query: str) -> str:
    """Retrieve information about the homestay."""
    try:
        retriever = _get_retriever()        # fresh load every time
        docs = retriever.invoke(query)
        if not docs:
            return ""
        return _format_rag_docs(docs)
    except Exception as e:
        return f"RAG_ERROR::{e}"

# ==================================================
# TOOL REGISTRY
# ==================================================

tools = [
    rag_tool,
]