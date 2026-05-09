import os
import json
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
INDEX_PATH  = str(Path(__file__).parent.parent / "index")

# ==================================================
# HELPERS
# ==================================================

def _get_retriever():
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
# Returns JSON: { "content": "...", "sources": ["file1.pdf", ...] }
# graph_voice.py reads this to extract sources for the UI
# ==================================================

@tool
def rag_tool(query: str) -> str:
    """Retrieve information about the homestay."""
    try:
        retriever = _get_retriever()
        docs = retriever.invoke(query)

        if not docs:
            return json.dumps({"content": "", "sources": []})

        sources = list({
            doc.metadata.get("source", "unknown")
            for doc in docs
        })

        content = _format_rag_docs(docs)

        return json.dumps({
            "content": content,
            "sources": sources,
        })

    except Exception as e:
        return json.dumps({"content": f"RAG_ERROR::{e}", "sources": []})

# ==================================================
# TOOL REGISTRY
# ==================================================

tools = [
    rag_tool,
]