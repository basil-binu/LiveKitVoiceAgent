import os
import time
import requests
import dateparser
from collections import defaultdict
from datetime import timezone
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS

load_dotenv(".env.local")

# ==================================================
# ICS CALENDAR
# ==================================================
EMBED_MODEL="text-embedding-3-small"





# ==================================================
# RAG (FAISS)
# ==================================================

def setup_rag():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=os.getenv("OPENAI_API_KEY"))

    vectorstore = FAISS.load_local(
        str(Path(__file__).parent.parent / "index"),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})


retriever = setup_rag()


def _format_rag_docs(docs: list) -> str:
    grouped = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        grouped[source].append(doc.page_content)
    return "\n\n".join(
        f"{src}:\n" + "\n".join(chunks)
        for src, chunks in grouped.items()
    )


@tool
def rag_tool(query: str) -> str:
    """Retrieve information about the homestay."""
    try:
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
    rag_tool

]
