import os
import json
import logging
from typing import Annotated, TypedDict, List
import httpx
import atexit
import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_core.runnables import RunnableConfig

from .tools_voice import tools
from .memory import memory

load_dotenv(".env.local")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TOKENS = 1000
MAX_STEPS  = 6
TIMEOUT    = 30

# LangSmith
os.environ["LANGCHAIN_TRACING_V2"]  = "true"
os.environ["LANGSMITH_ENDPOINT"]    = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]     = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"]     = "Production - Petes Inn Resort"

# -------------------------------------------------------------------
# LLM
# -------------------------------------------------------------------

_http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=10,
        max_keepalive_connections=5,
        keepalive_expiry=30,
    ),
    timeout=httpx.Timeout(
        connect=5.0,
        read=TIMEOUT,
        write=10.0,
        pool=5.0,
    ),
)

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1-mini",
    temperature=0.4,
    timeout=TIMEOUT,
    max_retries=2,
    max_tokens=500,
    http_async_client=_http_client,
)
llm_with_tools = llm.bind_tools(tools)

def _cleanup():
    try:
        asyncio.run(_http_client.aclose())
    except Exception:
        pass

atexit.register(_cleanup)

# -------------------------------------------------------------------
# STATE
# rag_sources accumulates source filenames across tool calls
# -------------------------------------------------------------------

def _merge_sources(a: List[str], b: List[str]) -> List[str]:
    return list(dict.fromkeys(a + b))   # deduplicated, order preserved

class State(TypedDict):
    messages:    Annotated[list[BaseMessage], add_messages]
    rag_sources: Annotated[List[str], _merge_sources]

# -------------------------------------------------------------------
# MESSAGE FILTERING
# -------------------------------------------------------------------

def filter_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    filtered = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            filtered.append(AIMessage(
                content=msg.content or "",
                tool_calls=getattr(msg, "tool_calls", []),
            ))
        elif isinstance(msg, ToolMessage):
            # Strip JSON wrapper — send only plain content to LLM
            raw = msg.content or ""
            try:
                parsed = json.loads(raw)
                plain  = parsed.get("content", raw)
            except Exception:
                plain = raw

            filtered.append(ToolMessage(
                content=plain,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))
    return filtered

# -------------------------------------------------------------------
# SOURCE EXTRACTION
# Reads rag_tool JSON responses and collects source filenames
# -------------------------------------------------------------------

def _extract_sources(messages: list[BaseMessage]) -> List[str]:
    sources = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name != "rag_tool":
            continue
        try:
            parsed = json.loads(msg.content or "{}")
            sources.extend(parsed.get("sources", []))
        except Exception:
            pass
    return list(dict.fromkeys(sources))

# -------------------------------------------------------------------
# LLM NODE
# -------------------------------------------------------------------

def llm_node(state: State, config: RunnableConfig) -> State:
    try:
        sender_id  = config["configurable"].get("thread_id", "unknown")
        step_count = config["configurable"].get("step_count", 0)

        if step_count >= MAX_STEPS:
            logger.warning(f"Max steps reached for {sender_id}")
            return {
                "messages": [AIMessage(content="I'm having trouble completing this. Can you simplify your request?")],
                "rag_sources": [],
            }

        trimmed = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=MAX_TOKENS,
            start_on="human",
            end_on=("human", "tool"),
        )
        clean = filter_messages(trimmed)

        # Collect any sources from tool messages in this turn
        new_sources = _extract_sources(state["messages"])

        system_prompt = config["configurable"].get("system_prompt", "")
        conversation  = [SystemMessage(content=system_prompt)] + clean

        tokens = count_tokens_approximately(conversation)
        logger.info(f"Tokens: {tokens} | Step: {step_count + 1} | User: {sender_id}")

        response = llm_with_tools.invoke(conversation)

        return {
            "messages":    [response],
            "rag_sources": new_sources,
        }

    except httpx.TimeoutException:
        logger.error(f"OpenAI timeout for {sender_id}")
        return {"messages": [AIMessage(content="Sorry, the request timed out.")], "rag_sources": []}

    except httpx.ConnectError:
        logger.error(f"OpenAI connection failed for {sender_id}")
        return {"messages": [AIMessage(content="Connection issue. Please try again.")], "rag_sources": []}

    except httpx.HTTPStatusError as e:
        logger.error(f"OpenAI HTTP error for {sender_id}: {e.response.status_code}")
        return {"messages": [AIMessage(content="Service error. Please try again.")], "rag_sources": []}

    except Exception as e:
        logger.error(f"Unexpected error for {sender_id}: {type(e).__name__}: {e}")
        return {"messages": [AIMessage(content="An error occurred. Please try again.")], "rag_sources": []}

# -------------------------------------------------------------------
# GRAPH
# -------------------------------------------------------------------

builder = StateGraph(State)
builder.add_node("LLM", llm_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "LLM")
builder.add_conditional_edges("LLM", tools_condition)
builder.add_edge("tools", "LLM")

graph = builder.compile(checkpointer=memory)

logger.info("✅ Graph ready")