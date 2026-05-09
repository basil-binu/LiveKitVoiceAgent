import asyncio
import logging
import re
import json

from dotenv import load_dotenv

load_dotenv(".env.local")

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
)

from livekit.plugins import (
    sarvam,
    cartesia,
    silero,
)

from graph.graph_voice import graph

logger = logging.getLogger("agent")

logging.basicConfig(level=logging.INFO)

# =========================================================
# CLEANUP
# =========================================================

_CLEAN_RE = re.compile(
    r"http\S+|[•\*#`]"
)

def clean(text: str):

    return re.sub(
        r"\n",
        " ",
        _CLEAN_RE.sub("", text),
    ).strip()

# =========================================================
# AGENT
# =========================================================

class VoiceAssistant(Agent):

    def __init__(self, thread_id: str):

        super().__init__(
            instructions=" ",
            allow_interruptions=True,
        )

        self._thread_id = thread_id
        self._room = None

        logger.info(
            f"AGENT CREATED thread={thread_id}"
        )

    async def on_enter(self):

        logger.info("on_enter FIRED")

    async def on_user_turn_completed(
        self,
        turn_ctx,
        new_message,
    ):

        transcript = new_message.text_content

        logger.info(
            f"USER SAID: {transcript!r}"
        )

        if (
            not transcript
            or not transcript.strip()
        ):
            return

        # =====================================================
        # GET ROOM METADATA
        # =====================================================

        room_info = self._room

        metadata = json.loads(
            room_info.metadata or "{}"
        )

        system_prompt = metadata.get(
            "system_prompt",
            "",
        )

        logger.info(
            f"SYSTEM PROMPT: {system_prompt}"
        )

        # =====================================================
        # LANGGRAPH CONFIG
        # =====================================================

        config = {
            "configurable": {
                "thread_id":
                    self._thread_id,

                "system_prompt":
                    system_prompt,
            }
        }

        loop = asyncio.get_running_loop()

        full_response = ""
        rag_sources   = []

        # =====================================================
        # RUN GRAPH
        # =====================================================

        def run_graph():

            nonlocal full_response, rag_sources

            logger.info(
                "graph.stream START"
            )

            for chunk, _ in graph.stream(
                {"messages": transcript, "rag_sources": []},
                config,
                stream_mode="messages",
            ):

                if not hasattr(
                    chunk,
                    "content",
                ):
                    continue

                if type(chunk).__name__ not in (
                    "AIMessageChunk",
                    "AIMessage",
                ):
                    continue

                if chunk.content:
                    full_response += (
                        chunk.content
                    )

            # Get final state to read rag_sources
            try:
                final_state = graph.get_state(config)
                rag_sources = final_state.values.get("rag_sources", [])
                logger.info(
                    f"RAG SOURCES: {rag_sources}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not read rag_sources from state: {e}"
                )

            logger.info(
                f"graph.stream DONE response={full_response!r}"
            )

        await loop.run_in_executor(
            None,
            run_graph,
        )

        # =====================================================
        # FINAL RESPONSE
        # =====================================================

        response = clean(
            full_response.strip()
        )

        if not response:
            return

        logger.info(
            f"FINAL RESPONSE: {response!r}"
        )

        # =====================================================
        # SEND RAG SOURCES TO FRONTEND
        # =====================================================

        if rag_sources:
            try:
                payload = json.dumps({
                    "type":    "rag_sources",
                    "sources": rag_sources,
                }).encode("utf-8")

                await self._room.local_participant.publish_data(
                    payload,
                    reliable=True,
                )
                logger.info(
                    f"RAG SOURCES SENT: {rag_sources}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not send rag_sources: {e}"
                )

        await self.session.say(
            response,
            allow_interruptions=True,
        )

# =========================================================
# SERVER
# =========================================================

server = AgentServer()

# =========================================================
# PREWARM
# =========================================================

def prewarm(proc: JobProcess):

    logger.info("PREWARM CALLED")

    proc.userdata["vad"] = (
        silero.VAD.load()
    )

    logger.info("VAD LOADED")

server.setup_fnc = prewarm

# =========================================================
# RTC SESSION
# =========================================================

@server.rtc_session(
    agent_name="voice-agent"
)
async def voice_session(
    ctx: JobContext,
):

    logger.info(
        f"SESSION STARTED room={ctx.room.name}"
    )

    ctx.log_context_fields = {
        "room": ctx.room.name
    }

    # =====================================================
    # CONNECT
    # =====================================================

    await ctx.connect()

    logger.info("CONNECTED")

    # =====================================================
    # AGENT
    # =====================================================

    agent = VoiceAssistant(
        thread_id=ctx.room.name
    )

    # IMPORTANT
    agent._room = ctx.room

    # =====================================================
    # SESSION
    # =====================================================

    session = AgentSession(

        stt=sarvam.STT(
            language="en-IN",
            model="saaras:v3",
            mode="transcribe",
        ),

        llm=None,

        tts=cartesia.TTS(
            model="sonic-3",

            voice="6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
        ),

        vad=ctx.proc.userdata["vad"],
    )

    logger.info(
        "SESSION OBJECT CREATED"
    )

    # =====================================================
    # START SESSION
    # =====================================================

    await session.start(
        agent=agent,
        room=ctx.room,
    )

    logger.info("SESSION STARTED")

    # =====================================================
    # GREETING
    # =====================================================

    await session.say(
        "Hello! How can I help you today?",
        allow_interruptions=True,
    )

    logger.info("GREETING SENT")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    cli.run_app(server)