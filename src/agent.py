"""
agent.py — Pete's Inn Voice Agent (LiveKit + LangGraph)
"""

import asyncio
import logging
import random
import re
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

from livekit.plugins import sarvam, cartesia, silero
from graph.graph_voice import graph

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)



# =========================================================
# CLEANUP
# =========================================================

_CLEAN_RE = re.compile(r"http\S+|[•\*#`]")

def clean(text: str) -> str:
    return re.sub(r"\n", " ", _CLEAN_RE.sub("", text)).strip()

# =========================================================
# AGENT
# =========================================================

class PetesInnAssistant(Agent):

    def __init__(self, thread_id: str,system_prompt: str = None):

        super().__init__(
            instructions=(" "),
            allow_interruptions=True,
        )

        self._thread_id = thread_id

        logger.info(f" AGENT CREATED thread={thread_id}")

    async def on_enter(self):
        logger.info(" on_enter FIRED")

    async def on_user_turn_completed(self, turn_ctx, new_message):

        transcript = new_message.text_content

        logger.info(
            f" USER SAID: {transcript!r}"
        )

        if not transcript or not transcript.strip():
            logger.info(" Empty transcript")
            return

        session = self.session

        config = {
            "configurable": {
                "thread_id": self._thread_id
            }
        }

        loop = asyncio.get_running_loop()

        full_response = ""
        filler_sent = False

        # =========================================================
        # RUN LANGGRAPH
        # =========================================================

        def run_graph():

            nonlocal full_response

            logger.info(" graph.stream START")

            for chunk, _ in graph.stream(
                {"messages": transcript},
                config,
                stream_mode="messages",
            ):

                if not hasattr(chunk, "content"):
                    continue

                if type(chunk).__name__ not in (
                    "AIMessageChunk",
                    "AIMessage",
                ):
                    continue

                if chunk.content:
                    full_response += chunk.content

            logger.info(
                f" graph.stream DONE response={full_response!r}"
            )
        await loop.run_in_executor(
            None,
            run_graph,
        )

        # =========================================================
        # FINAL RESPONSE
        # =========================================================

        response = clean(full_response.strip())

        if not response:
            logger.info(" Empty response")
            return

        logger.info(f" FINAL RESPONSE: {response!r}")

        await session.say(
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

    logger.info(" PREWARM CALLED")

    proc.userdata["vad"] = silero.VAD.load()

    logger.info(" VAD LOADED")

server.setup_fnc = prewarm

# =========================================================
# RTC SESSION
# =========================================================

@server.rtc_session(agent_name="petes-inn-agent")
async def petes_inn_session(ctx: JobContext):

    logger.info(
        f" SESSION STARTED room={ctx.room.name}"
    )

    ctx.log_context_fields = {
        "room": ctx.room.name
    }

    # -----------------------------------------------------
    # CONNECT FIRST
    # -----------------------------------------------------

    await ctx.connect()

    logger.info(" CONNECTED")

    # -----------------------------------------------------
    # CREATE AGENT
    # -----------------------------------------------------

    agent = PetesInnAssistant(
        thread_id=ctx.room.name
    )

    # -----------------------------------------------------
    # CREATE SESSION
    # -----------------------------------------------------

    session = AgentSession(

        stt=sarvam.STT(
            language="unknown",
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

    logger.info(" SESSION OBJECT CREATED")

    # -----------------------------------------------------
    # START SESSION
    # -----------------------------------------------------

    await session.start(
        agent=agent,
        room=ctx.room,
    )

    logger.info(" SESSION STARTED")

    # -----------------------------------------------------
    # GREETING
    # -----------------------------------------------------

    await session.say(
        "Hello! Welcome to Pete's Inn Resort. How can I assist you today?",
        allow_interruptions=True,
    )

    logger.info(" GREETING SENT")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    cli.run_app(server)