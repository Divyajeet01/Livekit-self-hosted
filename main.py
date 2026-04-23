import asyncio
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

from livekit import api  
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    ConversationItemAddedEvent
)
from livekit.plugins import deepgram, openai, silero

load_dotenv()
logging.getLogger("livekit").setLevel(logging.ERROR)

# ===============================
# AGENT DEFINITION
# ===============================

class MyAgent(Agent):
    def __init__(self, transcript_file_path: str):
        super().__init__(instructions="Your Helpful Assistant. Provide output in plain text.")
        self.transcript_file_path = transcript_file_path

    async def on_enter(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.transcript_file_path), exist_ok=True)
        self._file = open(self.transcript_file_path, "a", encoding="utf-8")
        self.session.on("conversation_item_added", self.on_conversation_item_added)
        await self.session.say(
            "Hello! I’m your assistant. How can I help you today?"
        )
    
    def on_conversation_item_added(self, event: ConversationItemAddedEvent):
        text = getattr(event.item, "text_content", "")
        if text:
            line = f"{datetime.utcnow()} | {event.item.role.upper()} | {text}\n"
            self._file.write(line)
            self._file.flush()
            print(line, end="")

    async def on_exit(self):
        if hasattr(self, "_file"):
            self._file.close()

# ===============================
# SERVER SETUP
# ===============================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server = AgentServer(setup_fnc=prewarm)

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    await asyncio.sleep(2)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_id = ctx.job.id
    room_name = ctx.room.name

    transcript_path = (
        rf"..\transcript\{job_id}.txt"
    )

    # --- AGENT SESSION ---
    session = AgentSession(
        stt=deepgram.STTv2(model="flux-general-en"),
        tts=deepgram.TTS(model="aura-asteria-en"),
        llm=openai.LLM(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            base_url="https://.../v1",
        ),
    )

    try:
        await session.start(agent=MyAgent(transcript_path), room=ctx.room)
        
        # Keep entrypoint alive as long as participants are in the room
        while len(ctx.room.remote_participants) > 0:
            await asyncio.sleep(1)
            
    finally:
        # Clean up
        print("🛑 Session finished. Closing API...")
   

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
