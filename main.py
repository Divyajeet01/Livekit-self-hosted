"""
LiveKit Voice Agent - Transcript & Egress Only
"""

import asyncio
import logging
from datetime import datetime, timezone
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    ConversationItemAddedEvent,
)
from livekit.plugins import deepgram, noise_cancellation, openai, silero

load_dotenv()

# ===============================
# DIRECTORIES
# ===============================

BASE_DIR = Path(os.getcwd())
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
RECORDINGS_DIR = BASE_DIR / "recordings"

TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# LOGGING
# ===============================

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
logger.addHandler(_h)

# ===============================
# LLM CONFIG
# ===============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ===============================
# TRANSCRIPT BROADCASTING
# ===============================

async def broadcast_transcript(room: rtc.Room, role: str, text: str):
    """Broadcast transcript message to all participants via data channel."""
    if not room or not text or not text.strip():
        return

    try:
        if room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
            return
    except Exception:
        return

    try:
        message_data = {
            "type": "transcript",
            "role": role,
            "text": text.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        message_bytes = json.dumps(message_data).encode("utf-8")

        if not room.local_participant:
            return

        await room.local_participant.publish_data(message_bytes, reliable=True)
        logger.info(f"[broadcast] ✅ {role}: {text[:60]}")
    except Exception as e:
        logger.error(f"[broadcast] ❌ {e}")


# ===============================
# AGENT
# ===============================

_current_room: rtc.Room = None


class QRNAgent(Agent):
    def __init__(self, transcript_file_path: str, instructions: str):
        super().__init__(instructions=instructions)
        self.transcript_file_path = transcript_file_path
        self._file = None
        self._room = None

    async def on_enter(self):
        global _current_room

        self._room = _current_room

        try:
            os.makedirs(os.path.dirname(self.transcript_file_path), exist_ok=True)
            self._file = open(self.transcript_file_path, "a", encoding="utf-8")
            logger.info(f"[on_enter] Transcript file: {self.transcript_file_path}")
        except Exception as e:
            logger.error(f"[on_enter] Could not open transcript: {e}")
            self._file = None

        if self._file:
            self.session.on("conversation_item_added", self.on_conversation_item_added)

        try:
            greeting = "Hello! How can I help you today?"
            await self.session.say(greeting)
            if self._room:
                await broadcast_transcript(self._room, "agent", greeting)
        except Exception as e:
            logger.warning(f"[on_enter] Could not say greeting: {e}")

    def on_conversation_item_added(self, event: ConversationItemAddedEvent):
        if self._room is None:
            return

        text = getattr(event.item, "text_content", None)
        role = getattr(event.item, "role", "unknown")

        if not text:
            return

        # Write to transcript file
        if self._file:
            line = f"| {role.upper()} | {text}\n"
            try:
                self._file.write(line)
                self._file.flush()
            except Exception as e:
                logger.warning(f"[transcript] File write failed: {e}")
            print(line, end="")

        # Broadcast to client
        if self._room:
            transcript_role = "user" if role.lower() == "user" else "agent"
            asyncio.create_task(broadcast_transcript(self._room, transcript_role, text))

    async def on_exit(self):
        global _current_room

        if self._file:
            try:
                self._file.close()
            except Exception:
                pass

        _current_room = None
        self._room = None
        logger.info("[on_exit] Agent exit complete")


# ===============================
# SERVER SETUP
# ===============================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    global _current_room

    job_id = ctx.job.id
    room_name = getattr(ctx.room, "name", "?")
    logger.info("[entrypoint] job_id=%s room=%s", job_id, room_name)

    if not (os.getenv("DEEPGRAM_API_KEY") or "").strip():
        raise RuntimeError("DEEPGRAM_API_KEY required")

    await ctx.connect(
        auto_subscribe=AutoSubscribe.AUDIO_ONLY,
        rtc_config=rtc.RtcConfiguration(
            ice_transport_type=rtc.IceTransportType.TRANSPORT_ALL,
        ),
    )

    _current_room = ctx.room
    logger.info(f"[entrypoint] Room connected: {ctx.room.name}")

    await asyncio.sleep(2)

    transcript_path = str(TRANSCRIPT_DIR / f"{job_id}.txt")
    audio_filename = f"{job_id}.mp3"
    audio_path = f"out/{audio_filename}"

    _lk_http = (
        (os.getenv("LIVEKIT_URL", "ws://localhost:7880") or "")
        .strip()
        .replace("ws://", "http://")
        .replace("wss://", "https://")
    )

    lkapi = api.LiveKitAPI(
        _lk_http,
        os.getenv("LIVEKIT_API_KEY", "devkey"),
        os.getenv("LIVEKIT_API_SECRET", "secret"),
    )

    # ===============================
    # EGRESS
    # ===============================
    egress_id = None
    if os.getenv("SKIP_EGRESS", "1").strip().lower() in ("0", "false", "no"):
        logger.info("[entrypoint] Starting egress...")
        try:
            egress_request = api.RoomCompositeEgressRequest(
                room_name=ctx.room.name,
                audio_only=True,
                file_outputs=[
                    api.EncodedFileOutput(
                        file_type=api.EncodedFileType.MP3,
                        filepath=audio_path,
                    )
                ],
            )
            info = await lkapi.egress.start_room_composite_egress(egress_request)
            egress_id = info.egress_id
            logger.info(f"[entrypoint] Egress started: {egress_id}")
        except Exception as e:
            logger.warning(f"[entrypoint] Egress failed: {e}")
    else:
        logger.info("[entrypoint] Egress skipped (SKIP_EGRESS=1)")

    # ===============================
    # SESSION
    # ===============================
    tts_model = os.getenv("TTS_MODEL", "aura-2-phoebe-en")
    stt_model = os.getenv("STT_MODEL", "nova-3")
    agent_instructions = os.getenv("AGENT_INSTRUCTIONS", "You are a helpful voice assistant.")

    session = AgentSession(
        stt=deepgram.STT(model=stt_model),
        tts=deepgram.TTS(model=tts_model),
        llm=openai.LLM(
            model=OPENAI_MODEL,
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
        ),
    )

    try:
        await session.start(
            agent=QRNAgent(transcript_path, instructions=agent_instructions),
            room=ctx.room
        )
        logger.info("[entrypoint] Agent running")

        while True:
            if _current_room is None:
                break
            try:
                if len(ctx.room.remote_participants) == 0:
                    logger.info("[entrypoint] User disconnected")
                    break
            except Exception:
                pass
            try:
                if ctx.room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.3)

    except Exception as e:
        logger.exception("[entrypoint] Session failed: %s", e)
        raise
    finally:
        await lkapi.aclose()
        _current_room = None
        logger.info("[entrypoint] Cleanup complete")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )
