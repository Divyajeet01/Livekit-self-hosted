"""
WebRTC Service for LiveKit Voice Agent
Provides token generation and room management for remote clients
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
try:
    BaseModel.model_config["protected_namespaces"] = ()
except Exception:
    pass

from livekit import api

load_dotenv()

# LiveKit configuration (defaults set for local docker-compose)
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secretsecretsecretsecretsecretsecret")


class TokenRequest(BaseModel):
    """Request model for token generation"""

    room_name: str | None = None
    participant_name: str | None = None


class TokenResponse(BaseModel):
    """Response model containing connection details"""

    token: str
    room_name: str
    participant_name: str
    livekit_url: str


class RoomInfo(BaseModel):
    """Room information model"""

    room_name: str
    participant_count: int
    created_at: datetime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    print("🚀 WebRTC Service starting...")
    print(f"📡 LiveKit URL: {LIVEKIT_URL}")

    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        print("⚠️  Warning: LIVEKIT_API_KEY or LIVEKIT_API_SECRET not configured")

    yield
    print("👋 WebRTC Service shutting down...")


app = FastAPI(
    title="LiveKit Voice Agent WebRTC Service",
    description="WebRTC service for remote voice agent access",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_room_token(room_name: str, participant_name: str) -> str:
    """Generate a LiveKit access token for a room and participant"""
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(participant_name).with_name(participant_name).with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
        )
    )
    return token.to_jwt()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "livekit-voice-agent-webrtc",
        "livekit_configured": bool(LIVEKIT_API_KEY and LIVEKIT_API_SECRET),
    }


@app.post("/api/token", response_model=TokenResponse)
async def create_token(request: TokenRequest):
    """
    Generate a LiveKit access token for joining a room
    """
    # Generate room name if not provided
    room_name = request.room_name or f"voice-agent-{uuid.uuid4().hex[:8]}"
    participant_name = request.participant_name or f"user-{uuid.uuid4().hex[:6]}"

    token = generate_room_token(room_name, participant_name)

    return TokenResponse(
        token=token,
        room_name=room_name,
        participant_name=participant_name,
        livekit_url=LIVEKIT_URL,
    )


@app.get("/api/rooms")
async def list_rooms():
    """
    List active rooms (requires admin API access)
    """
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(
            status_code=500, detail="LiveKit API credentials not configured"
        )

    try:
        lkapi = api.LiveKitAPI(
            LIVEKIT_URL,
            LIVEKIT_API_KEY,
            LIVEKIT_API_SECRET,
        )
        rooms = await lkapi.room.list_rooms(api.ListRoomsRequest())
        await lkapi.aclose()

        return {
            "rooms": [
                {
                    "name": room.name,
                    "num_participants": room.num_participants,
                    "creation_time": room.creation_time,
                }
                for room in rooms.rooms
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Entry point for the WebRTC service"""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(f"🌐 Starting WebRTC service at http://{host}:{port}"
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
