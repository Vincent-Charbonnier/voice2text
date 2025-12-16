import os
import tempfile
import logging
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
from openai import OpenAI

from app.auth import verify_token, verify_ws_token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VoiceScribe API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model (cached)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
        whisper_model = whisper.load_model(WHISPER_MODEL)
    return whisper_model

# OpenAI client for summaries
openai_client = None

def get_openai_client():
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        openai_client = OpenAI(api_key=api_key)
    return openai_client


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class TranscriptionResponse(BaseModel):
    segments: List[TranscriptSegment]
    text: str


class SummaryRequest(BaseModel):
    text: str
    prompt: Optional[str] = None
    segments: Optional[List[TranscriptSegment]] = None


class SummaryResponse(BaseModel):
    summary: str


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    user: dict = Depends(verify_token)
):
    """Transcribe uploaded audio file using Whisper."""
    logger.info(f"Transcription request from user: {user.get('sub', 'unknown')}")
    
    # Validate file type
    if not audio.content_type or not audio.content_type.startswith(("audio/", "video/")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio or video file.")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Transcribe with Whisper
        model = get_whisper_model()
        result = model.transcribe(tmp_path, verbose=False)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Format segments
        segments = [
            TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip()
            )
            for seg in result.get("segments", [])
        ]
        
        return TranscriptionResponse(
            segments=segments,
            text=result.get("text", "").strip()
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/summarize", response_model=SummaryResponse)
async def summarize_transcript(
    request: SummaryRequest,
    user: dict = Depends(verify_token)
):
    """Generate AI summary of transcript using OpenAI."""
    logger.info(f"Summary request from user: {user.get('sub', 'unknown')}")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for summarization")
    
    try:
        client = get_openai_client()
        
        system_prompt = """You are an expert at summarizing meeting transcripts and audio recordings. 
Create clear, actionable summaries that highlight:
- Key points and decisions
- Action items and owners
- Important deadlines or dates mentioned
- Any unresolved questions or topics for follow-up"""

        user_prompt = request.text
        if request.prompt:
            user_prompt = f"Instructions: {request.prompt}\n\nTranscript:\n{request.text}"
        
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content
        return SummaryResponse(summary=summary)
        
    except Exception as e:
        logger.error(f"Summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket, token: Optional[str] = None):
    """WebSocket endpoint for real-time streaming transcription."""
    # Verify token
    user = await verify_ws_token(token)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return
    
    await websocket.accept()
    logger.info(f"WebSocket connected for user: {user.get('sub', 'unknown')}")
    
    audio_buffer = bytearray()
    
    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            # Process in chunks (e.g., every 5 seconds of audio)
            # This is a simplified example - production would need more sophisticated buffering
            if len(audio_buffer) > 80000:  # ~5 seconds at 16kHz
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                    tmp.write(bytes(audio_buffer))
                    tmp_path = tmp.name
                
                try:
                    model = get_whisper_model()
                    result = model.transcribe(tmp_path, verbose=False)
                    
                    if result.get("text", "").strip():
                        await websocket.send_json({
                            "type": "final",
                            "text": result["text"].strip(),
                            "start": 0,
                            "end": len(audio_buffer) / 16000
                        })
                finally:
                    os.unlink(tmp_path)
                
                audio_buffer.clear()
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))
