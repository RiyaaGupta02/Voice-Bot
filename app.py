import os
import base64
import tempfile
import io
import wave
import time
from typing import Dict
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.requests import Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

from pydub import AudioSegment

# FFmpeg Path Override
import pydub.utils
FFMPEG_BIN_PATH = r"C:\Users\HP\Desktop\MACHINE LEARNING STUDY\ML_MiniProjects\Projects_LLMBased\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"

try:
    pydub.AudioSegment.converter = FFMPEG_BIN_PATH
    print(f"✅ pydub converter path set to: {FFMPEG_BIN_PATH}")
except Exception as e:
    print(f"⚠️  Could not set pydub converter path: {e}")

load_dotenv()

from STT_Services import transcribe_audio
from Intent_Router_Thinkingprocess import process_user_query
from TTS_Service import speak_text
from VAD_Service import VADService

app = FastAPI(title="Real Estate Voice Bot - Natural Conversations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vad_service = VADService()


def convert_webm_to_wav(webm_path: str, wav_path: str) -> bool:
    """Convert WebM to WAV (mono, 16kHz)"""
    try:
        print(f"[CONVERT] Converting {webm_path} to WAV...")
        audio = AudioSegment.from_file(webm_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_path, format="wav")
        
        if os.path.exists(wav_path):
            print(f"[CONVERT] ✅ Successfully converted to {wav_path}")
            return True
        return False
        
    except Exception as e:
        print(f"[CONVERT ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


class ConversationSession:
    def __init__(self, session_id: str):
        self.id = session_id
        self.history = []
        self.created_at = time.time()
        self.last_activity = time.time()
        self.processing_lock = False  # Prevent race conditions
        self.last_response_id = None  # Track last response to prevent duplicates
    
    def add_turn(self, user_query: str, bot_response: str, response_id: str = None):
        # Prevent duplicate entries
        if response_id and response_id == self.last_response_id:
            print(f"[SESSION] Skipping duplicate response: {response_id}")
            return False
        
        self.history.append({
            "user": user_query,
            "bot": bot_response,
            "timestamp": time.time(),
            "response_id": response_id
        })
        self.last_activity = time.time()
        self.last_response_id = response_id
        return True
    
    def get_context(self, n=5):
        """Get last n conversation turns for context"""
        return self.history[-n:] if len(self.history) > 0 else []
    
    def is_expired(self, timeout_seconds=600):
        return (time.time() - self.last_activity) > timeout_seconds


sessions: Dict[str, ConversationSession] = {}


def cleanup_old_sessions():
    expired = [sid for sid, session in sessions.items() if session.is_expired()]
    for sid in expired:
        print(f"[CLEANUP] Removing expired session: {sid}")
        del sessions[sid]


def get_real_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def is_localhost_request(client_ip: str) -> bool:
    return client_ip in ["127.0.0.1", "localhost", "::1", "0.0.0.0"]


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 22050) -> bytes:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return wav_buffer.getvalue()


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.post("/vad_check")
async def vad_check(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(None)
):
    """
    SEPARATE ENDPOINT: Only checks for speech presence
    Returns: {"has_speech": true/false}
    Used by frontend for silence detection ONLY
    """
    audio_bytes = await file.read()
    
    if len(audio_bytes) < 1000:
        return JSONResponse({"has_speech": False})
    
    tmp_webm = None
    tmp_wav = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_webm = tmp.name
        
        tmp_wav = tmp_webm.replace(".webm", "_vad.wav")
        if not convert_webm_to_wav(tmp_webm, tmp_wav):
            return JSONResponse({"has_speech": False})
        
        has_speech = vad_service.has_speech(tmp_wav, min_speech_duration=0.3)
        return JSONResponse({"has_speech": has_speech})
    
    except Exception as e:
        print(f"[VAD_CHECK ERROR] {e}")
        return JSONResponse({"has_speech": False})
    
    finally:
        for f in [tmp_webm, tmp_wav]:
            if f and os.path.exists(f):
                try: os.unlink(f)
                except: pass


@app.post("/process_voice")
async def process_voice(
    request: Request, 
    file: UploadFile = File(...), 
    grok_api_key: str = Form(None),
    session_id: str = Form(None)
):
    """
    MAIN PROCESSING: Only called when user finishes speaking
    Does: STT → LLM → TTS (exactly once per user turn)
    """
    print(f"\n{'='*70}")
    print(f"[PROCESS_VOICE] New request - Session: {session_id}")
    
    cleanup_old_sessions()
    
    # API Key Logic
    client_ip = get_real_client_ip(request)
    is_local = is_localhost_request(client_ip)
    
    final_api_key = None
    if grok_api_key and len(grok_api_key.strip()) > 10:
        final_api_key = grok_api_key.strip()
    elif is_local:
        env_key = os.getenv("GROQ_API_KEY")
        if env_key:
            final_api_key = env_key.strip()
        else:
            raise HTTPException(status_code=500, detail="Backend GROQ_API_KEY not configured")
    else:
        raise HTTPException(status_code=400, detail="Valid Groq API key required")
    
    # Get or create session
    session = sessions.get(session_id)
    if not session and session_id:
        print(f"[SESSION] Creating new session: {session_id}")
        session = ConversationSession(session_id)
        sessions[session_id] = session
    
    # Prevent race conditions
    if session and session.processing_lock:
        print("[SESSION] Already processing, ignoring duplicate request")
        return JSONResponse({
            "transcript": "",
            "response_text": "",
            "audio_b64": "",
            "duplicate_request": True
        })
    
    if session:
        session.processing_lock = True
    
    audio_bytes = await file.read()
    print(f"[PROCESS_VOICE] Audio size: {len(audio_bytes)} bytes")
    
    # Minimum audio check
    if len(audio_bytes) < 5000:
        if session:
            session.processing_lock = False
        return JSONResponse({
            "transcript": "",
            "response_text": "",
            "audio_b64": "",
            "too_short": True
        })
    
    tmp_webm = None
    tmp_wav = None
    
    try:
        # Save and convert
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_webm = tmp.name
        
        tmp_wav = tmp_webm.replace(".webm", "_process.wav")
        if not convert_webm_to_wav(tmp_webm, tmp_wav):
            raise HTTPException(status_code=500, detail="Audio conversion failed")
        
        # Final VAD check
        print("[VAD] Final speech verification...")
        has_speech = vad_service.has_speech(tmp_wav, min_speech_duration=0.5)
        if not has_speech:
            print("[VAD] No speech detected in final audio")
            if session:
                session.processing_lock = False
            return JSONResponse({
                "transcript": "",
                "response_text": "",
                "audio_b64": "",
                "no_speech": True
            })
        
        # STT
        print("[STT] Transcribing audio...")
        transcript = transcribe_audio(tmp_wav)
        print(f"[STT] Result: '{transcript}'")
        
        if not transcript or len(transcript.strip()) < 2:
            if session:
                session.processing_lock = False
            return JSONResponse({
                "transcript": "",
                "response_text": "I didn't catch that. Could you please repeat?",
                "audio_b64": ""
            })
        
        # LLM Processing with conversation context
        print("[LLM] Processing query with context...")
        context = session.get_context() if session else []
        response_text = process_user_query(
            transcript,
            grok_api_key=final_api_key,
            is_local_client=is_local,
            conversation_history=context
        )
        
        if not response_text or len(response_text.strip()) < 2:
            response_text = "I'm here to help you find the perfect property in Ahmedabad. What would you like to know?"
        
        print(f"[LLM] Response: '{response_text[:100]}...'")
        
        # Generate unique response ID
        response_id = str(uuid.uuid4())
        
        # Save to session history (with duplicate prevention)
        if session:
            added = session.add_turn(transcript, response_text, response_id)
            if not added:
                print("[SESSION] Duplicate response prevented")
                session.processing_lock = False
                return JSONResponse({
                    "transcript": transcript,
                    "response_text": "",
                    "audio_b64": "",
                    "duplicate_response": True
                })
        
        # TTS
        print("[TTS] Generating speech...")
        pcm_audio = speak_text(response_text)
        
        if not pcm_audio or len(pcm_audio) < 50:
            if session:
                session.processing_lock = False
            return JSONResponse({
                "transcript": transcript,
                "response_text": response_text,
                "audio_b64": "",
                "tts_failed": True
            })
        
        wav_audio = pcm_to_wav_bytes(pcm_audio)
        audio_b64 = base64.b64encode(wav_audio).decode("utf-8")
        
        print(f"[SUCCESS] Complete response generated - {len(wav_audio)} bytes")
        print(f"{'='*70}\n")
        
        if session:
            session.processing_lock = False
        
        return {
            "transcript": transcript,
            "response_text": response_text,
            "audio_b64": audio_b64,
            "audio_mime": "audio/wav",
            "session_active": True,
            "response_id": response_id
        }
    
    except HTTPException:
        if session:
            session.processing_lock = False
        raise
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        if session:
            session.processing_lock = False
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        for f in [tmp_webm, tmp_wav]:
            if f and os.path.exists(f):
                try: os.unlink(f)
                except: pass


@app.post("/synthesize_tts")
async def synthesize_tts(text: str = Form(...)):
    """Generate TTS for welcome message"""
    try:
        pcm_audio = speak_text(text)
        wav_audio = pcm_to_wav_bytes(pcm_audio)
        audio_b64 = base64.b64encode(wav_audio).decode("utf-8")
        return {"audio_b64": audio_b64, "audio_mime": "audio/wav"}
    except Exception as e:
        return {"audio_b64": "", "audio_mime": "audio/wav", "error": str(e)}


@app.post("/end_session")
async def end_session(session_id: str = Form(...)):
    """End conversation with personalized goodbye"""
    print(f"\n[END_SESSION] Closing session: {session_id}")
    
    session = sessions.get(session_id)
    
    if session:
        num_queries = len(session.history)
        duration = int(time.time() - session.created_at)
        
        if num_queries == 0:
            goodbye = "Thank you for connecting! Feel free to call back anytime you need property assistance. Have a wonderful day!"
        elif num_queries == 1:
            goodbye = "Thank you for your question! I'm always here to help with your real estate needs. Have a great day!"
        else:
            goodbye = f"It's been a pleasure assisting you today! We covered {num_queries} topics together. Don't hesitate to call back anytime. Take care!"
        
        print(f"[END_SESSION] Duration: {duration}s, Exchanges: {num_queries}")
        
        try:
            pcm_audio = speak_text(goodbye)
            wav_audio = pcm_to_wav_bytes(pcm_audio)
            audio_b64 = base64.b64encode(wav_audio).decode("utf-8")
        except Exception as e:
            print(f"[ERROR] TTS failed for goodbye: {e}")
            audio_b64 = ""
        
        del sessions[session_id]
        print(f"[END_SESSION] Session {session_id} deleted")
        
        return {
            "message": goodbye,
            "audio_b64": audio_b64,
            "audio_mime": "audio/wav",
            "total_queries": num_queries,
            "duration_seconds": duration
        }
    
    goodbye = "Thank you for calling! Have a great day!"
    return {
        "message": goodbye,
        "audio_b64": "",
        "audio_mime": "audio/wav"
    }


@app.get("/ping")
async def ping():
    """Health check"""
    return {
        "status": "ok",
        "message": "Real Estate Voice Bot - Natural Conversations",
        "active_sessions": len(sessions),
        "backend_key_configured": bool(os.getenv("GROQ_API_KEY")),
        "vad_enabled": True,
        "audio_converter": "pydub",
        "features": ["session_management", "duplicate_prevention", "context_aware"]
    }


@app.get("/sessions")
async def list_sessions():
    """Debug endpoint for active sessions"""
    return {
        "active_sessions": len(sessions),
        "sessions": [
            {
                "id": sid,
                "exchanges": len(session.history),
                "age_seconds": int(time.time() - session.created_at),
                "last_activity": int(time.time() - session.last_activity),
                "processing": session.processing_lock
            }
            for sid, session in sessions.items()
        ]
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse({"message": "No favicon"}, status_code=404)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏠 REAL ESTATE VOICE BOT - NATURAL PHONE CONVERSATIONS")
    print("="*70)
    
    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        print(f"✅ Backend Groq API key loaded: {env_key[:10]}...")
    else:
        print("⚠️  No GROQ_API_KEY in .env - users must provide keys")
    
    print("📞 Natural conversation flow enabled")
    print("🎙️  Separate VAD endpoint for silence detection")
    print("🔒 Duplicate response prevention active")
    print("💬 Context-aware responses (last 5 turns)")
    print("🎵 Audio converter: pydub")
    print("="*70 + "\n")
    
    port = int(os.environ.get("PORT", "7860"))
    print(f"[INFO] Starting server on port {port}")
    print(f"[INFO] Access at: http://localhost:{port}\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )