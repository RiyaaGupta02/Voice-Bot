"""
Voice Bot Demo - Natural Phone Call Experience
Architecture: Browser → FastAPI → STT → RAG → TTS
Features: Continuous conversation, auto-silence detection, no manual recording controls
"""

import os
import wave
import struct
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json

# Your existing imports
from Intent_Router_Thinkingprocess import process_user_query

app = FastAPI(title="Voice Bot - Phone Experience")

# CORS for ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

# Voice Activity Detection parameters
SILENCE_THRESHOLD = 500  # Audio amplitude threshold
SILENCE_DURATION = 3.0   # Seconds of silence before processing
SAMPLE_RATE = 16000

# Natural Phone Experience Interface
HTML_PHONE_INTERFACE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 AI Voice Bot - Real Estate Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .phone-container {
            background: white;
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 100%;
            padding: 40px;
            text-align: center;
        }
        
        h1 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2em;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .call-status {
            padding: 20px;
            border-radius: 15px;
            margin: 25px 0;
            font-weight: 600;
            font-size: 1.1em;
            transition: all 0.3s ease;
        }
        
        .status-idle {
            background: #f0f0f0;
            color: #666;
        }
        
        .status-connecting {
            background: #fff3cd;
            color: #856404;
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        .status-active {
            background: #d4edda;
            color: #155724;
        }
        
        .status-listening {
            background: #d1ecf1;
            color: #0c5460;
            animation: pulse 1s ease-in-out infinite;
        }
        
        .status-processing {
            background: #fff3cd;
            color: #856404;
        }
        
        .status-speaking {
            background: #cce5ff;
            color: #004085;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .call-button {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: none;
            font-size: 3em;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .call-button:hover:not(:disabled) {
            transform: scale(1.1);
        }
        
        .call-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .start-call {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
        }
        
        .end-call {
            background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
            color: white;
        }
        
        .transcript-box {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin: 25px 0;
            max-height: 300px;
            overflow-y: auto;
            text-align: left;
        }
        
        .message {
            margin: 15px 0;
            padding: 12px;
            border-radius: 10px;
            line-height: 1.5;
        }
        
        .user-message {
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
        }
        
        .bot-message {
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        
        .message-label {
            font-weight: bold;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        
        .user-label { color: #2196F3; }
        .bot-label { color: #9c27b0; }
        
        .info-box {
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            font-size: 0.95em;
            text-align: left;
        }
        
        .info-box ul {
            margin: 10px 0 0 20px;
        }
        
        .info-box li {
            margin: 5px 0;
        }
        
        .audio-visualizer {
            height: 60px;
            background: #f0f0f0;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
            align-items: center;
            justify-content: center;
            gap: 3px;
            padding: 10px;
        }
        
        .audio-visualizer.active {
            display: flex;
        }
        
        .visualizer-bar {
            width: 4px;
            background: #667eea;
            border-radius: 2px;
            transition: height 0.1s ease;
        }
    </style>
</head>
<body>
    <div class="phone-container">
        <h1>🤖 AI Voice Bot</h1>
        <div class="subtitle">Real Estate Assistant</div>
        
        <div id="callStatus" class="call-status status-idle">
            📵 Ready to Start Call
        </div>
        
        <div>
            <button id="startCallBtn" class="call-button start-call" title="Start Call">
                📞
            </button>
            <button id="endCallBtn" class="call-button end-call" disabled title="End Call">
                📴
            </button>
        </div>
        
        <div class="audio-visualizer" id="visualizer">
            <div class="visualizer-bar"></div>
            <div class="visualizer-bar"></div>
            <div class="visualizer-bar"></div>
            <div class="visualizer-bar"></div>
            <div class="visualizer-bar"></div>
        </div>
        
        <div class="transcript-box" id="transcript">
            <div style="text-align: center; color: #999;">
                Conversation will appear here...
            </div>
        </div>
        
        <div class="info-box">
            <strong>📞 How to use:</strong>
            <ul>
                <li>Click <strong>Start Call</strong> to connect</li>
                <li>Speak naturally when you hear the greeting</li>
                <li>Bot detects silence and responds automatically</li>
                <li>Click <strong>End Call</strong> when done</li>
            </ul>
        </div>
    </div>

    <script>
         let ws = null;
let mediaRecorder = null;
let audioContext = null;
let isCallActive = false;
let audioChunks = [];

const startCallBtn = document.getElementById('startCallBtn');
const endCallBtn = document.getElementById('endCallBtn');
const callStatus = document.getElementById('callStatus');
const transcript = document.getElementById('transcript');
const visualizer = document.getElementById('visualizer');

function updateStatus(message, className) {
    callStatus.textContent = message;
    callStatus.className = 'call-status ' + className;
    console.log(`[${new Date().toLocaleTimeString()}] ${message}`);
}

function addMessage(text, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
    
    const label = document.createElement('div');
    label.className = 'message-label ' + (isUser ? 'user-label' : 'bot-label');
    label.textContent = isUser ? '👤 You:' : '🤖 Bot:';
    
    const content = document.createElement('div');
    content.textContent = text;
    
    messageDiv.appendChild(label);
    messageDiv.appendChild(content);
    transcript.appendChild(messageDiv);
    
    // Auto scroll to bottom
    transcript.scrollTop = transcript.scrollHeight;
}

function clearTranscript() {
    transcript.innerHTML = '<div style="text-align: center; color: #999;">Conversation will appear here...</div>';
}

async function startCall() {
    try {
        updateStatus('🔄 Connecting...', 'status-connecting');
        startCallBtn.disabled = true;
        
        // Request microphone
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        });
        
        console.log('🎤 Microphone access granted');
        
        // WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/call`;
        console.log('🔌 Connecting to:', wsUrl);
        
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('✅ WebSocket connected, readyState:', ws.readyState);
            updateStatus('📞 Call Connected - Listening...', 'status-active');
            isCallActive = true;
            endCallBtn.disabled = false;
            clearTranscript();
            visualizer.classList.add('active');
            
            // Start recording
            startRecording(stream);
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleServerMessage(data);
        };
        
        ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            updateStatus('❌ Connection Error', 'status-idle');
            cleanup();
        };
        
        ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            if (isCallActive) {
                updateStatus('📵 Call Ended', 'status-idle');
                cleanup();
            }
        };
        
    } catch (error) {
        console.error('❌ Error starting call:', error);
        alert('Please allow microphone access to make calls.');
        updateStatus('📵 Ready to Start Call', 'status-idle');
        startCallBtn.disabled = false;
    }
}

function startRecording(stream) {
    // Use better codec settings
    const options = { 
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 16000
    };
    
    // Fallback for Safari
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        console.log('⚠️ Opus not supported, using default codec');
        options.mimeType = 'audio/webm';
    }
    
    mediaRecorder = new MediaRecorder(stream, options);
    audioChunks = [];
    
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
            console.log(`📦 Chunk: ${event.data.size} bytes`);
        }
    };
    
    mediaRecorder.onstop = () => {
        console.log(`🎬 Recording stopped. Chunks: ${audioChunks.length}`);
        
        if (audioChunks.length > 0 && isCallActive) {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            console.log(`📦 Total blob: ${audioBlob.size} bytes`);
            
            if (audioBlob.size > 10000) { // Only send if > 10KB
                sendAudioToServer(audioBlob);
            } else {
                console.log('⚠️ Audio too small, skipping');
                restartRecording();
            }
            audioChunks = [];
        }
    };
    
    mediaRecorder.onerror = (event) => {
        console.error('❌ MediaRecorder error:', event.error);
    };
    
    // Start recording
    mediaRecorder.start();
    console.log('🎙️ Recording started');
    
    // Setup silence detection
    setupSilenceDetection(stream);
}

function restartRecording() {
    if (isCallActive && mediaRecorder) {
        audioChunks = [];
        if (mediaRecorder.state === 'inactive') {
            mediaRecorder.start();
            console.log('🔄 Recording restarted');
        }
    }
}

function setupSilenceDetection(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    const analyzer = audioContext.createAnalyser();
    analyzer.fftSize = 2048;
    source.connect(analyzer);
    
    const bufferLength = analyzer.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    let silenceStart = null;
    let speechStart = null;
    let isSpeaking = false;
    const SILENCE_THRESHOLD = 25;
    const SILENCE_DURATION = 2000; // 2 seconds
    const MIN_SPEECH_DURATION = 500; // 0.5 seconds minimum
    
    function detectSilence() {
        if (!isCallActive) return;
        
        analyzer.getByteTimeDomainData(dataArray);
        
        // Calculate audio level
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
            sum += Math.abs(dataArray[i] - 128);
        }
        const average = sum / bufferLength;
        
        // Update visualizer
        updateVisualizer(average);
        
        // Detect speech vs silence
        if (average > SILENCE_THRESHOLD) {
            // Speech detected
            if (!isSpeaking) {
                isSpeaking = true;
                speechStart = Date.now();
                updateStatus('🎤 Listening to you...', 'status-listening');
            }
            silenceStart = null;
        } else {
            // Silence detected
            if (isSpeaking) {
                if (!silenceStart) {
                    silenceStart = Date.now();
                } else {
                    const silenceDuration = Date.now() - silenceStart;
                    const speechDuration = speechStart ? Date.now() - speechStart : 0;
                    
                    if (silenceDuration > SILENCE_DURATION && speechDuration > MIN_SPEECH_DURATION) {
                        // Silence long enough - process recording
                        isSpeaking = false;
                        silenceStart = null;
                        speechStart = null;
                        processCurrentRecording();
                    }
                }
            }
        }
        
        requestAnimationFrame(detectSilence);
    }
    
    detectSilence();
}

function updateVisualizer(level) {
    const bars = visualizer.querySelectorAll('.visualizer-bar');
    bars.forEach((bar, index) => {
        const height = Math.min(40, level * (index + 1) * 0.5);
        bar.style.height = height + 'px';
    });
}

function processCurrentRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        console.log('⏹️ Stopping recording for processing');
        mediaRecorder.stop();
        updateStatus('⏳ Processing...', 'status-processing');
        
        // Restart recording after processing starts
        setTimeout(() => {
            restartRecording();
        }, 500); // Increased delay
    }
}

async function sendAudioToServer(audioBlob) {
    try {
        console.log(`📤 Sending ${audioBlob.size} bytes to server`);
        
        const reader = new FileReader();
        reader.onload = () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(reader.result);
                console.log('✅ Audio sent successfully');
            } else {
                console.error('❌ WebSocket not open, readyState:', ws ? ws.readyState : 'null');
            }
        };
        reader.onerror = (error) => {
            console.error('❌ FileReader error:', error);
        };
        reader.readAsArrayBuffer(audioBlob);
    } catch (error) {
        console.error('❌ Error sending audio:', error);
    }
}

function handleServerMessage(data) {
    console.log('📨 Server message:', data.type);
    
    try {
        if (data.type === 'greeting') {
            addMessage(data.text, false);
            playAudio(data.audio_url);
        } else if (data.type === 'transcript') {
            addMessage(data.text, true);
        } else if (data.type === 'response') {
            addMessage(data.text, false);
            playAudio(data.audio_url);
        } else if (data.type === 'goodbye') {
            addMessage(data.text, false);
            playAudio(data.audio_url);
            setTimeout(() => endCall(), 3000);
        } else if (data.type === 'status') {
            updateStatus(data.message, data.className);
        } else if (data.type === 'pong') {
            console.log('🏓 Pong received');
        }
    } catch (error) {
        console.error('❌ Error handling message:', error);
    }
}

function playAudio(audioUrl) {
    updateStatus('🔊 Bot is speaking...', 'status-speaking');
    const audio = new Audio(audioUrl);
    
    audio.onerror = (error) => {
        console.error('❌ Audio playback error:', error);
        updateStatus('⚠️ Audio error', 'status-error');
        if (isCallActive) {
            setTimeout(() => {
                updateStatus('🎤 Listening...', 'status-listening');
            }, 1000);
        }
    };
    
    audio.onended = () => {
        if (isCallActive) {
            updateStatus('🎤 Listening...', 'status-listening');
        }
    };
    
    audio.play().catch(error => {
        console.error('❌ Audio play failed:', error);
    });
}

function endCall() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        console.log('📴 Sending end call signal');
        ws.send(JSON.stringify({ type: 'end_call' }));
    }
    cleanup();
}

function cleanup() {
    console.log('🧹 Cleaning up...');
    isCallActive = false;
    
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    
    if (ws) {
        ws.close();
        ws = null;
    }
    
    visualizer.classList.remove('active');
    startCallBtn.disabled = false;
    endCallBtn.disabled = true;
    updateStatus('📵 Ready to Start Call', 'status-idle');
}

startCallBtn.addEventListener('click', startCall);
endCallBtn.addEventListener('click', endCall);

window.addEventListener('beforeunload', cleanup);

// Ping server every 30 seconds
setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN && isCallActive) {
        ws.send(JSON.stringify({ type: 'ping' }));
        console.log('🏓 Ping sent');
    }
}, 30000);

// Check connection every 5 seconds
setInterval(() => {
    if (ws && isCallActive) {
        if (ws.readyState !== WebSocket.OPEN) {
            console.error('❌ Connection lost, readyState:', ws.readyState);
            updateStatus('⚠️ Connection lost - please refresh', 'status-error');
            cleanup();
        }
    }
}, 5000);

    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve phone interface"""
    return HTML_PHONE_INTERFACE


@app.websocket("/ws/call")
async def websocket_call(websocket: WebSocket):
    """Handle voice call via WebSocket"""
    await websocket.accept()
    
    conversation_count = 0
    session_id = int(asyncio.get_event_loop().time())  # Unique session ID
    
    try:
        print("\n" + "="*70)
        print(f"📞 NEW CALL CONNECTED - Session {session_id}")
        print("="*70)
        
        # Send greeting
        greeting = "Hello! I'm your real estate assistant. How can I help you today?"
        greeting_audio = await generate_tts_async(greeting, session_id, 0)
        
        if greeting_audio:
            await websocket.send_json({
                "type": "greeting",
                "text": greeting,
                "audio_url": f"/audio/{greeting_audio}"
            })
            print(f"🤖 Bot: {greeting}")
        
        # Main conversation loop
        while True:
            try:
                # Wait for data with timeout
                data = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=300.0
                )
                
                # Handle text messages (ping/end_call)
                if 'text' in data:
                    try:
                        message = json.loads(data['text'])
                        
                        # Handle ping
                        if message.get('type') == 'ping':
                            await websocket.send_json({"type": "pong"})
                            print("🏓 Pong sent")
                            continue
                        
                        # Handle end call
                        if message.get('type') == 'end_call':
                            goodbye = "Thank you for calling! Have a great day. Goodbye!"
                            goodbye_audio = await generate_tts_async(goodbye, session_id, 999)
                            
                            if goodbye_audio:
                                await websocket.send_json({
                                    "type": "goodbye",
                                    "text": goodbye,
                                    "audio_url": f"/audio/{goodbye_audio}"
                                })
                            
                            print(f"🤖 Bot: {goodbye}")
                            await asyncio.sleep(1)  # Give time for audio to send
                            break
                    except json.JSONDecodeError:
                        print("⚠️ Invalid JSON in text message")
                        continue
                
                # Handle audio bytes
                if 'bytes' not in data:
                    continue
                
                audio_bytes = data['bytes']
                print(f"📥 Received audio: {len(audio_bytes):,} bytes")
                
                # Size validation
                if len(audio_bytes) < 10000:
                    print(f"⚠️ Audio too small ({len(audio_bytes)} bytes), skipping")
                    continue
                
                conversation_count += 1
                
                print("\n" + "─"*50)
                print(f"🎤 Turn #{conversation_count}")
                print("─"*50)
                
                # Save audio
                audio_path = TEMP_DIR / f"session_{session_id}_turn_{conversation_count}.webm"
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
                print(f"💾 Saved: {audio_path}")
                
                # Convert to WAV
                print("🔄 Converting to WAV...")
                wav_path = await convert_to_wav(audio_path)
                
                if not wav_path or not wav_path.exists():
                    print("❌ Conversion failed")
                    await websocket.send_json({
                        "type": "status",
                        "message": "⚠️ Audio processing error",
                        "className": "status-error"
                    })
                    continue
                
                # Check duration
                duration = get_audio_duration(wav_path)
                print(f"⏱️ Duration: {duration:.2f}s")
                
                if duration < 0.5:
                    print("⚠️ Audio too short")
                    continue
                
                # Transcribe
                print("📝 Transcribing...")
                user_text = await transcribe_audio_async(wav_path)
                
                if not user_text or len(user_text.strip()) < 3:
                    print("⚠️ No clear speech detected")
                    await websocket.send_json({
                        "type": "status",
                        "message": "🎤 Didn't catch that, please speak again",
                        "className": "status-listening"
                    })
                    continue
                
                print(f"👤 User: \"{user_text}\"")
                
                # Send transcript to frontend
                await websocket.send_json({
                    "type": "transcript",
                    "text": user_text
                })
                
                # Check for goodbye
                if is_goodbye_intent(user_text):
                    goodbye = "Thank you for calling! Have a great day. Goodbye!"
                    goodbye_audio = await generate_tts_async(goodbye, session_id, conversation_count)
                    
                    if goodbye_audio:
                        await websocket.send_json({
                            "type": "goodbye",
                            "text": goodbye,
                            "audio_url": f"/audio/{goodbye_audio}"
                        })
                    
                    print(f"🤖 Bot: {goodbye}")
                    await asyncio.sleep(2)
                    break
                
                # Process query
                print("🤔 Processing query...")
                response = await process_query_async(user_text)
                
                if not response:
                    response = "I'm sorry, I couldn't process that. Could you please rephrase?"
                
                print(f"🤖 Bot: {response[:100]}...")
                
                # Generate TTS
                print("🔊 Generating speech...")
                audio_file = await generate_tts_async(response, session_id, conversation_count)
                
                if not audio_file:
                    print("❌ TTS generation failed")
                    continue
                
                # Send response
                await websocket.send_json({
                    "type": "response",
                    "text": response,
                    "audio_url": f"/audio/{audio_file}"
                })
                
                print("✅ Response sent")
                
                # Cleanup old files
                cleanup_old_audio_files()
                
            except asyncio.TimeoutError:
                print("⏱️ Timeout - no activity")
                break
            except WebSocketDisconnect:
                print("📴 Client disconnected")
                break
            except Exception as e:
                print(f"❌ Loop error: {e}")
                import traceback
                traceback.print_exc()
                
                # Try to inform client
                try:
                    await websocket.send_json({
                        "type": "status",
                        "message": "⚠️ Processing error, please try again",
                        "className": "status-error"
                    })
                except:
                    pass
                
                continue
    
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n📊 Call ended - Total turns: {conversation_count}")
        print("="*70 + "\n")
        
        # Cleanup session files
        try:
            for file in TEMP_DIR.glob(f"session_{session_id}_*"):
                file.unlink()
        except:
            pass


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve audio files"""
    file_path = TEMP_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    return {"error": "File not found"}



# helper functions ------------------------------------------------------------

async def generate_tts_async(text: str, session_id: int, turn: int) -> str:
    """Generate TTS with unique filename"""
    try:
        from TTS_Service import speak_text_to_file
        
        # Create unique filename
        import time
        timestamp = int(time.time() * 1000)
        filename = f"session_{session_id}_turn_{turn}_{timestamp}.wav"
        output_path = TEMP_DIR / filename
        
        print(f"📝 TTS: '{text[:50]}...'")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, speak_text_to_file, text, str(output_path))
        
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"✅ TTS generated: {output_path.stat().st_size:,} bytes")
            return filename
        else:
            print("❌ TTS file empty or missing")
            return None
            
    except Exception as e:
        print(f"❌ TTS error: {e}")
        import traceback
        traceback.print_exc()
        return None

async def transcribe_audio_async(audio_path: Path) -> str:
    """Transcribe audio using STT or Groq Whisper"""
    try:
        try:
            from STT_Services import preprocess_audio_file, transcribe_audio_file
            cleaned = preprocess_audio_file(str(audio_path))
            text = transcribe_audio_file(cleaned)
            if text:
                return text
        except:
            pass

        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(audio_path.name, f.read()),
                model="whisper-large-v3-turbo",
            )

        return transcription.text.strip()

    except Exception as e:
        print(f"❌ STT error: {e}")
        return None


async def process_query_async(user_text: str) -> str:
    """Process user query through RAG"""
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, process_user_query, user_text)
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        return None


async def generate_tts_async(text: str, session_id: int, turn: int) -> str:
    """Generate TTS with unique filename"""
    try:
        from TTS_Service import speak_text_to_file
        
        # Create unique filename
        import time
        timestamp = int(time.time() * 1000)
        filename = f"session_{session_id}_turn_{turn}_{timestamp}.wav"
        output_path = TEMP_DIR / filename
        
        print(f"📝 TTS: '{text[:50]}...'")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, speak_text_to_file, text, str(output_path))
        
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"✅ TTS generated: {output_path.stat().st_size:,} bytes")
            return filename
        else:
            print("❌ TTS file empty or missing")
            return None
            
    except Exception as e:
        print(f"❌ TTS error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
async def convert_to_wav(webm_path: Path) -> Path:
    """Enhanced audio conversion with better error handling"""
    wav_path = webm_path.with_suffix('.wav')
    
    try:
        import subprocess
        
        # Check if ffmpeg exists
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ ffmpeg not installed! Install: https://ffmpeg.org/download.html")
            return None
        
        # Convert with optimized settings
        result = subprocess.run([
            'ffmpeg',
            '-i', str(webm_path),
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',       # Mono
            '-acodec', 'pcm_s16le',
            '-af', 'loudnorm',  # Normalize audio
            '-y',             # Overwrite
            str(wav_path)
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode != 0:
            print(f"❌ ffmpeg error: {result.stderr}")
            return None
        
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            print(f"✅ Converted: {wav_path.stat().st_size:,} bytes")
            return wav_path
        else:
            print("❌ Conversion produced invalid file")
            return None
    
    except subprocess.TimeoutExpired:
        print("❌ Conversion timeout")
        return None
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return None
    

def is_goodbye_intent(text: str) -> bool:
    """Detect goodbye intent"""
    if not text:
        return False

    text_lower = text.lower()
    goodbye_phrases = [
        'goodbye','bye','bye bye','good bye','thank you bye','thanks bye',
        'ok bye','end call','hang up','disconnect','that\'s all',
        'nothing else','no thanks','i\'m done','that\'s it',
        'okay thanks','ok thanks'
    ]

    return any(phrase in text_lower for phrase in goodbye_phrases)


def cleanup_old_audio_files():
    """Remove old audio files"""
    try:
        import time
        now = time.time()

        for file in TEMP_DIR.glob("*.wav"):
            if file.stat().st_mtime < now - 300:
                file.unlink()

        for file in TEMP_DIR.glob("*.webm"):
            if file.stat().st_mtime < now - 300:
                file.unlink()

    except:
        pass


def get_audio_duration(wav_path: Path) -> float:
    """Get audio duration in seconds"""
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception as e:
        print(f"❌ Duration check failed: {e}")
        return 0.0



@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "service": "voice-bot-phone-experience",
        "features": [
            "continuous-conversation",
            "auto-silence-detection",
            "natural-phone-experience"
        ]
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 VOICE BOT - NATURAL PHONE EXPERIENCE")
    print("="*70)
    print("\n✨ Features:")
    print("   ✅ Continuous conversation (no manual recording)")
    print("   ✅ Auto silence detection (3 seconds)")
    print("   ✅ Natural phone call experience")
    print("   ✅ Just 2 buttons: Start Call / End Call")
    print("\n📍 Local: http://localhost:8000")
    print("🌐 Use ngrok: ngrok http 8000")
    print("\n💡 Setup:")
    print("   1. Set GROQ_API_KEY environment variable")
    print("   2. Install: pip install fastapi uvicorn websockets groq")
    print("   3. Optional: Install ffmpeg for better audio conversion")
    print("\n🛑 Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)