# FreeSWITCH Voice Bot - Production Ready
# Supports BOTH WebRTC (browser) AND SIP (Linphone/Zoiper) simultaneously
# Architecture: WebRTC/SIP → FreeSWITCH → Voice Bot (STT → RAG → TTS)

import socket
import os
import wave
import struct
import threading
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from scipy import signal
from TTS_Service import PiperTTSAsync
import asyncio

# Import your existing functions
from Intent_Router_Thinkingprocess import process_user_query

# Setup paths
PROJECT_DIR = Path(__file__).parent
TEMP_DIR = PROJECT_DIR / "temp_audio"
TEMP_DIR.mkdir(exist_ok=True)
PIPER_MODEL_PATH = "en_US-libritts-high.onnx"  # or your preferred Piper model

# FastAPI app
app = FastAPI(
    title="FreeSWITCH Voice Bot - Production",
    description="AI Voice Bot with WebRTC and SIP support",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebRTC Client HTML (Updated for Verto protocol)
WEBRTC_CLIENT_HTML = """
<!DOCTYPE html>
<html lang="en">
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
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
            padding: 40px;
        }
        
        h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .status {
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 25px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .status.disconnected {
            background: #fee;
            color: #c33;
        }
        
        .status.connecting {
            background: #ffeaa7;
            color: #d63031;
        }
        
        .status.connected {
            background: #dfe;
            color: #2d3;
        }
        
        .status.error {
            background: #fee;
            color: #c33;
            font-size: 0.9em;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
        }
        
        button {
            flex: 1;
            padding: 15px 25px;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        #startBtn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        #startBtn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        #hangupBtn {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        #hangupBtn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(245, 87, 108, 0.4);
        }
        
        .info-box {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .info-box h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .info-box ul {
            list-style: none;
            padding-left: 0;
        }
        
        .info-box li {
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
            color: #555;
            line-height: 1.6;
        }
        
        .info-box li:before {
            content: "✓";
            position: absolute;
            left: 0;
            color: #667eea;
            font-weight: bold;
        }
        
        .connection-info {
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            border-radius: 8px;
            font-size: 0.9em;
            color: #555;
        }
        
        .connection-info strong {
            color: #3498db;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .connecting .status {
            animation: pulse 1.5s ease-in-out infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Voice Bot</h1>
        <div class="subtitle">Real Estate Assistant</div>
        
        <div id="status" class="status disconnected">
            🔴 Not Connected
        </div>
        
        <div class="controls">
            <button id="startBtn">📞 Start Call</button>
            <button id="hangupBtn" disabled>❌ Hang Up</button>
        </div>
        
        <div class="info-box">
            <h3>How it works:</h3>
            <ul>
                <li>Click "Start Call" to connect via WebRTC</li>
                <li>Allow microphone access when prompted</li>
                <li>Speak naturally to the AI assistant</li>
                <li>Get real-time responses about real estate</li>
            </ul>
        </div>
        
        <div class="connection-info">
            <strong>Connection:</strong> WebRTC via FreeSWITCH Verto<br>
            <strong>Extension:</strong> 5000 (routes to AI bot)
        </div>
    </div>
 <!-- jQuery (Verto requires 1.x) -->
<script src="https://code.jquery.com/jquery-1.11.3.min.js"></script>

<!-- JSON-RPC client -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-jsonrpc/0.0.1/jquery.jsonrpc.min.js"></script>

<!-- Official FreeSWITCH Verto JS -->
<script src="https://freeswitch.org/stash/projects/FS/repos/freeswitch/raw/html5/verto/verto-min.js"></script>


<script>
    // Sanity checks for loading
    if (typeof jQuery === 'undefined') alert("❌ jQuery failed to load!");
    if (typeof $.verto === 'undefined') alert("❌ Verto library failed to load!");

    const statusEl = document.getElementById('status');
    const startBtn = document.getElementById('startBtn');
    const hangupBtn = document.getElementById('hangupBtn');

    let vertoHandle = null;
    let currentCall = null;

    // FreeSWITCH Verto configuration
    const vertoConfig = {
        login: "1008@192.168.1.4",
        passwd: "1234",
        socketUrl: "ws://192.168.1.4:8082",  // ✅ Change from wss to ws
        tag: "webclient",
        ringFile: "sounds/bell_ring.wav",
        videoParams: {},
        audioParams: {
            googAutoGainControl: true,
            googNoiseSuppression: true,
            googHighpassFilter: true
        },
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
        ringSleep: 6000
    };

    function updateStatus(message, type) {
        statusEl.textContent = message;
        statusEl.className = `status ${type}`;
    }

    function initVerto() {
        try {
            vertoHandle = new $.verto(vertoConfig, {

                onWSLogin: (success) => {
                    if (success) {
                        console.log("Verto login successful");
                        updateStatus("🔵 Connected - Ready to call", "connected");
                        makeCall();
                    } else {
                        console.error("Verto login failed");
                        updateStatus("❌ Login failed", "error");
                        cleanup();
                    }
                },

                onWSClose: () => {
                    console.log("WebSocket closed");
                    updateStatus("🔴 Connection closed", "disconnected");
                    cleanup();
                },

                onDialogState: (dlgState) => {
                    console.log("Dialog state:", dlgState);

                    if (dlgState.state === $.verto.enum.state.active) {
                        updateStatus("🟢 Call Active - Speak now!", "connected");
                        hangupBtn.disabled = false;

                    } else if (dlgState.state === $.verto.enum.state.destroy) {
                        updateStatus("🔴 Call Ended", "disconnected");
                        cleanup();
                    }
                }
            });

        } catch (error) {
            console.error("Verto initialization error:", error);
            updateStatus("❌ Verto init failed: " + error.message, "error");
            cleanup();
        }
    }

    function makeCall() {
        try {
            currentCall = vertoHandle.newCall({
                destination_number: "5000",
                caller_id_name: "WebRTC User",
                caller_id_number: "webrtc_user",
                outgoingBandwidth: "default",
                incomingBandwidth: "default",
                useVideo: false,
                useStereo: false,
                useCamera: false,
                useMic: true,
                dedEnc: false,
                mirrorInput: false
            });

            updateStatus("📞 Calling extension 5000...", "connecting");

        } catch (error) {
            console.error("Call error:", error);
            updateStatus("❌ Call failed: " + error.message, "error");
            cleanup();
        }
    }

    async function startCall() {
        try {
            updateStatus("🔄 Connecting to FreeSWITCH...", "connecting");
            startBtn.disabled = true;

            // Request mic permission
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(t => t.stop());

            initVerto();

        } catch (error) {
            console.error("Error starting call:", error);
            updateStatus("❌ Microphone access denied or error", "error");
            alert("Please allow microphone access to make calls.");
            cleanup();
        }
    }

    function hangup() {
        try {
            if (currentCall) currentCall.hangup();
            updateStatus("🔴 Ending call...", "disconnected");

        } catch (error) {
            console.error("Hangup error:", error);
        }

        cleanup();
    }

    function cleanup() {
        if (vertoHandle) {
            try { vertoHandle.logout(); } catch {}
        }
        vertoHandle = null;
        currentCall = null;

        startBtn.disabled = false;
        hangupBtn.disabled = true;

        updateStatus("🔴 Not Connected", "disconnected");
    }

    startBtn.addEventListener("click", startCall);
    hangupBtn.addEventListener("click", hangup);
    window.addEventListener("beforeunload", cleanup);
</script>
</body>
</html>
"""


class ProductionVoiceBot:
    """
    Production-ready voice bot that works with FreeSWITCH.
    Handles BOTH WebRTC and SIP calls through the same code path.
    """
    
    def __init__(self, host='0.0.0.0', port=8084):
        self.host = host
        self.port = port
        self.sample_rate = 8000
        self.channels = 1
        
    def start_server(self):
        # Start socket server - handles BOTH WebRTC and SIP
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            server.bind((self.host, self.port))
            server.listen(10)
        except Exception as e:
            print(f"❌ Failed to bind/listen on {self.host}:{self.port}: {e}")
            try:
                server.close()
            except Exception:
                pass
            return

        print("\n" + "="*70)
        print("🏭 PRODUCTION VOICE BOT SERVER")
        print("="*70)
        print(f"✅ Socket Server: {self.host}:{self.port}")
        print(f"✅ Handles: WebRTC + SIP (both via FreeSWITCH)")
        print("\n📋 ARCHITECTURE:")
        print("   Browser (WebRTC) → FreeSWITCH (Verto) ─┐")
        print("   SIP Client       → FreeSWITCH (SIP)    ├→ Dialplan (5000)")
        print("                                           ↓")
        print("                                    Voice Bot (8084)")
        print("\n🌐 ACCESS POINTS:")
        print("   WebRTC: http://localhost:8000 (browser)")
        print("   SIP: sip:5000@your-server-ip (any SIP client)")
        print("\n✅ FreeSWITCH Verto Status:")
        print("   ws://192.168.1.4:8081 (WebSocket)")
        print("   wss://192.168.1.4:8082 (Secure WebSocket)")
        print("\n🛑 Press Ctrl+C to stop")
        print("="*70 + "\n")

        # ------------------------------------------------------------
        # 🔍 NEW: Verify socket actually listening using `ss`
        # ------------------------------------------------------------
        print(f"🔌 Socket bound to {self.host}:{self.port}")
        print(f"🔌 Testing connectivity...")

        try:
            import subprocess
            test = subprocess.run(
                ['ss', '-tlnp', f'sport = :{self.port}'],
                capture_output=True, text=True
            )
            if str(self.port) in test.stdout:
                print(f"✅ Port {self.port} is CONFIRMED LISTENING")
            else:
                print(f"⚠️ Port {self.port} may NOT be accessible from outside!")
        except Exception as e:
            print(f"⚠️ Could not verify port status: {e}")

        print("="*70 + "\n")
        print("⏳ Waiting for FreeSWITCH connections...")
        print("   (Dial extension 5000 to test)\n")

        # ------------------------------------------------------------
        # MAIN SERVER LOOP
        # ------------------------------------------------------------
        call_count = 0

        while True:
            try:
                # NEW DEBUG: show that accept() is waiting
                print("👂 accept() waiting for connection...")

                client_socket, address = server.accept()

                # NEW DEBUG: confirm connection immediately
                print(f"🎉 CONNECTION RECEIVED from {address}!")

                call_count += 1
                print(f"\n{'='*70}")
                print(f"📞 CALL #{call_count} from {address}")
                print("="*70)

                self.handle_call(client_socket, call_count)

            except KeyboardInterrupt:
                print(f"\n\n👋 Shutting down... (Total calls handled: {call_count})")
                break
            except Exception as e:
                print(f"❌ Server error: {e}")
                
            finally:
                server.close()
                print("🔌 Server socket closed.")

    
    def handle_call(self, sock, call_num):
        # Handle call from FreeSWITCH (WebRTC or SIP - same code)
        # Ensures FreeSWITCH receives connect, myevents, and answer so media flows
        # Wraps the flow in try/except/finally for robust cleanup and logging
        try:
            # Optional: set a socket timeout so we don't block forever
            try:
                sock.settimeout(60)  # adjust if needed
            except Exception:
                # not fatal if socket doesn't support timeout setting
                pass

            # Read FreeSWITCH channel data
            channel_data = self.read_channel_data(sock) or {}

            # Identify call source
            caller_id = channel_data.get('Caller-Caller-ID-Number', 'Unknown')
            channel_name = channel_data.get('Channel-Name', '')

            if 'verto' in channel_name.lower() or 'webrtc' in caller_id.lower():
                call_source = "🌐 WebRTC (Browser)"
            else:
                call_source = "📞 SIP Client"

            print(f"[Call {call_num}]    Source: {call_source}")
            print(f"[Call {call_num}]    Caller: {caller_id}")
            print(f"[Call {call_num}]    Channel: {channel_name[:50]}...")

            # -------------------------
            # CRITICAL: Establish media
            # -------------------------
            # Send connect, subscribe to events and answer the call so RTP starts.
            try:
                self.send_command(sock, "connect\n\n")
                self.send_command(sock, "myevents\n\n")
                self.send_command(sock, "answer\n\n")
            except Exception as e:
                # If send_command fails, raise to outer handler to log & cleanup
                raise RuntimeError(f"Failed to send initial ESL commands: {e}")

            # allow a tiny pause for FreeSWITCH to transition the call into media mode
            import time
            time.sleep(0.3)

            print(f"[Call {call_num}] ✅ Call answered and events subscribed!")

            # Welcome message
            welcome = (
                "Hello! Welcome to our real estate agency. "
                "I'm your AI assistant. How can I help you today?"
            )

            # Send TTS/initial prompt
            try:
                self.speak_to_caller(sock, welcome)
            except Exception as e:
                # Non-fatal: log and continue — caller may still speak
                print(f"[Call {call_num}] ⚠️ speak_to_caller failed: {e}")

            # Main conversation loop
            turn = 0
            max_turns = 50

            while turn < max_turns:
                turn += 1
                print(f"\n{'─'*50}")
                print(f"[Call {call_num}] 🎯 Turn {turn}/{max_turns}")
                print(f"{'─'*50}")

                # Listen to user (record); if no audio, prompt again
                try:
                    heard = self.record_from_caller(sock)
                except Exception as e:
                    print(f"[Call {call_num}] ⚠️ record_from_caller exception: {e}")
                    heard = False

                if not heard:
                    print(f"[Call {call_num}] ⚠️ No audio detected on turn {turn}")
                    try:
                        self.speak_to_caller(
                            sock,
                            "I didn't hear you. Please speak after the beep."
                        )
                    except Exception as e:
                        print(f"[Call {call_num}] ⚠️ speak_to_caller failed: {e}")
                    # continue to next iteration (do not increment turn further)
                    continue

                # Transcribe audio
                try:
                    user_text = self.transcribe_audio()
                except Exception as e:
                    print(f"[Call {call_num}] ⚠️ transcribe_audio exception: {e}")
                    user_text = None

                if not user_text:
                    print(f"[Call {call_num}] ⚠️ Transcription failed on turn {turn}")
                    try:
                        self.speak_to_caller(
                            sock,
                            "Sorry, I couldn't understand. Please speak clearly."
                        )
                    except Exception as e:
                        print(f"[Call {call_num}] ⚠️ speak_to_caller failed: {e}")
                    continue

                print(f"[Call {call_num}] 📝 User: \"{user_text}\"")

                # Check for goodbye
                try:
                    if self.is_goodbye(user_text):
                        print(f"[Call {call_num}] 👋 User ended call")
                        try:
                            self.speak_to_caller(
                                sock,
                                "Thank you for calling! Have a great day. Goodbye!"
                            )
                        except Exception:
                            pass
                        break
                except Exception as e:
                    print(f"[Call {call_num}] ⚠️ is_goodbye check failed: {e}")

                # Process query (RAG pipeline)
                try:
                    response = process_user_query(user_text)
                except Exception as e:
                    print(f"[Call {call_num}] ⚠️ process_user_query exception: {e}")
                    response = None

                if not response:
                    response = (
                        "I apologize, I couldn't process that. "
                        "Could you please rephrase?"
                    )

                print(f"[Call {call_num}] 💬 Bot: {response[:200]}")

                # Speak response
                try:
                    self.speak_to_caller(sock, response)
                except Exception as e:
                    print(f"[Call {call_num}] ⚠️ speak_to_caller failed: {e}")

                print(f"[Call {call_num}] ✅ Turn {turn} complete")

            # If loop exited due to reaching max turns, give a polite closing
            if turn >= max_turns:
                print(f"[Call {call_num}] ⚠️ Max turns reached")
                try:
                    self.speak_to_caller(
                        sock,
                        "We've had a long conversation. "
                        "Please call back if you need more help. Goodbye!"
                    )
                except Exception:
                    pass

        except Exception as e:
            # Top-level catch: log full stack for diagnostics
            print(f"[Call {call_num}] ❌ Call error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure we try to gracefully exit the socket/ESL connection and always close the socket
            try:
                # Try to hangup/exit the ESL session; ignore errors
                try:
                    self.send_command(sock, "hangup\n\n")
                except Exception:
                    # Not fatal; try exit anyway
                    pass
                try:
                    self.send_command(sock, "exit\n\n")
                except Exception:
                    pass
            except Exception:
                pass

            try:
                sock.close()
            except Exception:
                pass

            print(f"[Call {call_num}] 📴 Call ended")



    def read_channel_data(self, sock) -> Dict[str, str]:
        """Read FreeSWITCH channel data from socket"""
        data = {}

        # Increased timeout to allow FreeSWITCH to send channel data
        sock.settimeout(5.0)

        try:
            buffer = b""
            max_attempts = 10  # Increased attempts for robustness
            attempts = 0

            while attempts < max_attempts:
                try:
                    # Increased buffer sixe (FS can send large headers)
                    chunk = sock.recv(8192)
                    if not chunk:
                        break

                    buffer += chunk
                    attempts += 1

                    # FreeSWITCH ends header with blank line
                    if b"\n\n" in buffer:
                        break

                except socket.timeout: 
                    # If we received some data then break
                    if buffer:
                       break
                    attempts += 1   

            # Decode the full received headers
            text = buffer.decode("utf-8", errors="ignore")
            lines = text.split("\n")

            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    data[key.strip()] = value.strip()
                    
        except Exception as e:
            print(f"   ⚠️ Channel data read error: {e}")

        return data
    


    def send_command(self, sock, command: str):
        """Send command to FreeSWITCH"""
        try:
            sock.sendall(command.encode('utf-8'))
        except Exception as e:
            print(f"   ⚠️ Command send error: {e}")
    
    def record_from_caller(self, sock, max_duration: int = 5) -> bool:
        """Record audio from caller"""
        recording_path = TEMP_DIR / "recording.wav"
        
        print(f"   🎤 Recording (max {max_duration}s)...")
        audio_chunks = []
        sock.settimeout(0.2)
        
        try:
            frames_recorded = 0
            silent_frames = 0
            recording_started = False
            
            for _ in range(max_duration * 10):
                try:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    
                    audio_chunks.append(chunk)
                    frames_recorded += 1
                    
                    if len(chunk) >= 2:
                        samples = struct.unpack(f'{len(chunk)//2}h', chunk)
                        max_amp = max(abs(s) for s in samples) if samples else 0
                        
                        if max_amp > 500:
                            recording_started = True
                            silent_frames = 0
                        elif recording_started:
                            silent_frames += 1
                            if silent_frames > 15:
                                print("   ✅ Speech completed")
                                break
                
                except socket.timeout:
                    if recording_started:
                        silent_frames += 1
                        if silent_frames > 15:
                            break
                    continue
            
            if audio_chunks and frames_recorded > 5:
                with wave.open(str(recording_path), 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(audio_chunks))
                
                print(f"   💾 Saved {frames_recorded} frames")
                return True
            else:
                print("   ⚠️ Insufficient audio")
                return False
        
        except Exception as e:
            print(f"   ❌ Recording error: {e}")
            return False
    
    def transcribe_audio(self) -> str:
        """Transcribe recorded audio using STT"""
        recording_path = TEMP_DIR / "recording.wav"
        
        try:
            from STT_Services import preprocess_audio_file, transcribe_audio_file     # doing this so that in this freeswitch server stt + nlp func of stt_serice.py file work but also its nose redn & VAD trimming + silence removal work
            cleaned_file = preprocess_audio_file(str(recording_path))
            text = transcribe_audio_file(cleaned_file)
            return text if text else None
        
        except ImportError:
            try:
                from groq import Groq
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                
                with open(recording_path, "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        file=(recording_path.name, f.read()),
                        model="whisper-large-v3-turbo",
                    )
                return transcription.text.strip()
            
            except Exception as e:
                print(f"   ❌ Groq STT error: {e}")
                return None
        
        except Exception as e:
            print(f"   ❌ STT error: {e}")
            return None
    
    
    async def speak_to_caller(self, sock, text: str):
        """Stream TTS audio to caller in realtime"""
        print(f"   🔊 TTS: \"{text[:50]}...\"")
        
        try:
            # Start async TTS streaming process
            process = await asyncio.create_subprocess_exec(
                "piper",
                "--model", PIPER_MODEL_PATH,
                "--output-raw",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.PIPE
            )

            # Send text to Piper
            process.stdin.write(text.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

            # Stream the audio chunks as they are ready
            while True:
                chunk = await process.stdout.read(320)
                if not chunk:
                    break
                sock.sendall(chunk)

            await process.wait()
            print("   ✅ Streaming playback complete")

        except Exception as e:
            print(f"   ❌ Error during async TTS stream: {e}")
            print("   ✅ Streaming playback complete")



    def generate_tts(self, text: str, output_path: Path):
        """Generate TTS audio file"""
        try:
            from TTS_Service import speak_text_to_file
            speak_text_to_file(text, str(output_path))
        
        except ImportError:
            try:
                from gtts import gTTS
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(str(output_path))
            except Exception as e:
                print(f"   ❌ gTTS error: {e}")
        
        except Exception as e:
            print(f"   ❌ TTS generation error: {e}")
    
    def is_goodbye(self, text: str) -> bool:
        """Check if user wants to end call"""
        if not text:
            return False
        
        text_lower = text.lower()
        goodbye_phrases = [
            'goodbye', 'bye', 'bye bye', 'good bye',
            'thank you bye', 'thanks bye',
            'end call', 'hang up', 'disconnect',
            'that\'s all', 'that is all',
            'nothing else', 'no thanks',
            'i\'m done', 'i am done'
        ]
        
        return any(phrase in text_lower for phrase in goodbye_phrases)


# Global bot instance
bot_instance = None


# FastAPI Routes

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve WebRTC client interface"""
    return WEBRTC_CLIENT_HTML


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        'status': 'ok',
        'service': 'voice-bot-production',
        'endpoints': {
            'webrtc': 'wss://192.168.1.4:8082',
            'sip': 'sip:5000@192.168.1.4',
            'socket': 'localhost:8084'
        }
    }


@app.get("/api/status")
async def status():
    """Get bot status"""
    return {
        'status': 'running',
        'temp_dir': str(TEMP_DIR),
        'sample_rate': 8000,
        'channels': 1,
        'max_turns': 50
    }


@app.on_event("startup")
async def startup_event():
    """Start voice bot server on FastAPI startup"""
    global bot_instance
    
    print("🚀 Starting Production Voice Bot...")
    print(f"📁 Project: {PROJECT_DIR}")
    
    # Check environment
    if not os.environ.get("GROQ_API_KEY"):
        print("⚠️  WARNING: GROQ_API_KEY not set")
        print("   Set it: export GROQ_API_KEY='your-key-here'\n")
    
    # Start voice bot in separate thread
    bot_instance = ProductionVoiceBot(host='0.0.0.0', port=8084)
    bot_thread = threading.Thread(target=bot_instance.start_server, daemon=True)
    bot_thread.start()
    
    print("✅ Both servers started!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n👋 Shutting down Production Voice Bot...")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🚀 PRODUCTION VOICE BOT - WEBRTC + SIP")
    print("="*70)
    print("\nStarting servers...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()