# FreeSWITCH Voice Bot Integration
# Connects FreeSWITCH to your existing voice bot pipeline

import socket
import os
import wave
import struct
from pathlib import Path

# Import your existing functions
from Intent_Router_Thinkingprocess import process_user_query

# Setup paths for your project
PROJECT_DIR = Path(__file__).parent
TEMP_DIR = PROJECT_DIR / "temp_audio"
TEMP_DIR.mkdir(exist_ok=True)


class FreeSwitchVoiceBot:
    """Connects FreeSWITCH to your existing voice bot pipeline"""
    
    def __init__(self, host='127.0.0.1', port=8084):
        self.host = host
        self.port = port
        self.sample_rate = 8000  # FreeSWITCH default
        self.channels = 1
        self.recording_path = TEMP_DIR / "recording.wav"
        self.response_path = TEMP_DIR / "response.wav"
        
    def start_server(self):
        """Start socket server for FreeSWITCH connections"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind((self.host, self.port))
            server.listen(5)
            
            print("\n" + "="*70)
            print("🏠 REAL ESTATE VOICE BOT - FREESWITCH MODE")
            print("="*70)
            print(f"✅ Server running on {self.host}:{self.port}")
            print(f"✅ Project: {PROJECT_DIR}")
            print(f"✅ Temp audio: {TEMP_DIR}")
            print("\n📞 DIAL 5000 from SIP client to test")
            print("🛑 Press Ctrl+C to stop\n")
            print("="*70)
            
            call_count = 0
            
            while True:
                try:
                    client_socket, address = server.accept()
                    call_count += 1
                    print(f"\n{'='*70}")
                    print(f"📞 CALL #{call_count} from {address}")
                    print(f"{'='*70}")
                    
                    self.handle_call(client_socket, call_count)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Shutting down...")
                    print(f"📊 Total calls: {call_count}")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
        finally:
            server.close()
    
    def handle_call(self, sock, call_num):
        """Handle individual call"""
        
        try:
            # Read FreeSWITCH channel data
            print("📋 Reading channel data...")
            channel_data = self.read_channel_data(sock)
            
            if channel_data:
                print(f"   Channel: {channel_data.get('Channel-Unique-ID', 'N/A')[:8]}...")
            
            # Send connect command
            self.send_command(sock, "connect\n\n")
            
            # Welcome message
            print("🎤 Playing welcome...")
            welcome = "Hello! Welcome to our real estate agency. I'm your AI assistant. How can I help you today?"
            self.speak_to_caller(sock, welcome)
            
            turn = 0
            
            # Conversation loop
            while True:
                turn += 1
                print(f"\n{'─'*50}")
                print(f"🎯 Turn {turn}")
                print(f"{'─'*50}")
                
                # Record user
                print("👂 Listening...")
                recorded = self.record_from_caller(sock)
                
                if not recorded:
                    print("⚠️  No audio")
                    self.speak_to_caller(sock, "I didn't hear you. Please speak after the beep.")
                    continue
                
                # Transcribe
                print("🎧 Transcribing...")
                user_text = self.transcribe_audio()
                
                if not user_text:
                    print("⚠️  Transcription failed")
                    self.speak_to_caller(sock, "Sorry, I couldn't understand. Please speak clearly.")
                    continue
                
                print(f"📝 User: \"{user_text}\"")
                
                # Check goodbye
                if self.is_goodbye(user_text):
                    print("👋 Ending call")
                    self.speak_to_caller(sock, "Thank you for calling! Have a great day. Goodbye!")
                    break
                
                # Process with YOUR pipeline
                print("🤔 Processing with your Intent Router...")
                response = process_user_query(user_text)
                
                if not response:
                    response = "I apologize, I couldn't process that. Could you rephrase?"
                
                print(f"💬 Bot: {response[:80]}...")
                
                # Speak response
                print("🔊 Speaking...")
                self.speak_to_caller(sock, response)
                
                print("✅ Turn complete")
                
        except Exception as e:
            print(f"❌ Call error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                self.send_command(sock, "exit\n\n")
            except:
                pass
            sock.close()
            print(f"📴 Call #{call_num} ended\n")
    
    def read_channel_data(self, sock):
        """Read FreeSWITCH channel information"""
        data = {}
        sock.settimeout(2.0)
        
        try:
            while True:
                line = sock.recv(1024).decode('utf-8', errors='ignore')
                if not line or line == '\n':
                    break
                
                if ':' in line:
                    key, value = line.split(':', 1)
                    data[key.strip()] = value.strip()
        except socket.timeout:
            pass
        except Exception as e:
            print(f"   Warning: Error reading channel data: {e}")
        
        return data
    
    def send_command(self, sock, command):
        """Send command to FreeSWITCH"""
        try:
            sock.sendall(command.encode('utf-8'))
        except Exception as e:
            print(f"   Error sending command: {e}")
    
    def record_from_caller(self, sock, max_duration=5):
        """Record audio from caller"""
        print(f"   Recording (max {max_duration}s)...")
        
        audio_chunks = []
        sock.settimeout(0.2)
        
        try:
            frames_recorded = 0
            silent_frames = 0
            recording_started = False
            
            for _ in range(max_duration * 10):  # Check 10 times per second
                try:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    
                    audio_chunks.append(chunk)
                    frames_recorded += 1
                    
                    # Simple voice activity detection
                    if len(chunk) >= 2:
                        samples = struct.unpack(f'{len(chunk)//2}h', chunk)
                        max_amp = max(abs(s) for s in samples) if samples else 0
                        
                        if max_amp > 500:  # Threshold
                            recording_started = True
                            silent_frames = 0
                        elif recording_started:
                            silent_frames += 1
                            
                            if silent_frames > 15:  # 1.5 seconds silence
                                print("   ✅ Speech completed")
                                break
                
                except socket.timeout:
                    if recording_started:
                        silent_frames += 1
                        if silent_frames > 15:
                            break
                    continue
            
            # Save to WAV
            if audio_chunks:
                with wave.open(str(self.recording_path), 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(audio_chunks))
                
                print(f"   💾 Saved {frames_recorded} frames")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"   ❌ Recording error: {e}")
            return False
    
    def transcribe_audio(self):
        """Transcribe recorded audio using YOUR STT service"""
        try:
            # Use your existing STT
            from STT_Services import transcribe_audio_file
            
            if self.recording_path.exists():
                text = transcribe_audio_file(str(self.recording_path))
                return text
            else:
                return None
                
        except ImportError:
            print("   ⚠️  STT_Services not found, using placeholder")
            # Fallback: Use Groq directly
            try:
                import os
                from groq import Groq
                
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                
                with open(self.recording_path, "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        file=(self.recording_path.name, f.read()),
                        model="whisper-large-v3-turbo",
                    )
                    return transcription.text.strip()
            except Exception as e:
                print(f"   ❌ STT error: {e}")
                return None
        except Exception as e:
            print(f"   ❌ Transcription error: {e}")
            return None
    
    def speak_to_caller(self, sock, text):
        """Generate TTS and play to caller"""
        try:
            print(f"   🔊 TTS: \"{text[:50]}...\"")
            
            # Generate TTS using YOUR service
            self.generate_tts(text)
            
            if not self.response_path.exists():
                print("   ❌ TTS file not generated")
                return
            
            # Stream audio to FreeSWITCH
            with wave.open(str(self.response_path), 'rb') as wf:
                chunk_size = 320  # 20ms at 8kHz
                
                while True:
                    data = wf.readframes(chunk_size)
                    if not data:
                        break
                    sock.sendall(data)
            
            print("   ✅ Playback done")
            
        except Exception as e:
            print(f"   ❌ TTS playback error: {e}")
    
    def generate_tts(self, text):
        """Generate TTS audio using YOUR TTS service"""
        try:
            # Try to use your existing TTS
            from TTS_Service import speak_text_to_file
            
            speak_text_to_file(text, str(self.response_path))
            
        except ImportError:
            print("   ⚠️  TTS_Service not found, using gTTS")
            # Fallback: Use gTTS
            try:
                from gtts import gTTS
                
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(str(self.response_path))
                
            except Exception as e:
                print(f"   ❌ TTS generation error: {e}")
        except Exception as e:
            print(f"   ❌ TTS error: {e}")
    
    def is_goodbye(self, text):
        """Check if user wants to end call"""
        if not text:
            return False
        
        text_lower = text.lower()
        goodbye_words = [
            'goodbye', 'bye', 'bye bye', 'thank you bye',
            'end call', 'hang up', 'that\'s all',
            'nothing else', 'no thanks', 'i\'m done'
        ]
        
        return any(word in text_lower for word in goodbye_words)


def check_environment():
    """Check if environment is set up correctly"""
    print("\n🔍 Checking environment...")
    
    issues = []
    
    # Check GROQ API key
    if not os.environ.get("GROQ_API_KEY"):
        issues.append("⚠️  GROQ_API_KEY not set")
        print("   Set with: $env:GROQ_API_KEY='your-key'")
    else:
        print("   ✅ GROQ_API_KEY set")
    
    # Check imports
    try:
        from Intent_Router_Thinkingprocess import process_user_query
        print("   ✅ Intent_Router_Thinkingprocess found")
    except ImportError as e:
        issues.append(f"⚠️  Intent_Router_Thinkingprocess not found: {e}")
    
    try:
        from STT_Services import transcribe_audio_file
        print("   ✅ STT_Services found")
    except ImportError:
        print("   ⚠️  STT_Services not found (will use fallback)")
    
    try:
        from TTS_Service import speak_text_to_file
        print("   ✅ TTS_Service found")
    except ImportError:
        print("   ⚠️  TTS_Service not found (will use gTTS)")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
        print()
    else:
        print("\n✅ Environment OK!\n")
    
    return len(issues) == 0


def main():
    """Main entry point"""
    print("🚀 Starting FreeSWITCH Voice Bot...")
    print(f"📁 Project: {PROJECT_DIR}")
    
    # Check environment
    check_environment()
    
    # Start bot
    bot = FreeSwitchVoiceBot()
    bot.start_server()


if __name__ == "__main__":
    main()