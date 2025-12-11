import asyncio
import os
import tempfile
import wave
import sys
import subprocess
from pathlib import Path
from typing import AsyncGenerator, Optional, Callable

import numpy as np
from scipy import signal

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PIPER_PATH = "piper/piper/piper.exe"
DEFAULT_PIPER_SAMPLE_RATE = 22050  # Piper outputs 22.05k by default
DEFAULT_CHUNK_BYTES = 4096


# ============================================================================
# ASYNC TTS ENGINE (For Production/Telephony)
# ============================================================================

class PiperTTSAsync:
    """Async version of Piper TTS for production/streaming use cases"""
    
    def __init__(self,
                 piper_path: str = DEFAULT_PIPER_PATH,
                 voice_model: str = "piper/voices/en_US-ryan-high.onnx",
                 piper_sample_rate: int = DEFAULT_PIPER_SAMPLE_RATE,
                 chunk_bytes: int = DEFAULT_CHUNK_BYTES):
        self.piper_path = piper_path
        self.voice_model = voice_model
        self.piper_sample_rate = piper_sample_rate
        self.chunk_bytes = chunk_bytes

        if not Path(self.piper_path).exists():
            raise FileNotFoundError(f"Piper executable not found: {self.piper_path}")
        if not Path(self.voice_model).exists():
            raise FileNotFoundError(f"Piper voice model not found: {self.voice_model}")

    async def stream_tts(self, text: str, *, ssml: bool = False, voice: Optional[str] = None,
                         timeout: int = 30) -> AsyncGenerator[bytes, None]:
        """
        Async generator yielding raw PCM (int16 little-endian) chunks as Piper emits them.
        Caller should send yielded bytes to the telephony socket in real-time.
        """
        if not text or text.strip() == "":
            return

        # Simple SSML handling: strip tags for now
        if ssml:
            import re
            text = re.sub(r"<[^>]+>", " ", text)

        voice_model = voice if voice else self.voice_model

        # Launch piper process, piping stdin/stdout
        proc = await asyncio.create_subprocess_exec(
            self.piper_path,
            '--model', str(voice_model),
            '--output-raw',  # raw PCM stdout
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            # Write input text (utf-8) and close stdin
            proc.stdin.write(text.encode('utf-8'))
            await proc.stdin.drain()
            proc.stdin.close()

            # Read raw stdout in chunks
            while True:
                try:
                    chunk = await asyncio.wait_for(proc.stdout.read(self.chunk_bytes), timeout=timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    break

                if not chunk:
                    break

                yield chunk

            await proc.wait()

            # Check stderr
            stderr = await proc.stderr.read()
            if stderr:
                print("Piper stderr:", stderr.decode('utf-8', errors='ignore'), file=sys.stderr)

        except asyncio.CancelledError:
            try:
                proc.kill()
            except Exception:
                pass
            raise
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            raise

    async def generate_wav_file(self, text: str, output_file: str,
                                target_rate: int = 22050) -> Optional[str]:
        """
        Produce a WAV file by streaming Piper to a temp file then resampling+writing WAV file.
        Returns the output_file path on success, else None.
        """
        # Stream into a temp raw PCM file
        tmp_raw = tempfile.NamedTemporaryFile(delete=False)
        tmp_raw.close()
        
        try:
            async for chunk in self.stream_tts(text):
                with open(tmp_raw.name, "ab") as f:
                    f.write(chunk)

            # Convert raw PCM (int16) to numpy
            raw_bytes = Path(tmp_raw.name).read_bytes()
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16)

            # Resample if needed
            if target_rate != self.piper_sample_rate:
                num_samples = int(len(audio_np) * target_rate / self.piper_sample_rate)
                audio_resampled = signal.resample(audio_np, num_samples).astype(np.int16)
            else:
                audio_resampled = audio_np

            # Save WAV
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(target_rate)
                wf.writeframes(audio_resampled.tobytes())

            return output_file
        except Exception as e:
            print("TTS generate_wav_file error:", e)
            return None
        finally:
            try:
                os.unlink(tmp_raw.name)
            except Exception:
                pass


# ============================================================================
# SYNC TTS ENGINE (For Laptop/Testing - Original Implementation)
# ============================================================================

class PiperTTS:
    """Synchronous version of Piper TTS - maintains backward compatibility"""
    
    def __init__(self,
                 piper_path: str = DEFAULT_PIPER_PATH,
                 voice_model: str = "piper/voices/en_US-ryan-high.onnx"):
        # Paths
        self.piper_path = piper_path
        self.model_path = voice_model
        
        # For async compatibility
        self._async_engine = None
        self._loop = None
        
        # Verify files exist
        if not Path(self.piper_path).exists():
            raise FileNotFoundError(f"Piper executable not found: {self.piper_path}")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Voice model not found: {self.model_path}")
        
        # Check for required DLLs
        self._check_dependencies()
        
        print("✅ Piper TTS initialized (Ryan voice)")
    
    def _check_dependencies(self):
        """Check if piper.exe can run (detect missing DLLs)"""
        try:
            result = subprocess.run(
                [self.piper_path, "--help"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0 and result.returncode != 1:
                print(f"⚠️  Warning: Piper may have dependency issues (exit code: {result.returncode})")
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot execute {self.piper_path}")
        except Exception as e:
            print(f"⚠️  Could not verify Piper dependencies: {e}")
    
    def speak_streaming(self, text: str, play_audio: bool = True):
        """
        Convert text to speech with STREAMING (no file saved)
        Uses stdout to avoid saving files
        
        Args:
            text: Text to convert
            play_audio: Whether to play audio immediately
        
        Returns:
            bytes: Raw audio data
        """
        
        if not text or text.strip() == "":
            print("⚠️  Empty text, skipping TTS")
            return None
        
        try:
            print(f"📝 Speaking: '{text[:50]}...'")
            
            # STREAMING MODE: Output to stdout (--output-raw)
            process = subprocess.Popen(
                [
                    self.piper_path,
                    '--model', self.model_path,
                    '--output-raw'
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Send text and get audio data
            audio_data, stderr = process.communicate(input=text.encode('utf-8'), timeout=30)
            
            if stderr:
                stderr_text = stderr.decode('utf-8', errors='ignore')
                if 'error' in stderr_text.lower():
                    print(f"⚠️  Piper stderr: {stderr_text}")
            
            if process.returncode != 0:
                print(f"❌ Piper failed with exit code: {process.returncode}")
                print(f"Error output: {stderr.decode('utf-8', errors='ignore')}")
                return None
            
            if not audio_data or len(audio_data) < 100:
                print(f"❌ No audio data generated (got {len(audio_data) if audio_data else 0} bytes)")
                return None
            
            print(f"✅ Generated {len(audio_data):,} bytes of audio")
            
            # Play audio if requested
            if play_audio:
                self._play_audio_from_bytes(audio_data)
            
            return audio_data

        except subprocess.TimeoutExpired:
            print("❌ TTS timeout - text too long or Piper hung")
            process.kill()
            return None
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _play_audio_from_bytes(self, audio_data: bytes):
        """Play raw PCM audio data directly (no file saved)"""
        
        try:
            if os.name == 'nt':  # Windows
                # Use temporary file (auto-deleted)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                    
                    # Convert raw PCM to WAV
                    with wave.open(tmp_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(22050)  # Piper default
                        wav_file.writeframes(audio_data)
                
                # Play and delete
                subprocess.run(
                    ['powershell', '-c', 
                     f'(New-Object Media.SoundPlayer "{tmp_path}").PlaySync()'],
                    check=True
                )
                os.unlink(tmp_path)
                
            else:  # Linux/Mac
                # Use aplay with stdin
                process = subprocess.Popen(
                    ['aplay', '-r', '22050', '-f', 'S16_LE', '-c', '1'],
                    stdin=subprocess.PIPE
                )
                process.communicate(input=audio_data)
            
            print("🔊 Audio played (no file saved)")
        
        except Exception as e:
            print(f"⚠️  Could not play audio: {e}")
    
    # ========================================================================
    # ASYNC COMPATIBILITY METHODS (Bridge to async engine)
    # ========================================================================
    
    def _ensure_loop(self):
        """Ensure event loop exists for async operations"""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
    
    def _get_async_engine(self) -> PiperTTSAsync:
        """Lazy-load async engine"""
        if self._async_engine is None:
            self._async_engine = PiperTTSAsync(
                piper_path=self.piper_path,
                voice_model=self.model_path
            )
        return self._async_engine
    
    def stream_tts_sync(self, text: str, chunk_callback: Callable[[bytes], None], 
                        ssml: bool = False, voice: Optional[str] = None):
        """
        Synchronous wrapper for async streaming that calls chunk_callback(chunk_bytes) for each chunk.
        Useful for telephony integration.
        
        Args:
            text: Text to convert
            chunk_callback: Function called with each audio chunk
            ssml: Whether text contains SSML
            voice: Optional voice override
        """
        self._ensure_loop()
        
        async def _runner():
            async for chunk in self._get_async_engine().stream_tts(text, ssml=ssml, voice=voice):
                chunk_callback(chunk)
        
        self._loop.run_until_complete(_runner())
    
    def generate_wav_file_sync(self, text: str, output_file: str, target_rate: int = 22050) -> Optional[str]:
        """
        Synchronous wrapper for async WAV file generation.
        
        Args:
            text: Text to convert
            output_file: Output WAV file path
            target_rate: Target sample rate (default 22050)
        
        Returns:
            str: Path to generated file, or None on failure
        """
        self._ensure_loop()
        return self._loop.run_until_complete(
            self._get_async_engine().generate_wav_file(text, output_file, target_rate)
        )


# ============================================================================
# FREESWITCH INTEGRATION (File output for telephony)
# ============================================================================

def speak_text_to_file(text: str, output_file: str = "response.wav") -> Optional[str]:
    """
    Generate TTS and save to file (for FreeSWITCH integration)
    Converts Piper's raw PCM to standard WAV format for telephony
    
    Args:
        text: Text to convert to speech
        output_file: Output WAV file path
    
    Returns:
        str: Path to generated file, or None on failure
    """
    global _tts_engine
    
    if _tts_engine is None:
        _tts_engine = PiperTTS()
    
    if not text or text.strip() == "":
        print("⚠️  Empty text, skipping TTS")
        return None
    
    try:
        print(f"📝 Generating TTS to file: '{text[:50]}...'")
        
        # Generate audio (streaming mode, no playback)
        audio_data = _tts_engine.speak_streaming(text, play_audio=False)
        
        if not audio_data:
            print("❌ TTS generation failed")
            return None
        
        # Convert raw PCM to WAV file
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)      # Mono
            wav_file.setsampwidth(2)       # 16-bit
            wav_file.setframerate(22050)   # Piper default
            wav_file.writeframes(audio_data)
        
        print(f"✅ Audio saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ TTS file generation error: {e}")
        import traceback
        traceback.print_exc()
        return None


def speak_text_to_file_telephony(text: str, output_file: str = "response.wav", 
                                  sample_rate: int = 8000) -> Optional[str]:
    """
    Generate TTS optimized for telephony (8kHz)
    FreeSWITCH works best with 8kHz audio for phone calls
    
    Args:
        text: Text to convert
        output_file: Output file path
        sample_rate: 8000 for telephony, 22050 for high quality
    
    Returns:
        str: Path to generated file, or None on failure
    """
    global _tts_engine
    
    if _tts_engine is None:
        _tts_engine = PiperTTS()
    
    try:
        # Generate at native rate first
        audio_data = _tts_engine.speak_streaming(text, play_audio=False)
        
        if not audio_data:
            return None
        
        # Convert to numpy for resampling
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Resample to target rate if needed
        if sample_rate != 22050:
            num_samples = int(len(audio_np) * sample_rate / 22050)
            audio_resampled = signal.resample(audio_np, num_samples)
            audio_resampled = audio_resampled.astype(np.int16)
        else:
            audio_resampled = audio_np
        
        # Save as WAV
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_resampled.tobytes())
        
        print(f"✅ Telephony audio ({sample_rate}Hz) saved to: {output_file}")
        return output_file
        
    except ImportError:
        print("⚠️  scipy not available, using basic WAV save")
        return speak_text_to_file(text, output_file)
    except Exception as e:
        print(f"❌ Telephony TTS error: {e}")
        return None


# ============================================================================
# SIMPLE API (Backward compatible with existing code)
# ============================================================================

_tts_engine = None


def speak_text(text: str):
    """
    Speak text with STREAMING (no file saved)
    BACKWARD COMPATIBLE - works with existing Main_completion.py
    
    Args:
        text: What to say
    
    Returns:
        bytes: Audio data
    """
    global _tts_engine
    
    if _tts_engine is None:
        _tts_engine = PiperTTS()
    
    return _tts_engine.speak_streaming(text, play_audio=True)


# ============================================================================
# DIRECT TEST
# ============================================================================

if __name__ == "__main__":
    print("🔧 Testing Piper TTS (Sync + Async Mode)...\n")
    
    # Test if piper.exe has DLL issues
    print("📋 Checking dependencies...")
    piper_path = Path("piper/piper/piper.exe")
    
    if piper_path.exists():
        result = subprocess.run(
            [str(piper_path), "--help"],
            capture_output=True
        )
        
        if result.returncode == 3221225781 or result.returncode == -1073741515:
            print("\n❌ CRITICAL ERROR: Missing DLL dependencies!")
            print("\n🔧 SOLUTIONS:")
            print("1. Download Microsoft Visual C++ Redistributable:")
            print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("\n2. OR use piper_phonemize.exe from GitHub:")
            print("   https://github.com/rhasspy/piper/releases")
            sys.exit(1)
    
    # Test sync mode (for laptop)
    print("\n1️⃣  Testing SYNC mode (laptop)...")
    test_text = "Hello! This is a synchronous streaming test."
    
    try:
        audio_data = speak_text(test_text)
        if audio_data:
            print(f"✅ Sync test successful! Generated {len(audio_data):,} bytes\n")
        else:
            print("❌ Sync test failed - no audio generated\n")
    except Exception as e:
        print(f"❌ Sync TTS Test Failed: {e}\n")
    
    # Test async mode (for production)
    print("2️⃣  Testing ASYNC mode (production)...")
    
    async def test_async():
        try:
            tts_async = PiperTTSAsync()
            chunks_received = 0
            total_bytes = 0
            
            async for chunk in tts_async.stream_tts("This is an async streaming test for production."):
                chunks_received += 1
                total_bytes += len(chunk)
            
            print(f"✅ Async test successful! Received {chunks_received} chunks, {total_bytes:,} bytes total")
        except Exception as e:
            print(f"❌ Async TTS Test Failed: {e}")
    
    asyncio.run(test_async())
    
    print("\n" + "="*70)
    print("✅ All tests complete!")
    print("="*70)