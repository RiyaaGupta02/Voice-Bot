from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import torch
import torchaudio
import tempfile
import os
from typing import List, Dict

class VADService:
    """
    Backend Voice Activity Detection using Silero VAD
    Detects speech and silence in audio files
    """
    
    def __init__(self):
        print("[VAD] Loading Silero VAD model...")
        self.model = load_silero_vad()
        print("[VAD] ✅ Model loaded successfully")
    
    def has_speech(self, audio_path: str, min_speech_duration: float = 0.3) -> bool:
        """
        Check if audio file contains speech
        
        Args:
            audio_path: Path to audio file
            min_speech_duration: Minimum speech duration in seconds
            
        Returns:
            True if speech detected, False otherwise
        """
        try:
            # Read audio file
            wav = read_audio(audio_path)
            
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                wav,
                self.model,
                return_seconds=True,
                min_speech_duration_ms=int(min_speech_duration * 1000)
            )
            
            if len(speech_timestamps) > 0:
                total_speech = sum(
                    ts['end'] - ts['start'] 
                    for ts in speech_timestamps
                )
                print(f"[VAD] Detected {total_speech:.2f}s of speech")
                return total_speech >= min_speech_duration
            
            print("[VAD] No speech detected")
            return False
            
        except Exception as e:
            print(f"[VAD ERROR] {e}")
            # On error, assume there's speech to avoid blocking
            return True
    
    def get_speech_segments(self, audio_path: str) -> List[Dict]:
        """
        Get detailed speech segments with timestamps
        
        Returns:
            List of dicts with 'start' and 'end' timestamps in seconds
        """
        try:
            wav = read_audio(audio_path)
            
            speech_timestamps = get_speech_timestamps(
                wav,
                self.model,
                return_seconds=True
            )
            
            return speech_timestamps
            
        except Exception as e:
            print(f"[VAD ERROR] {e}")
            return []