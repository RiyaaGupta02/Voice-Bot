"""
STT Service (Record → Noise Handling → Whisper → Corrections → Final Clean Text)
Optimized for: Phone audio, Indian accent, locality names, noisy environments
Compatible with current architecture — NO breaking changes.
"""

import atexit
import collections
import io
import re
import traceback
import os

from difflib import get_close_matches
from collections import Counter, deque

import jellyfish            # pip install jellyfish
import noisereduce as nr    # pip install noisereduce
import numpy as np
import pyaudio
import soundfile as sf      # pip install soundfile

# Model loader (singleton) - unchanged usage
from A_model_loader import model_manager    

# ---------------------------------------------------
# Config / Globals
# ---------------------------------------------------

DEBUG = False  # Set True for verbose logs during development

# ---------------------------------------------------
# Load Models from Model Loader (Singleton)
# ---------------------------------------------------

whisper_model = model_manager.get_whisper()
vad = model_manager.get_vad()

# ---------------------------------------------------
# Audio Globals for Cleanup
# ---------------------------------------------------

audio_interface = None
audio_stream = None


def cleanup_audio():
    """Ensure audio resources are released"""
    global audio_stream, audio_interface
    try:
        if audio_stream:
            audio_stream.stop_stream()
            audio_stream.close()
            audio_stream = None
        if audio_interface:
            audio_interface.terminate()
            audio_interface = None
    except Exception:
        # swallow cleanup errors
        pass


# Register cleanup on exit
atexit.register(cleanup_audio)


# ---------------------------------------------------
# Locality & Phonetic Correction Layer (Conservative)
# ---------------------------------------------------

LOCALITY_MAP = {
    "paldi": ["valdi", "baldi", "paldy", "waldy", "baldy", "paldhi"],
    "bopal": ["popal", "vopal", "gopal", "bopel"],
    "naranpura": ["naran pura", "narampura", "narampuda"],
    "maninagar": ["manninger", "maningar", "maninagur", "maninagarh"],
    "gota": ["goda", "gotaah", "gotta"],
}

# Phonetic locality mapping (kept but used conservatively)
LOCALITY_PHONETIC_MAP = {
    "bodakdev": ["bodak dev", "bodakdev", "bodek dev", "boda dev", "vodakdev"],
    "satellite": ["satelite", "sattelite", "satalite", "satellitte"],
    "gota": ["goda", "gotaah", "gotta", "ghota"],
    "paldi": ["valdi", "baldi", "paldy", "waldy"],
    "bopal": ["popal", "vopal", "gopal", "bopel"],
    "vastrapur": ["wastrapur", "vastra pur", "vaastrapur"],
    "prahlad nagar": ["prahlad nagar", "prahladnagar", "prahlad", "prahaladnagar"],
    "thaltej": ["thaltej", "taltej", "thaltaij", "thaltaj"],
    "maninagar": ["manninger", "maningar", "maninagur"],
    "sg highway": ["sg highway", "s g highway", "sg high way", "es ji highway"]
}

# Known vocabulary (include locality keys explicitly)
KNOWN_VOCAB = (
    list(LOCALITY_MAP.keys())
    + list(LOCALITY_PHONETIC_MAP.keys())
    + [
        "price", "rent", "availability", "bhk", "property",
        "area", "safety", "connectivity", "location", "near",
        "tell", "about", "compare", "house", "flat", "buy", "sell",
        "what", "how", "is", "the", "in", "for", "i", "you", "me",
        "please", "there", "available", "show"
    ]
)

# Token regex - preserves punctuation as separate tokens
_TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", flags=re.UNICODE)


def _tokenize_preserve(text):
    """Return tokens preserving punctuation and numbers."""
    return _TOKEN_RE.findall(text)


def apply_locality_corrections(text):
    """
    Deterministic locality correction pass.
    This pass should be safe and small-scope (only replace known mapped patterns).
    """
    if not text:
        return ""

    tl = text.lower()

    # First pass: exact error corrections from LOCALITY_MAP
    for correct, errors in LOCALITY_MAP.items():
        for wrong in errors:
            # word boundary replace to avoid partial accidental matches
            tl = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, tl)

    # Second pass: phonetic-like deterministic replacements from LOCALITY_PHONETIC_MAP
    # These are still deterministic string replacements (safe), not fuzzy matching.
    for correct, variations in LOCALITY_PHONETIC_MAP.items():
        for variant in variations:
            tl = re.sub(r'\b' + re.escape(variant) + r'\b', correct, tl)

    return tl


def phonetic_correct(text, max_rel_distance=0.40, max_absolute_distance=2):
    """
    Conservative phonetic + fuzzy corrector.
    - Only replaces a token if:
      * metaphone matches exactly, OR
      * normalized Levenshtein distance <= max_rel_distance AND absolute distance <= max_absolute_distance.
    - Otherwise, preserve original token.
    """

    if not text:
        return ""

    tokens = _tokenize_preserve(text)
    corrected_tokens = []

    # Precompute metaphone of known vocab for speed
    vocab_metaphone = {w: jellyfish.metaphone(w) for w in KNOWN_VOCAB}

    for tok in tokens:
        # preserve punctuation tokens & numeric tokens unchanged
        if re.fullmatch(r"[^\w\s]", tok) or re.fullmatch(r"\d+", tok):
            corrected_tokens.append(tok)
            continue

        w_lower = tok.lower()

        # If already in known vocab, preserve it (lowercased)
        if w_lower in KNOWN_VOCAB:
            corrected_tokens.append(w_lower)
            continue

        # Compute metaphone
        try:
            w_meta = jellyfish.metaphone(w_lower)
        except Exception:
            w_meta = ""

        best_match = None
        best_abs = 999
        best_rel = 1.0

        # 1) Exact metaphone match (high confidence)
        if w_meta:
            for vocab_word, vm in vocab_metaphone.items():
                if vm and vm == w_meta:
                    best_match = vocab_word
                    best_abs = 0
                    best_rel = 0.0
                    break

        # 2) If no metaphone hit, compute small Levenshtein candidates
        if best_match is None:
            for vocab_word in KNOWN_VOCAB:
                try:
                    d = jellyfish.levenshtein_distance(w_lower, vocab_word)
                except Exception:
                    continue
                norm = d / max(len(w_lower), len(vocab_word), 1)
                # prefer lower absolute distances, then relative
                if (d < best_abs) or (d == best_abs and norm < best_rel):
                    best_match = vocab_word
                    best_abs = d
                    best_rel = norm

        # Decide whether to accept replacement
        accept = False
        if best_match:
            if best_rel == 0.0:
                accept = True  # metaphone exact
            elif (best_abs <= max_absolute_distance) and (best_rel <= max_rel_distance):
                accept = True

        if accept and best_match != w_lower:
            corrected_tokens.append(best_match)
        else:
            # leave the original (but lowercased for consistency)
            corrected_tokens.append(w_lower)

    # Rejoin tokens: avoid spaces before punctuation
    out = ""
    for t in corrected_tokens:
        if out and re.match(r"[^\w\s]", t):  # punctuation right after previous token
            out += t
        elif not out:
            out += t
        else:
            out += " " + t

    return out


def clean_stt_text(text):
    """
    Master normalization layer (conservative):
    - Lowercases and trims.
    - Always run small deterministic locality corrections.
    - Run phonetic/fuzzy correction only if the output looks 'uncertain' enough to justify it.
    """

    if not text:
        return ""

    txt = text.strip()
    txt_lower = txt.lower()

    # 1) Deterministic locality fixes (safe)
    txt_locality_fixed = apply_locality_corrections(txt_lower)

    # 2) Heuristics to decide if we need fuzzy phonetic corrections
    tokens = [t for t in _tokenize_preserve(txt_locality_fixed) if re.match(r"[\w']+", t)]

    if not tokens:
        return txt_locality_fixed

    # Count how many tokens are already in known vocab or look like numerics/roman numerals
    known_count = 0
    for w in tokens:
        if w in KNOWN_VOCAB or re.fullmatch(r"\d+|[ivx]+", w):
            known_count += 1

    prop_known = known_count / max(len(tokens), 1)

    # If majority of tokens are already known, skip fuzzy correction (trust Whisper)
    if prop_known >= 0.65:
        if DEBUG:
            print(f"   ℹ️ Skipping fuzzy correction (prop_known={prop_known:.2f})")
        return txt_locality_fixed

    # Otherwise run conservative phonetic correction
    corrected = phonetic_correct(txt_locality_fixed)

    if DEBUG:
        print(f"   🧩 After locality fixes: '{txt_locality_fixed}'")
        print(f"   ✨ After phonetic/fuzzy: '{corrected}'")

    return corrected


def validate_transcription(text):
    """
    Reject obviously wrong transcriptions
    Returns: (is_valid, cleaned_text)
    Conservative validation: only reject when clear signals of failure are present.
    """
    if not text or len(text.strip()) < 3:
        return False, ""

    text = text.strip().lower()
    words = [w for w in re.findall(r"[\w']+", text)]

    if not words:
        return False, ""

    # Reject if too many duplicates (more than 40% repeated words)
    unique_words = set(w.lower() for w in words)
    if len(words) > 5 and len(unique_words) / len(words) < 0.6:
        if DEBUG:
            print("   ⚠️  Too many duplicates detected")
        return False, ""

    # Reject if same word appears more than 3 times
    word_counts = Counter(w.lower() for w in words)
    if any(count > 3 for count in word_counts.values()):
        if DEBUG:
            print(f"   ⚠️  Excessive repetition: {word_counts.most_common(3)}")
        return False, ""

    # Check if it contains at least one real-estate-related keyword for short phrases
    valid_keywords = {
        'price', 'rent', 'bhk', 'apartment', 'house', 'flat',
        'locality', 'area', 'location', 'cost', 'available',
        'what', 'how', 'tell', 'know', 'find', 'in', 'near', 'show'
    }

    has_valid_keyword = any(kw in text for kw in valid_keywords)

    # More lenient: if the phrase is long enough, accept it (likely a sentence)
    if not has_valid_keyword and len(words) < 8:
        if DEBUG:
            print("   ⚠️  No valid real estate keywords found in short phrase")
        return False, ""

    return True, text


# ---------------------------------------------------
# VAD (Voice Activity Detection)
# ---------------------------------------------------

def is_speech(frame, sample_rate=16000):
    try:
        return vad.is_speech(frame, sample_rate)
    except Exception:
        # If VAD fails, conservatively assume speech to avoid false negatives
        return True


# ---------------------------------------------------
# Audio Recording With Noise Reduction
# ---------------------------------------------------

def record_until_silence(silence_duration=1.5, max_duration=15):
    """
    Records audio until silence is detected using WebRTC VAD.
    Adds noise reduction + best device detection.
    Returns a filename to the saved WAV or None on failure/no-speech.
    """
    global audio_interface, audio_stream

    CHUNK = 480             # 30ms @ 16kHz
    RATE = 16000
    FORMAT = pyaudio.paInt16

    try:
        audio_interface = pyaudio.PyAudio()
    except Exception as e:
        if DEBUG:
            print(" ⚠️ PyAudio init failed:", e)
        return None

    # Auto-select a working microphone
    input_device_index = None
    try:
        for i in range(audio_interface.get_device_count()):
            dev_info = audio_interface.get_device_info_by_index(i)
            if dev_info.get('maxInputChannels', 0) > 0:
                input_device_index = i
                break
    except Exception:
        # If device enumeration fails, leave index None and let open choose default
        input_device_index = None

    try:
        audio_stream = audio_interface.open(
            format=FORMAT,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=CHUNK
        )
    except Exception as e:
        if DEBUG:
            print(" ⚠️ Could not open audio stream:", e)
        cleanup_audio()
        return None

    frames = []
    triggered = False
    chunk_count = 0
    silence_chunks = int(silence_duration * RATE / CHUNK)
    max_chunks = int(max_duration * RATE / CHUNK)
    silence_buffer = deque(maxlen=silence_chunks)

    if DEBUG:
        print("🎤 Listening... (speak now)")

    try:
        while chunk_count < max_chunks:
            chunk = audio_stream.read(CHUNK, exception_on_overflow=False)
            frames.append(chunk)
            chunk_count += 1

            is_talking = is_speech(chunk, RATE)
            silence_buffer.append(is_talking)

            if not triggered and is_talking:
                triggered = True
                if DEBUG:
                    print("🗣️ Recording...")

            if triggered and len(silence_buffer) == silence_chunks and sum(silence_buffer) == 0:
                if DEBUG:
                    print("✅ Detected end of speech; processing...")
                break

    except Exception as e:
        if DEBUG:
            print(" ⚠️ Error while recording:", e)
    finally:
        cleanup_audio()

    # No speech detected
    if not frames or not triggered:
        if DEBUG:
            print(" ⚠️ No speech detected in recording")
        return None

    # ------------------------------------------------------------
    # AUDIO POST-PROCESSING (Noise Reduction + WAV creation)
    # ------------------------------------------------------------

    # Join frames
    pcm_data = b"".join(frames)

    # Convert PCM → float32
    try:
        audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        audio_np = audio_np / 32768.0   # Normalize
    except Exception as e:
        if DEBUG:
            print(" ⚠️ Failed to convert PCM data:", e)
        return None

    # Apply noise reduction safely (guard the call)
    try:
        reduced_noise = nr.reduce_noise(
            y=audio_np,
            sr=RATE,
            stationary=True,
            prop_decrease=0.8,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50
        )
    except Exception:
        # If noise reduction fails, fall back to original audio (safer than crashing)
        reduced_noise = audio_np
        if DEBUG:
            print(" ⚠️ Noise reduction failed; using raw audio")

    # Convert back to int16 and clip
    reduced_noise = (reduced_noise * 32768.0).astype(np.int16)
    reduced_noise = np.clip(reduced_noise, -32768, 32767)

    # Save to WAV bytes
    wav_bytes = io.BytesIO()
    try:
        sf.write(wav_bytes, reduced_noise, RATE, format='wav')
    except Exception as e:
        if DEBUG:
            print(" ⚠️ soundfile.write failed:", e)
        return None

    wav_bytes.seek(0)

    # Save to file
    filename = "recorded_audio.wav"
    try:
        with open(filename, "wb") as f:
            f.write(wav_bytes.read())
    except Exception as e:
        if DEBUG:
            print(" ⚠️ Failed to write audio file:", e)
        return None

    return filename


# ---------------------------------------------------
# Whisper Transcription (Faster-Whisper compatible)
# ---------------------------------------------------

def remove_repetitions(text):
    """
    Remove consecutive duplicate words AND phrases.
    Handles: "gota gota", "bhk bhk", "price price price"
    Also attempts to remove simple alternating duplicates like "rent price rent price".
    """

    if not text:
        return ""

    words = text.split()

    if len(words) < 2:
        return text

    cleaned = []
    i = 0

    while i < len(words):
        word = words[i]
        # Count consecutive duplicates
        duplicate_count = 1
        while (i + duplicate_count < len(words) and
               words[i + duplicate_count].lower() == word.lower()):
            duplicate_count += 1

        cleaned.append(words[i])
        i += duplicate_count

    result = " ".join(cleaned)

    # Additional pass: collapse simple alternating duplicates (e.g. "rent price rent price")
    result_words = result.split()
    if len(result_words) >= 4:
        final = []
        skip_next = False
        for idx in range(len(result_words)):
            if skip_next:
                skip_next = False
                continue

            if (idx + 2 < len(result_words) and
                    result_words[idx].lower() == result_words[idx + 2].lower()):
                final.append(result_words[idx])
                skip_next = True
            else:
                final.append(result_words[idx])

        result = " ".join(final)

    return result


def transcribe_audio(audio_path):
    """Convert audio with conservative faster-whisper parameters and minimal post-processing"""
    if not audio_path:
        return ""

    try:
        # FASTER-WHISPER COMPATIBLE PARAMETERS ONLY
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters={
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 2000,
                "speech_pad_ms": 400
            },
            temperature=0.0,
            initial_prompt=(
                "This is a conversation about real estate properties in Ahmedabad, India. "
                "Property types include 1 BHK, 2 BHK, 3 BHK apartments. Localities include "
                "Bodakdev, Satellite, Bopal, SG Highway, Vastrapur, Prahlad Nagar, Thaltej, "
                "Maninagar, Gota, Paldi."
            ),
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=[-1],
            without_timestamps=False,
            max_initial_timestamp=1.0,
            word_timestamps=False
        )

        # Extract text conservatively
        raw_text = " ".join([segment.text for segment in segments]).strip()

        if DEBUG:
            print(f"   🗣️  Whisper raw: '{raw_text}'")

        if not raw_text:
            if DEBUG:
                print("   ⚠️  Empty transcription from Whisper")
            return ""

        # Remove obvious repeated tokens
        deduped = remove_repetitions(raw_text)
        if DEBUG and deduped != raw_text:
            print(f"   🔄 After dedup: '{deduped}'")
        else:
            if DEBUG:
                print(f"   🔄 Dedup not needed")

        # Conservative cleaning (locality corrections always applied)
        cleaned = clean_stt_text(deduped)
        if DEBUG:
            print(f"   ✨ Final clean: '{cleaned}'")

        return cleaned

    except Exception as e:
        print(f"⚠️ Whisper error: {e}")
        if DEBUG:
            traceback.print_exc()
        return ""


# ===================================================
# FUNCTIONALITY JUST FOR FREESWITCH INTEGRATION --> NOT FOR MAIN FILE THAT RUNS ON LAPTOP
# ===================================================
def preprocess_audio_file(input_path: str) -> str:
    """
    Takes FreeSWITCH-recorded WAV file and applies:
    - noise reduction
    - trimming silence
    - VAD cleanup
    - normalization
    - DC offset removal
    - channel/sample rate corrections

    Returns path to cleaned WAV file.
    """
    import librosa
    import soundfile as sf
    import numpy as np
    import tempfile
    import os

    # Load audio (mono)
    audio, sr = librosa.load(input_path, sr=16000, mono=True)

    # Apply noise reduction
    try:
        import noisereduce as nr
        audio = nr.reduce_noise(y=audio, sr=sr)
    except:
        pass

    # Trim silence (librosa)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)

    # Normalize volume
    max_val = np.max(np.abs(audio_trimmed))
    if max_val > 0:
        audio_norm = audio_trimmed / max_val * 0.9
    else:
        audio_norm = audio_trimmed

    # Create a cleaned temp file
    cleaned_path = os.path.join(tempfile.gettempdir(), "cleaned_recording.wav")
    sf.write(cleaned_path, audio_norm, 16000)

    return cleaned_path


# ---------------------------------------------------
# Main Export Function
# ---------------------------------------------------

def get_user_speech():
    """Main STT entry point with validation"""
    audio_file = record_until_silence()
    if not audio_file:
        return ""

    text = transcribe_audio(audio_file)

    # Validate before returning
    is_valid, validated_text = validate_transcription(text)

    if not is_valid:
        if DEBUG:
            print("   🔴 Transcription rejected by validator")
        return ""

    return validated_text


# ============================================================================
# FREESWITCH INTEGRATION (File-based transcription)
# ============================================================================

def transcribe_audio_file(audio_file_path):
    """
    Transcribe audio from file path (for FreeSWITCH integration)
    Uses same pipeline as get_user_speech() but reads from file instead of mic
    
    Args:
        audio_file_path: Path to WAV file (8kHz or 16kHz)
    
    Returns:
        str: Transcribed and cleaned text, or empty string on failure
    """
    
    if not audio_file_path or not os.path.exists(audio_file_path):
        if DEBUG:
            print(f"⚠️  Audio file not found: {audio_file_path}")
        return ""
    
    try:
        if DEBUG:
            print(f"📂 Transcribing file: {audio_file_path}")
        
        # Use same transcription pipeline as microphone input
        text = transcribe_audio(audio_file_path)
        
        # Use same validation as get_user_speech()
        is_valid, validated_text = validate_transcription(text)
        
        if not is_valid:
            if DEBUG:
                print("   🔴 File transcription rejected by validator")
            return ""
        
        return validated_text
        
    except Exception as e:
        print(f"⚠️ File transcription error: {e}")
        if DEBUG:
            traceback.print_exc()
        return ""
    


# ---------------------------------------------------
# Local Test Runner & Diagnostics
# ---------------------------------------------------

if __name__ == "__main__":
    # Quick interactive debug runner
    DEBUG = True
    print("=" * 60)
    print("  STT Test — Noise Reduction + Conservative Locality Fixes")
    print("=" * 60)

    try:
        while True:
            _ = input("\nPress Enter to speak…")
            out_text = get_user_speech()
            if out_text:
                print("\n📝 Clean STT Output:", out_text)
            else:
                print("\n❌ No valid speech detected or transcription rejected")
            print("-" * 60)
    except KeyboardInterrupt:
        print("\nExiting...")
