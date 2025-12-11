" main model_loader.py file --> named using A so that stays at the top of the folder structure"
" ← This file loads all models ONCE"
"Uses model_loader instead of loading Whisper again --> basically for all the models being loaded again & again in different files we will use this model_loader file to load them once & then use them everywhere"

# model_loader.py

from faster_whisper import WhisperModel
import webrtcvad
from sentence_transformers import SentenceTransformer
import chromadb

class ModelManager:
    """Singleton to load models once and reuse"""
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ModelManager._models_loaded:
            print("🔄 Loading models (this happens once)...")
            
            # STT Models
            print("   → Loading Whisper...")
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")  # Changed to 'tiny'
            self.vad = webrtcvad.Vad(2)
            
            # RAG Models
            print("   → Loading sentence transformer...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Vector DB
            print("   → Loading ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(path="./vectordb")
            
            ModelManager._models_loaded = True
            print("✅ All models loaded!\n")
    
    def get_whisper(self):
        return self.whisper_model
    
    def get_vad(self):
        return self.vad
    
    def get_embedder(self):
        return self.embedder
    
    def get_chroma_client(self):
        return self.chroma_client

# Global instance
model_manager = ModelManager()

# model_loader.py

# ... all your existing code ...

# At the very bottom:
if __name__ == "__main__":
    print("⚠️  This is a utility module, not meant to be run directly.")
    print("✅ Run 'main_application.py' instead.")
    print("\nUsage: python main_application.py")