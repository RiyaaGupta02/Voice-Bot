"""
Voice Bot Main Execution
Architecture: STT → Intent Router → Knowledge Base → TTS 
"""

import os
from STT_Services import get_user_speech
from Intent_Router_Thinkingprocess import process_user_query
from TTS_Service import speak_text
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()

# Verify API key is loaded
if not os.getenv("GROQ_API_KEY"):
    print("❌ ERROR: GROQ_API_KEY not found in .env file!")
    print("Please create a .env file with: GROQ_API_KEY=your_key_here")
    exit(1)

def voice_bot():
    """Complete voice bot pipeline with TTS"""
    
    print("\n" + "="*70)
    print("🏠 REAL ESTATE VOICE BOT")
    print("="*70)
    print("✅ Voice recognition: Active")
    print("✅ AI processing: Ready")
    print("✅ Text-to-speech: Enabled")
    print("\n💡 Speak naturally - I'm listening!")
    print("🛑 Press Ctrl+C to exit.\n")
    print("="*70)
    
    conversation_count = 0

    # 🔧 NEW: One-time greeting
    welcome = "Hello! I'm your real estate assistant. How can I help you?"
    print(f"\n🤖 {welcome}\n")
    speak_text(welcome)
    
    while True:
        try:
            conversation_count += 1
            print(f"\n{'='*70}")
            print(f"🎤 Conversation #{conversation_count}")
            print("="*70)
            
            # Step 1: STT - Get user speech
            print("\n👂 Listening...")
            user_text = get_user_speech()
            
            if not user_text:
                print("⚠️  No speech detected, try again")
                speak_text("I didn't catch that. Could you please repeat?")
                continue
            
            print(f"📝 You said: \"{user_text}\"")
            
            # Step 2: Process query (Intent + Routing + RAG)
            print("\n🤔 Processing your query...")
            response = process_user_query(user_text, grok_api_key=None, is_local_client=True)
            
            if not response:
                response = "I'm sorry, I couldn't process that request. Could you please rephrase?"
            
            # Step 3: Display response
            print(f"\n💬 Bot Response:")
            print(f"   {response}")
            
            # Step 4: TTS - Speak the response
            print("\n🔊 Speaking response...")
            speak_text(response)
            
            print(f"\n{'='*70}")
            print("✅ Ready for next question\n")
            
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("👋 Thank you for using Real Estate Voice Bot!")
            print(f"📊 Total conversations: {conversation_count}")
            print("="*70)
            speak_text("Goodbye! Have a great day!")
            break
            
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            error_message = "I encountered an error. Let me try again."
            print(f"🔊 Speaking: {error_message}")
            speak_text(error_message)
            continue


def test_components():
    """Test individual components before running full bot"""
    
    print("\n🧪 COMPONENT TEST MODE")
    print("="*70)
    
    # Test 1: TTS
    print("\n1️⃣  Testing TTS...")
    try:
        speak_text("Text to speech is working correctly.")
        print("✅ TTS: Working")
    except Exception as e:
        print(f"❌ TTS Failed: {e}")
        return False
    
    # Test 2: STT
    print("\n2️⃣  Testing STT (say something)...")
    try:
        text = get_user_speech()
        if text:
            print(f"✅ STT: Working (captured: '{text}')")
        else:
            print("⚠️  STT: No speech detected")
    except Exception as e:
        print(f"❌ STT Failed: {e}")
        return False
    
    # Test 3: Intent Processing
    print("\n3️⃣  Testing Intent Processing...")
    try:
        test_query = "What properties do you have?"
        response = process_user_query(test_query, grok_api_key=None, is_local_client=True, conversation_history=None) 
        # converstation_history set to None to avoid dependency on previous context --> harmless defaults
        if response:
            print(f"✅ Intent Processing: Working")
            print(f"   Test query: '{test_query}'")
            print(f"   Response: '{response[:100]}...'")
        else:
            print("⚠️  Intent Processing: No response")
    except Exception as e:
        print(f"❌ Intent Processing Failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("✅ All components tested!")
    print("="*70)
    return True


if __name__ == "__main__":
    import sys
    
    # Check if test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running in TEST mode...\n")
        if test_components():
            print("\n🚀 Ready to run full voice bot!")
            print("Run: python Main_completion.py")
    else:
        # Run full voice bot
        voice_bot()