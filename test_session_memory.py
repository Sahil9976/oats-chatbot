"""
Test script to verify session-based memory monitoring
"""

import requests
import json
import time

# Base URL - adjust if running on different port
BASE_URL = "http://localhost:5000"

def test_session_memory():
    """Test the session-based memory monitor"""
    print("🧪 Testing Session-Based Memory Monitor")
    print("="*50)
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    # Test 1: Check initial state
    print("\n1️⃣ Checking initial chat history...")
    response = session.get(f"{BASE_URL}/api/chat-history")
    data = response.json()
    print(f"   Initial messages: {data.get('total_chats', 0)}")
    print(f"   Session-based: {data.get('session_based', False)}")
    print(f"   Max messages: {data.get('max_messages', 'N/A')}")
    
    # Test 2: Send a few messages
    print("\n2️⃣ Sending test messages...")
    test_messages = [
        "What is OATS?",
        "Tell me about your features",
        "How does session memory work?"
    ]
    
    for i, msg in enumerate(test_messages, 1):
        print(f"   Sending message {i}: {msg}")
        chat_response = session.post(f"{BASE_URL}/chat", 
                                   json={"query": msg})
        if chat_response.status_code == 200:
            print(f"   ✅ Message {i} sent successfully")
        else:
            print(f"   ❌ Error sending message {i}")
        time.sleep(1)  # Small delay between messages
    
    # Test 3: Check updated history
    print("\n3️⃣ Checking updated chat history...")
    response = session.get(f"{BASE_URL}/api/chat-history")
    data = response.json()
    print(f"   Current messages: {data.get('total_chats', 0)}")
    print(f"   Chat history length: {len(data.get('chat_history', []))}")
    
    # Test 4: Test with a different session
    print("\n4️⃣ Testing with different session...")
    new_session = requests.Session()
    response = new_session.get(f"{BASE_URL}/api/chat-history")
    data = response.json()
    print(f"   New session messages: {data.get('total_chats', 0)}")
    print(f"   ✅ Sessions are independent!" if data.get('total_chats', 0) == 0 else "   ❌ Sessions not independent")
    
    # Test 5: Open memory monitor
    print("\n5️⃣ Memory Monitor Info:")
    print(f"   Open http://localhost:5000/memory-monitor to see live updates")
    print(f"   Each session will have its own chat history")
    print(f"   Maximum 30 messages per session (rolling window)")
    
    print("\n" + "="*50)
    print("✅ Session-based memory monitor test complete!")
    print("\nNOTE: To test rolling window:")
    print("1. Send more than 30 messages in one session")
    print("2. Watch as oldest messages are automatically removed")
    print("3. The memory monitor will show 'FULL' status when at 30 messages")

if __name__ == "__main__":
    try:
        test_session_memory()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("Make sure the OATS chatbot is running (python app.py)")
