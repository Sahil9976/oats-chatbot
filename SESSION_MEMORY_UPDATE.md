# Session-Based Memory Monitor Update

## Overview
The OATS Chatbot Memory Monitor has been updated to be session-based and dynamic, with a rolling window of 30 messages per session.

## Changes Made

### 1. Session-Based Chat History Management
- Added `get_session_chat_history()` function to retrieve session-specific chat history
- Added `add_to_session_chat_history()` function to add new messages with automatic rolling window management
- Each user session now maintains its own independent chat history

### 2. Rolling Window Implementation
- Maximum of 30 messages per session
- When the 31st message arrives, the 1st message is automatically removed
- When the 32nd message arrives, the 2nd message is removed, and so on
- This ensures memory efficiency while maintaining recent context

### 3. Updated API Endpoints
- `/api/chat-history` now returns session-specific chat history
- Response includes:
  - `session_based: true` flag
  - `max_messages: 30` indicator
  - Only the messages from the current session

### 4. Enhanced Memory Monitor UI
- Updated header to show "Live Session-Based Memory Tracking Dashboard"
- New statistics cards showing:
  - **Session Messages**: Current count out of 30 max
  - **Session Status**: Shows "ACTIVE" or "FULL" with remaining slots
  - **Memory Usage**: Dynamic calculation based on session messages
  - **Last Update**: Real-time timestamp
- Chat history display shows:
  - Message numbers for each chat
  - Rolling window warning when at capacity
  - Last 10 messages with proper numbering

## How It Works

1. **Session Initialization**: When a user starts chatting, a new session is created with an empty chat history
2. **Message Storage**: Each message is stored in both:
   - The global chat history file (existing behavior)
   - The session-specific history (new behavior)
3. **Rolling Window**: Session history automatically maintains only the last 30 messages
4. **Independent Sessions**: Each browser session has its own chat history
5. **Live Updates**: Memory monitor refreshes every 3 seconds showing current session data

## Testing

Run the test script to verify functionality:
```bash
python test_session_memory.py
```

## Benefits

1. **Privacy**: Each session's chat history is independent
2. **Performance**: Limited to 30 messages per session prevents memory bloat
3. **User Experience**: Clear visual indicators of session status and capacity
4. **Backward Compatibility**: Existing global chat history file is still maintained

## Visual Indicators

- **Green "ACTIVE" Status**: Session has room for more messages
- **Orange "FULL" Status**: Session at 30 message capacity
- **Red Warning**: Displayed when rolling window is active
- **Message Numbers**: Each message shows its position in the session

## Important Notes

- Session data is stored in Flask's filesystem session storage
- Sessions expire after 2 hours of inactivity (configurable)
- Refreshing the browser maintains the same session
- Opening in a new browser/incognito creates a new session
