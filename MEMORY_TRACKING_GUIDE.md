# OATS Chatbot Memory Tracking System

## 🚀 Quick Setup & Usage Guide

### What This Does
- **Real-time memory monitoring** of your OATS chatbot
- **Non-intrusive tracking** that won't affect chatbot performance
- **Live dashboard** to see memory changes as you chat
- **Detailed logging** of conversations and data caching

### Files Added
1. `memory_tracker.py` - Core tracking system
2. `demo_memory_tracking.py` - Standalone demo script
3. `templates/memory_monitor.html` - Web dashboard
4. Memory tracking routes added to `app.py`

### How to Use

#### Option 1: Live Web Dashboard (Recommended)
1. **Start your chatbot normally:**
   ```bash
   python app.py
   ```

2. **Open the memory monitor dashboard:**
   - Go to: `http://localhost:5000/memory-monitor`
   - You'll see a live dashboard showing:
     - Number of conversations stored
     - Data cache size
     - Memory usage
     - Latest interactions

3. **Chat with your bot and watch live changes:**
   - Open another tab: `http://localhost:5000`
   - Ask questions like "show me all jobs" or "find candidates"
   - Go back to memory monitor tab to see real-time updates

#### Option 2: Standalone Demo Script
```bash
python demo_memory_tracking.py
```
Choose option 1 for live monitoring in terminal.

#### Option 3: Check Log Files
```bash
python demo_memory_tracking.py
```
Choose option 2 to view generated JSON log files.

### What You'll See

#### Memory Dashboard Shows:
- **💬 Conversations:** Number of chat interactions stored
- **💾 Cached Data:** Number of API responses cached
- **🔧 Memory Usage:** RAM usage in MB
- **🔄 Latest Interaction:** Most recent chat with endpoints used

#### Log Files Generated:
- `live_memory.json` - Current memory state (updates every 2 seconds)
- `memory_logs.json` - Detailed history (keeps last 50 snapshots)

### Example Memory Tracking

When you ask: *"Show me all Python developers"*

You'll see:
```json
{
  "conversation_count": 1,
  "cache_count": 1,
  "latest_interaction": {
    "id": 1,
    "query": "Show me all Python developers",
    "endpoints_used": ["candidate_search"],
    "timestamp": "2025-01-XX 10:30:45"
  }
}
```

### Performance Impact
- **Minimal:** Background thread updates every 2-3 seconds
- **Safe:** Silent failure if tracking fails
- **Optional:** Can be disabled by removing `memory_tracker.py`

### Troubleshooting

#### If Memory Tracking Doesn't Start:
```bash
# Check if psutil is installed
pip install psutil

# Run the chatbot and check console for:
# ✅ Memory tracking enabled
# ✅ Memory tracking initialized successfully
```

#### If Dashboard Shows "Loading...":
1. Ensure chatbot is running (`python app.py`)
2. Check that `memory_tracker.py` exists
3. Look for `live_memory.json` file creation

#### To Disable Memory Tracking:
Simply rename or delete `memory_tracker.py` - the chatbot will continue working normally.

### Advanced Usage

#### Access Memory Data Programmatically:
```python
# In your own scripts
from memory_tracker import get_live_memory
memory_state = get_live_memory()
print(memory_state)
```

#### Monitor Specific Interactions:
```bash
# Watch the log files in real-time
tail -f memory_logs.json
```

### URLs for Monitoring
- **Main Chatbot:** `http://localhost:5000`
- **Memory Dashboard:** `http://localhost:5000/memory-monitor`
- **Live Memory JSON:** `http://localhost:5000/live_memory.json`
- **Detailed Logs JSON:** `http://localhost:5000/memory_logs.json`

---

## 🎯 Testing the System

1. **Start the chatbot:** `python app.py`
2. **Open memory dashboard:** Browser → `http://localhost:5000/memory-monitor`
3. **Chat with bot:** Another tab → `http://localhost:5000`
4. **Ask questions like:**
   - "Show me all jobs"
   - "Find Python developers"
   - "What's the dashboard data?"
5. **Watch memory changes in real-time on the dashboard**

The system will show you exactly how conversations are stored, what data is cached, and how memory usage changes as you interact with the chatbot!
