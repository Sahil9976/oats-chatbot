"""
OATS Chatbot Memory Tracker - Real-time Memory Monitoring
- Tracks conversation history, data cache, and user context
- Non-intrusive monitoring with minimal performance impact
- Provides live updates and detailed memory analysis
- Separate from main chatbot to avoid conflicts
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Any
import psutil

class MemoryTracker:
    """Non-intrusive memory tracking system for OATS chatbot"""
    
    def __init__(self, track_file="memory_logs.json", live_file="live_memory.json"):
        self.track_file = track_file
        self.live_file = live_file
        self.monitoring = False
        self.chatbot_instance = None
        
        # Initialize tracking files
        self._initialize_files()
        
    def _initialize_files(self):
        """Initialize tracking files with empty structure"""
        initial_data = {
            "tracking_started": datetime.now().isoformat(),
            "total_interactions": 0,
            "memory_snapshots": [],
            "performance_metrics": []
        }
        
        # Create tracking log file
        with open(self.track_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
            
        # Create live status file
        live_data = {
            "status": "initialized",
            "last_update": datetime.now().isoformat(),
            "current_memory": {}
        }
        
        with open(self.live_file, 'w') as f:
            json.dump(live_data, f, indent=2)
            
        print(f"✅ Memory tracker initialized:")
        print(f"   📊 Detailed logs: {self.track_file}")
        print(f"   🔴 Live status: {self.live_file}")
    
    def attach_to_chatbot(self, chatbot_instance):
        """Attach to chatbot instance for monitoring"""
        self.chatbot_instance = chatbot_instance
        print(f"🔗 Attached to chatbot instance: {type(chatbot_instance).__name__}")
        
        # Start monitoring thread
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        print("🚀 Memory monitoring started in background thread")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                if self.chatbot_instance:
                    self._capture_memory_snapshot()
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(5)
    
    def _capture_memory_snapshot(self):
        """Capture current memory state"""
        try:
            if not hasattr(self.chatbot_instance, 'conversation_history'):
                return
                
            # Get current memory state
            memory_state = self._extract_memory_state()
            
            # Update live file
            self._update_live_file(memory_state)
            
            # Log detailed snapshot
            self._log_memory_snapshot(memory_state)
            
        except Exception as e:
            print(f"❌ Error capturing memory: {e}")
    
    def _extract_memory_state(self) -> Dict:
        """Extract comprehensive memory state from chatbot"""
        chatbot = self.chatbot_instance
        
        # Basic memory info
        conversation_count = len(getattr(chatbot, 'conversation_history', []))
        cache_count = len(getattr(chatbot, 'conversation_data_cache', {}))
        
        # User context
        user_context = getattr(chatbot, 'user_context', {})
        
        # Latest interactions (last 3)
        recent_interactions = []
        if hasattr(chatbot, 'conversation_history') and chatbot.conversation_history:
            recent_interactions = chatbot.conversation_history[-3:]
        
        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Data cache details
        cache_details = {}
        if hasattr(chatbot, 'conversation_data_cache'):
            for cache_id, cache_data in chatbot.conversation_data_cache.items():
                cache_details[str(cache_id)] = {
                    'query': cache_data.get('query', 'N/A')[:50] + '...',
                    'timestamp': cache_data.get('timestamp', 0),
                    'data_types': list(cache_data.get('data', {}).keys()),
                    'endpoints': cache_data.get('endpoints', [])
                }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'conversation_count': conversation_count,
            'cache_count': cache_count,
            'memory_usage_mb': round(memory_mb, 2),
            'user_context': user_context,
            'recent_interactions': recent_interactions,
            'cache_details': cache_details,
            'memory_limits': {
                'max_history': getattr(chatbot, 'max_conversation_history', 'N/A'),
                'current_history': conversation_count,
                'cache_size': cache_count
            }
        }
    
    def _update_live_file(self, memory_state: Dict):
        """Update live monitoring file"""
        live_data = {
            'status': 'monitoring',
            'last_update': memory_state['timestamp'],
            'current_memory': {
                'conversations': memory_state['conversation_count'],
                'cached_data': memory_state['cache_count'],
                'memory_usage_mb': memory_state['memory_usage_mb'],
                'latest_query': None
            }
        }
        
        # Add latest query if available
        if memory_state['recent_interactions']:
            latest = memory_state['recent_interactions'][-1]
            live_data['current_memory']['latest_query'] = {
                'id': latest.get('id', 'N/A'),
                'query': latest.get('user_query', 'N/A')[:100] + '...',
                'timestamp': latest.get('timestamp', 'N/A'),
                'endpoints_used': latest.get('endpoints_used', [])
            }
        
        try:
            with open(self.live_file, 'w') as f:
                json.dump(live_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error updating live file: {e}")
    
    def _log_memory_snapshot(self, memory_state: Dict):
        """Log detailed memory snapshot"""
        try:
            # Read existing log
            with open(self.track_file, 'r') as f:
                log_data = json.load(f)
            
            # Add new snapshot
            log_data['memory_snapshots'].append(memory_state)
            log_data['total_interactions'] = memory_state['conversation_count']
            
            # Keep only last 50 snapshots to manage file size
            if len(log_data['memory_snapshots']) > 50:
                log_data['memory_snapshots'] = log_data['memory_snapshots'][-50:]
            
            # Write back
            with open(self.track_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error logging snapshot: {e}")
    
    def log_interaction(self, user_query: str, response_data: Dict = None, 
                       endpoints_used: List[str] = None, ai_response: str = None):
        """Log specific interaction details"""
        interaction_log = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'endpoints_used': endpoints_used or [],
            'response_data_keys': list(response_data.keys()) if response_data else [],
            'ai_response_length': len(ai_response) if ai_response else 0,
            'memory_state': self._extract_memory_state() if self.chatbot_instance else {}
        }
        
        print(f"📝 Logged interaction: {user_query[:50]}...")
        return interaction_log
    
    def get_memory_summary(self) -> Dict:
        """Get current memory summary"""
        if not self.chatbot_instance:
            return {'error': 'No chatbot attached'}
        
        return self._extract_memory_state()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        print("🛑 Memory monitoring stopped")
    
    def display_live_stats(self):
        """Display current memory statistics"""
        if not self.chatbot_instance:
            print("❌ No chatbot attached")
            return
        
        memory_state = self._extract_memory_state()
        
        print("\n" + "="*60)
        print("🧠 OATS CHATBOT MEMORY STATUS")
        print("="*60)
        print(f"📊 Conversations Stored: {memory_state['conversation_count']}")
        print(f"💾 Data Cache Size: {memory_state['cache_count']}")
        print(f"🔧 Memory Usage: {memory_state['memory_usage_mb']} MB")
        print(f"⏰ Last Update: {memory_state['timestamp']}")
        
        if memory_state['recent_interactions']:
            print(f"\n🔄 Recent Interactions:")
            for i, interaction in enumerate(memory_state['recent_interactions'][-3:], 1):
                print(f"  {i}. {interaction.get('user_query', 'N/A')[:60]}...")
        
        print("="*60)


# Global memory tracker instance
memory_tracker = MemoryTracker()

def initialize_memory_tracking(chatbot_instance):
    """Initialize memory tracking for chatbot"""
    global memory_tracker
    memory_tracker.attach_to_chatbot(chatbot_instance)
    return memory_tracker

def log_interaction(user_query: str, response_data: Dict = None, 
                   endpoints_used: List[str] = None, ai_response: str = None):
    """Quick logging function"""
    global memory_tracker
    return memory_tracker.log_interaction(user_query, response_data, endpoints_used, ai_response)

def get_live_memory():
    """Get current memory state"""
    global memory_tracker
    return memory_tracker.get_memory_summary()

def show_memory_stats():
    """Show current memory statistics"""
    global memory_tracker
    memory_tracker.display_live_stats()

if __name__ == "__main__":
    print("🔍 OATS Memory Tracker - Standalone Mode")
    print("This module should be imported and used with the chatbot.")
    print("Example usage:")
    print("  from memory_tracker import initialize_memory_tracking")
    print("  tracker = initialize_memory_tracking(chatbot_instance)")
