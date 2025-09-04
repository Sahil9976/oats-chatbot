#!/usr/bin/env python3
"""
Simple runner script for the OATS Flask Chatbot
"""

import sys
import subprocess
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def main():
    """Main runner function"""
    print("🚀 OATS Flask Chatbot Runner")
    print("=" * 40)
    
    print("\n🌐 Starting Flask application...")
    print("📍 The app will be available at: http://localhost:5000")
    print("🔐 Auto-login enabled - no manual login required")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 40)
    
    # Import and run the Flask app
    try:
        from app import main as app_main
        app_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the flask directory and requirements are installed")
        print("📦 Run: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n\n👋 Flask app stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()
