#!/usr/bin/env python3
"""
Start Ollama server automatically
"""

import subprocess
import time
import os
import sys
from pathlib import Path
import requests

def find_ollama_executable():
    """Find Ollama executable in common locations"""
    
    # Common Ollama installation paths on Windows
    possible_paths = [
        Path(os.environ.get('USERPROFILE', '')) / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
        Path("C:/Program Files (x86)/Ollama/ollama.exe"),
        Path(os.environ.get('LOCALAPPDATA', '')) / "Programs" / "Ollama" / "ollama.exe"
    ]
    
    # Check if ollama is already in PATH
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            return "ollama"  # Already in PATH
    except:
        pass
    
    # Search in common locations
    for path in possible_paths:
        if path.exists():
            print(f"✅ Found Ollama at: {path}")
            return str(path)
    
    return None

def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except:
        return False

def start_ollama_server():
    """Start Ollama server"""
    
    print("🚀 Starting Ollama Server")
    print("=" * 40)
    
    # Find Ollama executable
    ollama_cmd = find_ollama_executable()
    
    if not ollama_cmd:
        print("❌ Ollama executable not found")
        print("\n💡 Please install Ollama from: https://ollama.com/download")
        return False
    
    print(f"📍 Using Ollama: {ollama_cmd}")
    
    # Check if already running
    if check_ollama_running():
        print("✅ Ollama server is already running")
        return True
    
    print("🔄 Starting Ollama server...")
    
    try:
        # Start Ollama serve in background
        process = subprocess.Popen(
            [ollama_cmd, "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        
        print(f"🔄 Ollama server starting (PID: {process.pid})")
        print("⏳ Waiting for server to be ready...")
        
        # Wait for server to start (max 30 seconds)
        for i in range(30):
            time.sleep(1)
            if check_ollama_running():
                print("✅ Ollama server is ready!")
                return True
            print(f"   Waiting... ({i+1}/30)")
        
        print("❌ Ollama server failed to start within 30 seconds")
        return False
        
    except Exception as e:
        print(f"❌ Error starting Ollama: {str(e)}")
        return False

def list_available_models():
    """List available models"""
    
    if not check_ollama_running():
        print("❌ Ollama server not running")
        return
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            print(f"\n📦 Available Models ({len(models)}):")
            if models:
                for model in models:
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0) / (1024**3)  # Convert to GB
                    print(f"   📦 {name} ({size:.1f} GB)")
            else:
                print("   No models installed")
                print("   Install a vision model: python setup_ollama_vision.py")
        else:
            print(f"❌ Error listing models: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Main function"""
    
    print("🦙 Ollama Server Starter")
    print("=" * 50)
    
    # Start server
    if start_ollama_server():
        print("\n🎉 Ollama server is running!")
        
        # List models
        list_available_models()
        
        print("\n🚀 Next Steps:")
        print("   1. Install vision model: python setup_ollama_vision.py")
        print("   2. Test descriptions: python test_description_generation.py")
        print("   3. Keep this terminal open to keep Ollama running")
        
        # Keep script running to show server is active
        try:
            print("\n⏸️  Press Ctrl+C to stop monitoring...")
            while True:
                time.sleep(10)
                if not check_ollama_running():
                    print("⚠️ Ollama server stopped")
                    break
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped. Ollama server continues running.")
        
        return 0
    else:
        print("\n❌ Failed to start Ollama server")
        print("\n💡 Manual start:")
        print("   1. Open Command Prompt as Administrator")
        print("   2. Run: ollama serve")
        return 1

if __name__ == "__main__":
    exit(main())
