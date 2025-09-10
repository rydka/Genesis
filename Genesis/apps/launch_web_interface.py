#!/usr/bin/env python3
"""
Launch Fantasy RAG Web Interface

This script launches the web interface for the Fantasy RAG system.
"""

import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Launch the web interface"""
    
    print("🌐 Fantasy RAG Web Interface Launcher")
    print("=" * 50)
    
    try:
        # Import the Flask app
        from fantasy_rag.web.app import app, initialize_components
        
        print("🔧 Initializing Fantasy RAG components...")
        
        if not initialize_components():
            print("❌ Failed to initialize components")
            print("\n🔍 Troubleshooting:")
            print("   1. Make sure the Fantasy RAG database exists")
            print("   2. Run: python test_single_asset_pipeline.py")
            print("   3. Check that ChromaDB is properly installed")
            return 1
        
        print("✅ Components initialized successfully")
        print("\n🚀 Starting web server...")
        print("   URL: http://localhost:5000")
        print("   Press Ctrl+C to stop")
        print("\n📋 Available pages:")
        print("   • http://localhost:5000          - Recherche sémantique")
        print("   • http://localhost:5000/gallery  - Galerie d'assets")
        print("   • http://localhost:5000/api/stats - API statistiques")
        
        # Start the Flask app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("\n🔍 Troubleshooting:")
        print("   1. Make sure you're in the correct directory")
        print("   2. Install missing packages: pip install flask chromadb")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    exit(main())
