"""
Streamlit app launcher for Symptom-Diagnosis-GPT.
This script starts the Streamlit web interface.
"""
import os
import sys
import subprocess
import time
import requests

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def check_api_health(url="http://localhost:8000/health", timeout=5):
    """Check if the API server is running."""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False

def start_streamlit():
    """Start the Streamlit app."""
    print("🌐 Starting Symptom-Diagnosis-GPT Web Interface")
    print("=" * 50)
    
    # Check if API is running
    if check_api_health():
        print("✅ API server is running")
    else:
        print("⚠️  API server not detected")
        print("💡 Start the API server first: python run_api.py")
        print("💡 Or run in demo mode (limited functionality)")
        print()
    
    print("🚀 Starting Streamlit...")
    print("📍 Web interface will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    print()
    
    try:
        # Start Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "src/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ]
        
        subprocess.run(cmd, cwd=project_root)
        
    except KeyboardInterrupt:
        print("\n🛑 Streamlit stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    start_streamlit()