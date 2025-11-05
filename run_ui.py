"""
Streamlit UI launcher for Symptom-Diagnosis-GPT.
This script provides a simple way to start the Streamlit web interface.
"""
import os
import sys
import subprocess

def main():
    """Start the Streamlit app."""
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.environ['PYTHONPATH'] = project_root
    
    print("🌐 Starting Symptom-Diagnosis-GPT Web Interface")
    print("=" * 50)
    print("URL: http://localhost:8501")
    print("=" * 50)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        print("💡 Make sure streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    main()