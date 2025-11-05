"""
API server launcher for Symptom-Diagnosis-GPT.
This script provides a simple way to start the FastAPI server.
"""
import os
import sys
import uvicorn

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the app
try:
    from src.api import app
    from src.config import get_model_config
    
    def main():
        """Start the API server."""
        config = get_model_config()
        
        print("🚀 Starting Symptom-Diagnosis-GPT API Server")
        print("=" * 50)
        print(f"Host: {config.api_host}")
        print(f"Port: {config.api_port}")
        print(f"URL: http://{config.api_host}:{config.api_port}")
        print(f"Docs: http://{config.api_host}:{config.api_port}/docs")
        print("=" * 50)
        
        uvicorn.run(
            app,
            host=config.api_host,
            port=config.api_port,
            reload=False,  # Disable reload for better stability
            log_level="info"
        )

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the project root directory")
    print("💡 Install missing dependencies: pip install fastapi uvicorn torch tiktoken")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting API server: {e}")
    sys.exit(1)