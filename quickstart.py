"""
Quick Start Guide for Symptom-Diagnosis-GPT.
This script provides step-by-step instructions to get the system running.
"""
import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def print_header():
    """Print header."""
    print("🏥 Symptom-Diagnosis-GPT Quick Start")
    print("=" * 50)

def check_dependencies():
    """Check if required packages are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        "torch": "PyTorch",
        "tiktoken": "Tiktoken", 
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "streamlit": "Streamlit",
        "requests": "Requests"
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("💡 Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies available")
    return True

def check_data():
    """Check if data exists."""
    print("\n📊 Checking data...")
    
    data_dir = Path("data/processed")
    if data_dir.exists() and list(data_dir.glob("*.pkl")):
        print("✅ Dataset exists")
        return True
    else:
        print("❌ Dataset not found")
        return False

def check_model():
    """Check if trained model exists."""
    print("\n🤖 Checking model...")
    
    model_path = Path("data/processed/model.pt")
    if model_path.exists():
        print("✅ Trained model exists")
        return True
    else:
        print("❌ Trained model not found")
        return False

def create_data():
    """Create dataset."""
    print("\n📊 Creating dataset...")
    try:
        result = subprocess.run([
            sys.executable, "simple_train.py", "--check"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ System check passed")
        else:
            print("❌ System check failed")
            print(result.stdout)
            return False
        
        # Create data
        result = subprocess.run([
            sys.executable, "-m", "src.prepare_data", "--num-samples", "200"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Dataset created")
            return True
        else:
            print("❌ Dataset creation failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error creating data: {e}")
        return False

def train_model():
    """Train the model."""
    print("\n🎯 Training model...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.train", 
            "--epochs", "3", 
            "--batch-size", "16"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Model trained")
            return True
        else:
            print("❌ Training failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def check_api(port=8000):
    """Check if API is running."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_api():
    """Start the API server."""
    print("\n🚀 Starting API server...")
    
    if check_api():
        print("✅ API server already running")
        return True
    
    try:
        print("   Starting API in background...")
        process = subprocess.Popen([
            sys.executable, "run_api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        for i in range(10):
            time.sleep(2)
            if check_api():
                print("✅ API server started")
                return True
            print(f"   Waiting for API... ({i+1}/10)")
        
        print("❌ API server failed to start")
        return False
        
    except Exception as e:
        print(f"❌ Error starting API: {e}")
        return False

def start_streamlit():
    """Start Streamlit."""
    print("\n🌐 Starting web interface...")
    
    try:
        print("   Opening Streamlit (this will open a new window)...")
        subprocess.run([
            sys.executable, "run_streamlit.py"
        ])
        
    except KeyboardInterrupt:
        print("\n✅ Streamlit stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

def quick_setup():
    """Quick setup routine."""
    print_header()
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check/create data
    if not check_data():
        if not create_data():
            print("💡 You can still use demo mode")
    
    # Check/train model
    if not check_model():
        print("💡 Training a quick model (this may take a few minutes)...")
        if not train_model():
            print("💡 You can still use demo mode")
    
    print("\n🎉 Setup complete!")
    return True

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Symptom-Diagnosis-GPT Quick Start")
    parser.add_argument("--setup", action="store_true", help="Run complete setup")
    parser.add_argument("--check", action="store_true", help="Check system status")
    parser.add_argument("--api", action="store_true", help="Start API server only")
    parser.add_argument("--web", action="store_true", help="Start web interface only")
    
    args = parser.parse_args()
    
    if args.setup:
        if quick_setup():
            print("\n🚀 Starting services...")
            start_api()
            start_streamlit()
    elif args.check:
        print_header()
        check_dependencies()
        check_data()
        check_model()
        print(f"\nAPI running: {'✅' if check_api() else '❌'}")
    elif args.api:
        start_api()
    elif args.web:
        start_streamlit()
    else:
        print_header()
        print("Available commands:")
        print("  --setup  : Complete setup and start")
        print("  --check  : Check system status")
        print("  --api    : Start API server")
        print("  --web    : Start web interface")
        print("\nQuick start:")
        print("  python quickstart.py --setup")

if __name__ == "__main__":
    main()