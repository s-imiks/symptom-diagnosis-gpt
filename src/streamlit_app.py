"""
Streamlit UI for Symptom-Diagnosis-GPT.
Provides a user-friendly web interface for symptom analysis.
"""
import streamlit as st
import requests
import json
import time
from typing import Dict, Any


# Page configuration
st.set_page_config(
    page_title="Tibu GPT",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_diagnosis(symptoms: str, max_length: int = 50, temperature: float = 1.0) -> Dict[str, Any]:
    """Send prediction request to API."""
    try:
        payload = {
            "symptoms": symptoms,
            "max_length": max_length,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", response.text)
            except:
                error_detail = response.text
            
            return {"error": f"API Error: {response.status_code} - {error_detail}"}
            
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server. Please start the API server first."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def predict_diagnosis_demo(symptoms: str) -> Dict[str, Any]:
    """Demo prediction without API (fallback mode)."""
    import random
    
    # Mock diagnoses based on common symptoms
    symptom_keywords = symptoms.lower()
    
    if any(word in symptom_keywords for word in ["fever", "temperature", "hot"]):
        diagnoses = ["flu", "viral infection", "common cold"]
    elif any(word in symptom_keywords for word in ["chest", "pain", "heart"]):
        diagnoses = ["chest strain", "anxiety", "heart palpitations"]
    elif any(word in symptom_keywords for word in ["headache", "head", "migraine"]):
        diagnoses = ["tension headache", "migraine", "stress headache"]
    elif any(word in symptom_keywords for word in ["stomach", "nausea", "vomit"]):
        diagnoses = ["gastroenteritis", "food poisoning", "stomach flu"]
    else:
        diagnoses = ["viral infection", "common cold", "stress-related symptoms"]
    
    diagnosis = random.choice(diagnoses)
    confidence = random.uniform(0.6, 0.85)
    
    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "input_text": f"Symptoms: {symptoms}\nDiagnosis:",
        "generated_text": f"Symptoms: {symptoms}\nDiagnosis: {diagnosis}"
    }


def get_model_info() -> Dict[str, Any]:
    """Get model information from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to get model info"}
    except:
        return {"error": "Cannot connect to API"}


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🏥 Tibu GPT")
    st.markdown("### AI-Powered Symptom Analysis and Diagnosis Prediction")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # API health check
    api_healthy = check_api_health()
    if api_healthy:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.markdown("**Start API server:**")
        st.sidebar.code("python run_api.py", language="bash")
        st.sidebar.markdown("**Or use demo mode** (limited functionality)")
    
    # Show demo mode status
    if not api_healthy:
        st.sidebar.info("🎭 Running in Demo Mode")
        st.sidebar.markdown("Demo mode provides basic functionality without the trained AI model.")
    
    # Model parameters
    st.sidebar.subheader("Generation Parameters")
    max_length = st.sidebar.slider(
        "Max Generation Length",
        min_value=10,
        max_value=200,
        value=50,
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Sampling temperature (higher = more creative)"
    )
    
    # Model info
    if api_healthy:
        with st.sidebar.expander("Model Information"):
            model_info = get_model_info()
            if "error" not in model_info:
                st.json(model_info)
            else:
                st.error(model_info["error"])
    
    # Main interface
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Describe Your Symptoms")
        
        # Example symptoms
        examples = [
            "Select an example...",
            "I have fever, cough, and sore throat",
            "I experience chest pain and shortness of breath",
            "I have persistent headache and nausea",
            "I feel fatigue, joint pain, and have a rash",
            "I have stomach pain, diarrhea, and fever"
        ]
        
        example_choice = st.selectbox("Or choose an example:", examples)
        
        # Symptom input
        if example_choice != examples[0]:
            default_text = example_choice
        else:
            default_text = ""
        
        symptoms_text = st.text_area(
            "Enter your symptoms:",
            value=default_text,
            height=120,
            placeholder="Describe your symptoms in detail... (e.g., 'I have fever, headache, and body aches')",
            help="Be as specific as possible about your symptoms, their duration, and severity."
        )
        
        # Predict button
        predict_button = st.button(
            "🔍 Analyze Symptoms",
            type="primary",
            disabled=not api_healthy or not symptoms_text.strip()
        )
        
        # Disclaimer
        st.warning(
            "⚠️ **Medical Disclaimer**: This tool is for educational purposes only. "
            "Always consult with a qualified healthcare professional for medical advice. "
            "Do not use this as a substitute for professional medical diagnosis or treatment."
        )
    
    with col2:
        st.subheader("Diagnosis Prediction")
        
        if predict_button and symptoms_text.strip():
            if api_healthy:
                # Use real API
                with st.spinner("Analyzing symptoms..."):
                    result = predict_diagnosis(symptoms_text, max_length, temperature)
            else:
                # Use demo mode
                st.warning("⚠️ API not available - using demo mode")
                with st.spinner("Generating demo prediction..."):
                    result = predict_diagnosis_demo(symptoms_text)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
                
                # Offer demo mode as fallback
                if api_healthy:  # If API was supposed to work but failed
                    st.info("Trying demo mode...")
                    result = predict_diagnosis_demo(symptoms_text)
                    if "error" not in result:
                        st.warning("Using demo prediction (API failed)")
            
            if "error" not in result:
                # Display results
                if api_healthy:
                    st.success("Analysis Complete!")
                else:
                    st.info("Demo Analysis Complete!")
                
                # Main diagnosis
                st.markdown("### 🎯 Predicted Diagnosis")
                st.markdown(f"**{result['diagnosis']}**")
                
                if not api_healthy:
                    st.caption("🎭 This is a demo prediction. Start the API server for AI-powered results.")
                
                # Confidence score
                confidence = result.get('confidence', 0.0)
                st.markdown("### 📊 Confidence Score")
                st.progress(confidence)
                st.markdown(f"Confidence: {confidence:.1%}")
                
                # Additional information
                with st.expander("View Details"):
                    st.markdown("**Processed Input:**")
                    st.code(result.get('input_text', ''), language="text")
                    
                    st.markdown("**Generated Text:**")
                    st.code(result.get('generated_text', ''), language="text")
        
        elif not api_healthy:
            st.info("API server not available. You can still use demo mode by entering symptoms above.")
            st.markdown("**To enable full AI functionality:**")
            st.code("python run_api.py", language="bash")
        else:
            st.info("Enter your symptoms above to get a diagnosis prediction.")
    
    # Additional information
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔬 About This Tool")
        st.markdown(
            "This AI model uses a lightweight GPT-like transformer "
            "trained on symptom-diagnosis pairs to provide preliminary "
            "medical insights."
        )
    
    with col2:
        st.markdown("### 🚀 Distributed Training")
        st.markdown(
            "The model was trained using distributed computing with Ray, "
            "allowing for efficient training across multiple nodes and "
            "faster convergence."
        )
    
    with col3:
        st.markdown("### 💡 How to Use")
        st.markdown(
            "1. Enter your symptoms in detail\n"
            "2. Adjust generation parameters if needed\n"
            "3. Click 'Analyze Symptoms'\n"
            "4. Review the prediction and confidence score"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with Streamlit, FastAPI, PyTorch, and Ray • "
        "Symptom-Diagnosis-GPT v1.0"
    )


def run_demo():
    """Run a demo version without API dependency."""
    st.title("🏥 Tibu GPT - Demo Mode")
    st.warning("Running in demo mode. API server not available.")
    
    symptoms = st.text_area("Enter symptoms:", placeholder="Describe your symptoms...")
    
    if st.button("Analyze (Demo)"):
        if symptoms:
            # Mock response
            import random
            mock_diagnoses = [
                "common cold", "flu", "migraine", "gastroenteritis",
                "allergic reaction", "muscle strain", "viral infection"
            ]
            
            diagnosis = random.choice(mock_diagnoses)
            confidence = random.uniform(0.6, 0.9)
            
            st.success(f"**Demo Diagnosis:** {diagnosis}")
            st.progress(confidence)
            st.info("This is a mock response. Start the API for real predictions.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Falling back to demo mode...")
        run_demo()