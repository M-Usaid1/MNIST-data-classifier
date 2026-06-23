import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import os

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Title and description
st.title("🔢 MNIST Digit Recognition")
st.markdown("""
    ### Upload a handwritten digit (0-9)
    The model will predict which digit you've drawn!
""")

# API endpoint - update this with your deployed backend URL
# API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
# BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_URL = "https://Usaid2-mnist-backend.hf.space/predict"
BACKEND_URL = "https://Usaid2-mnist-backend.hf.space"

# Sidebar for instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload an image of a handwritten digit
    2. The image should be a single digit
    3. Click "Predict" to see the result
    
    **Note:** The model was trained on MNIST dataset (28x28 grayscale images)
    """)
    
    st.header("API Status")
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ Backend is running")
        else:
            st.error("❌ Backend not responding")
    except:
        st.error("❌ Cannot connect to backend")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload a grayscale or colored image of a handwritten digit"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("🔮 Predict Digit", type="primary"):
            with st.spinner("Analyzing digit..."):
                try:
                    # Prepare the image for API
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Send to API
                    files = {"file": ("image.png", img_byte_arr, "image/png")}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        with col2:
                            st.subheader("Prediction Result")
                            
                            # Display prediction with confidence
                            pred = result['prediction']
                            confidence = result['confidence'] * 100
                            
                            # Create a big display for the prediction
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                                <h1 style="font-size: 100px; margin: 0;">{pred}</h1>
                                <p style="font-size: 20px; margin: 0;">Confidence: {confidence:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show probability bar chart
                            probabilities = result['probabilities']
                            st.subheader("Probability Distribution")
                            
                            # Create a bar chart
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 4))
                            digits = list(range(10))
                            bars = ax.bar(digits, probabilities, color='skyblue')
                            ax.set_xlabel('Digit')
                            ax.set_ylabel('Probability')
                            ax.set_title('Prediction Probabilities')
                            ax.set_ylim([0, 1])
                            
                            # Highlight the predicted digit
                            bars[pred].set_color('orange')
                            
                            # Add value labels on bars
                            for bar, prob in zip(bars, probabilities):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{prob:.2f}', ha='center', va='bottom')
                            
                            st.pyplot(fig)
                            
                    else:
                        st.error(f"Error: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend server. Please make sure it's running.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

with col2:
    
        st.markdown("""
<div style="text-align: center;">
    <p>Model accuracy on test set: ~97.8%</p>
    <p style="font-size: 12px; color: gray;">Built with MNIST dataset • ANN with 2 hidden layers</p>
</div>
""", unsafe_allow_html=True)