import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 0.5rem;
        font-weight: 600;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model download link (Google Drive)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_PATH = "Modelenv.v1.h5"

# Download model if not present
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading AI model... (This happens only once)"):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return load_model(MODEL_PATH)

# Load model
try:
    model = download_and_load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    model_loaded = False

# Class labels with emojis and descriptions
class_info = {
    'Cloudy': {'emoji': '☁️', 'color': '#87CEEB', 'description': 'Cloud formations and overcast areas'},
    'Desert': {'emoji': '🏜️', 'color': '#DEB887', 'description': 'Arid and sandy terrain'},
    'Green_Area': {'emoji': '🌿', 'color': '#90EE90', 'description': 'Vegetation and forested regions'},
    'Water': {'emoji': '💧', 'color': '#4682B4', 'description': 'Water bodies and aquatic areas'}
}

class_names = list(class_info.keys())

# Header
st.markdown("""
<div class="main-header">
    <h1>🛰️ Satellite Image Classifier</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        Advanced AI-powered classification of satellite imagery
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Classification Categories")
    
    for class_name, info in class_info.items():
        with st.container():
            st.markdown(f"""
            <div style="padding: 1rem; margin: 0.5rem 0; background: {info['color']}20; 
                        border-radius: 8px; border-left: 4px solid {info['color']};">
                <h4 style="margin: 0; color: #333;">{info['emoji']} {class_name}</h4>
                <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.info("""
    This AI model analyzes satellite images and classifies them into four categories:
    - **Cloudy**: Weather patterns and cloud cover
    - **Desert**: Arid landscapes and sandy areas  
    - **Green Area**: Forests and vegetation
    - **Water**: Rivers, lakes, and oceans
    """)

# Main content
if not model_loaded:
    st.error("⚠️ Model could not be loaded. Please check your internet connection and try again.")
    st.stop()

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    
    # File uploader with custom styling
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a satellite image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a satellite image to classify it automatically"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        image_display = image.resize((400, 400))
        
        st.image(image_display, caption="📸 Uploaded Satellite Image", use_container_width=True)
        
        # Image info
        st.markdown(f"""
        **📋 Image Details:**
        - **Filename:** {uploaded_file.name}
        - **Size:** {uploaded_file.size:,} bytes
        - **Dimensions:** {image.size[0]} × {image.size[1]} pixels
        """)

with col2:
    if uploaded_file is not None:
        st.subheader("🔍 Classification Results")
        
        # Process image
        with st.spinner("🤖 Analyzing image..."):
            # Preprocess image
            image_processed = image.resize((256, 256))
            img_array = img_to_array(image_processed) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
        
        # Display main prediction
        info = class_info[predicted_class]
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: {info['color']}20; 
                    border-radius: 15px; border: 2px solid {info['color']};">
            <h2 style="margin: 0; color: #333;">{info['emoji']} {predicted_class}</h2>
            <p style="font-size: 1.5rem; margin: 1rem 0; color: #666;">
                Confidence: {confidence * 100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence meter
        st.markdown("### 📊 Confidence Level")
        confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
        st.progress(confidence)
        
        if confidence > 0.8:
            st.success(f"🎯 High confidence prediction ({confidence*100:.1f}%)")
        elif confidence > 0.6:
            st.warning(f"⚠️ Medium confidence prediction ({confidence*100:.1f}%)")
        else:
            st.error(f"❓ Low confidence prediction ({confidence*100:.1f}%)")
        
        # Detailed predictions chart
        st.markdown("### 📈 Detailed Predictions")
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'Category': class_names,
            'Probability': prediction * 100,
            'Emoji': [class_info[cat]['emoji'] for cat in class_names],
            'Color': [class_info[cat]['color'] for cat in class_names]
        })
        pred_df = pred_df.sort_values('Probability', ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(
            pred_df, 
            x='Probability', 
            y='Category',
            orientation='h',
            color='Color',
            color_discrete_map={row['Color']: row['Color'] for _, row in pred_df.iterrows()},
            title="Classification Probabilities"
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Probability (%)",
            yaxis_title="Category",
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_traces(
            texttemplate='%{x:.1f}%',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Probability: %{x:.1f}%<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction summary table
        st.markdown("### 📋 Prediction Summary")
        
        summary_df = pd.DataFrame({
            'Rank': range(1, len(class_names) + 1),
            'Category': [f"{class_info[cat]['emoji']} {cat}" for cat in pred_df['Category']],
            'Probability': [f"{prob:.1f}%" for prob in pred_df['Probability']],
            'Confidence': ['🔴 Low' if prob < 60 else '🟡 Medium' if prob < 80 else '🟢 High' 
                          for prob in pred_df['Probability']]
        })
        
        st.dataframe(
            summary_df.iloc[::-1],  # Reverse to show highest first
            use_container_width=True,
            hide_index=True
        )
        
        # Download results
        st.markdown("### 💾 Export Results")
        
        # Create results text
        results_text = f"""
Satellite Image Classification Results
=====================================

Image: {uploaded_file.name}
Predicted Class: {predicted_class}
Confidence: {confidence * 100:.2f}%

Detailed Predictions:
"""
        for i, (cat, prob) in enumerate(zip(class_names, prediction)):
            results_text += f"{i+1}. {cat}: {prob*100:.2f}%\n"
        
        st.download_button(
            label="📄 Download Results (TXT)",
            data=results_text,
            file_name=f"classification_results_{uploaded_file.name}.txt",
            mime="text/plain"
        )
    
    else:
        st.info("👈 Please upload a satellite image to start classification")
        
        # Show example or demo section
        st.markdown("### 🎯 How it works")
        st.markdown("""
        1. **Upload** a satellite image using the file uploader
        2. **AI Analysis** processes the image using deep learning
        3. **Classification** identifies the terrain type
        4. **Results** show confidence levels and detailed predictions
        """)
        
        st.markdown("### 🔧 Technical Details")
        st.markdown("""
        - **Model**: Convolutional Neural Network (CNN)
        - **Input Size**: 256×256 pixels
        - **Classes**: 4 terrain types
        - **Architecture**: Deep learning with TensorFlow/Keras
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>🛰️ Satellite Image Classifier | Powered by AI & Deep Learning</p>
</div>
""", unsafe_allow_html=True)
