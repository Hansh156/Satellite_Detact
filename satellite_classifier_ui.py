import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="🛰️ Satellite Image Classifier",
    page_icon="🌍",
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
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Model download link (Google Drive)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_PATH = "Modelenv.v1.h5"

# Enhanced download and load model function
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🔄 Downloading AI model... (this may take a moment)"):
            response = requests.get(MODEL_URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(MODEL_PATH, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            status_text.text(f"Downloaded {downloaded // (1024*1024)} MB of {total_size // (1024*1024)} MB")
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Model downloaded successfully!")
        time.sleep(1)
    
    return load_model(MODEL_PATH)

# Class labels with emojis and descriptions
class_info = {
    'Cloudy': {
        'emoji': '☁️',
        'color': '#87CEEB',
        'description': 'Cloud formations and atmospheric conditions'
    },
    'Desert': {
        'emoji': '🏜️',
        'color': '#F4A460',
        'description': 'Arid landscapes and sandy terrain'
    },
    'Green_Area': {
        'emoji': '🌿',
        'color': '#90EE90',
        'description': 'Vegetation, forests, and agricultural areas'
    },
    'Water': {
        'emoji': '💧',
        'color': '#4682B4',
        'description': 'Water bodies, rivers, and coastal areas'
    }
}

class_names = list(class_info.keys())

# Sidebar
with st.sidebar:
    st.markdown("## 🛰️ **Satellite Image Classifier**")
    st.markdown("---")
    
    st.markdown("### 📊 **Model Information**")
    st.info("**Model Type**: Convolutional Neural Network")
    st.info("**Input Size**: 256x256 pixels")
    st.info("**Classes**: 4 terrain types")
    
    st.markdown("### 🎯 **Classification Categories**")
    for class_name, info in class_info.items():
        st.markdown(f"**{info['emoji']} {class_name}**")
        st.caption(info['description'])
    
    st.markdown("---")
    st.markdown("### 💡 **Tips for Best Results**")
    st.markdown("• Use clear, high-resolution images")
    st.markdown("• Ensure good lighting conditions")
    st.markdown("• Avoid heavily processed images")
    st.markdown("• Satellite/aerial view works best")

# Main content
st.markdown("""
<div class="main-header">
    <h1>🛰️ Satellite Image Classifier</h1>
    <p>Advanced AI-powered terrain classification from satellite imagery</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("🔄 Loading AI model..."):
    model = download_and_load_model()

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 **Upload Your Image**")
    uploaded_file = st.file_uploader(
        "Choose a satellite image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a satellite or aerial image for terrain classification"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((256, 256))
        
        st.markdown("### 🖼️ **Uploaded Image**")
        st.image(image, caption="Image ready for classification", use_container_width=True)
        
        # Image info
        st.markdown("### 📋 **Image Details**")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("File Size", f"{uploaded_file.size // 1024} KB")
        with col_info2:
            st.metric("Dimensions", "256×256 px")

with col2:
    if uploaded_file is not None:
        st.markdown("### 🔍 **Classification Results**")
        
        # Process image
        with st.spinner("🧠 Analyzing image..."):
            img_array = img_to_array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            # Progress bar for dramatic effect
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            progress_bar.empty()
        
        # Display results
        class_emoji = class_info[predicted_class]['emoji']
        class_color = class_info[predicted_class]['color']
        
        st.markdown(f"### 🎯 **Prediction: {class_emoji} {predicted_class}**")
        
        # Confidence meter
        st.markdown("### 📊 **Confidence Score**")
        confidence_percentage = confidence * 100
        
        # Create confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Level"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': class_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Detailed predictions
        st.markdown("### 📈 **Detailed Classification Scores**")
        
        # Create DataFrame for predictions
        prediction_data = []
        for i, class_name in enumerate(class_names):
            prediction_data.append({
                'Class': f"{class_info[class_name]['emoji']} {class_name}",
                'Confidence': prediction[i] * 100,
                'Color': class_info[class_name]['color']
            })
        
        # Create horizontal bar chart
        fig_bar = px.bar(
            prediction_data,
            x='Confidence',
            y='Class',
            orientation='h',
            color='Color',
            color_discrete_map={info['color']: info['color'] for info in class_info.values()},
            title="Classification Confidence for All Classes"
        )
        fig_bar.update_layout(
            height=300,
            showlegend=False,
            xaxis_title="Confidence (%)",
            yaxis_title="Terrain Type"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Summary metrics
        st.markdown("### 📊 **Summary Statistics**")
        col_met1, col_met2, col_met3 = st.columns(3)
        
        with col_met1:
            st.metric(
                "Top Prediction",
                f"{class_emoji} {predicted_class}",
                f"{confidence_percentage:.1f}%"
            )
        
        with col_met2:
            second_best_idx = np.argsort(prediction)[-2]
            second_best_class = class_names[second_best_idx]
            second_best_confidence = prediction[second_best_idx] * 100
            st.metric(
                "Second Best",
                f"{class_info[second_best_class]['emoji']} {second_best_class}",
                f"{second_best_confidence:.1f}%"
            )
        
        with col_met3:
            entropy = -np.sum(prediction * np.log(prediction + 1e-8))
            certainty = 1 - entropy / np.log(len(class_names))
            st.metric(
                "Model Certainty",
                f"{certainty * 100:.1f}%",
                "High" if certainty > 0.8 else "Medium" if certainty > 0.6 else "Low"
            )
        
        # Interpretation
        st.markdown("### 🎯 **Result Interpretation**")
        if confidence > 0.8:
            st.success(f"🎉 **High Confidence**: The model is very confident this is a {predicted_class.lower()} terrain type.")
        elif confidence > 0.6:
            st.warning(f"⚠️ **Medium Confidence**: The model suggests this is likely a {predicted_class.lower()} terrain type.")
        else:
            st.error(f"❓ **Low Confidence**: The model is uncertain about the classification. Consider uploading a clearer image.")
        
        # Download results
        st.markdown("### 💾 **Export Results**")
        results_text = f"""
Satellite Image Classification Results
=====================================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {uploaded_file.name}

Primary Classification: {predicted_class}
Confidence: {confidence_percentage:.2f}%

Detailed Scores:
{chr(10).join([f"- {name}: {pred*100:.2f}%" for name, pred in zip(class_names, prediction)])}

Model Certainty: {certainty * 100:.1f}%
        """
        
        st.download_button(
            label="📊 Download Classification Report",
            data=results_text,
            file_name=f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    else:
        st.markdown("### 🔄 **Ready for Classification**")
        st.info("👆 Upload an image in the left panel to start the classification process")
        
        # Sample images section
        st.markdown("### 🖼️ **Sample Images**")
        st.markdown("Here are some examples of what each terrain type looks like:")
        
        sample_cols = st.columns(2)
        for i, (class_name, info) in enumerate(class_info.items()):
            with sample_cols[i % 2]:
                st.markdown(f"**{info['emoji']} {class_name}**")
                st.caption(info['description'])

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🛰️ <strong>Satellite Image Classifier</strong> | Powered by Deep Learning</p>
    <p>Built with TensorFlow & Streamlit | 🌍 Analyzing Earth from Above</p>
</div>
""", unsafe_allow_html=True)