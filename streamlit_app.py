import streamlit as st
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from huggingface_hub import hf_hub_download
import base64

# Set page configuration
st.set_page_config(
    page_title="ML Pet Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: white;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    .upload-section {
        padding: 1rem 0;
        margin: 2rem 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .prediction-text {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .confidence-text {
        font-size: 1.5rem;
        font-weight: 600;
        opacity: 0.8;
    }
    
    .cat-result {
        color: #FF6B6B;
    }
    
    .dog-result {
        color: #4ECDC4;
    }
    
    .uncertain-result {
        color: #FFB347;
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4ECDC4;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin: 0 0.5rem;
        flex: 1;
    }
    
    .emoji-large {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .progress-bar {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="main-header">
    <div class="main-title">🐾 AI Pet Classifier</div>
    <div class="subtitle">Klasifikasi Kucing vs Anjing dengan Vision Transformer</div>
</div>
""", unsafe_allow_html=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Model loading with enhanced caching
@st.cache_resource
def load_model():
    with st.spinner("🤖 Loading AI Model..."):
        model = timm.create_model("vit_base_patch16_224", pretrained=False)
        model.head = nn.Linear(model.head.in_features, 2)
        model.load_state_dict(torch.load("vit_model.pth", map_location=device))
        model.to(device)
        model.eval()
    return model

# Create columns for layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Info section
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Cara Menggunakan:</h3>
        <p>1. Upload gambar kucing atau anjing (format: JPG, JPEG, PNG)</p>
        <p>2. AI akan menganalisis gambar menggunakan Vision Transformer</p>
        <p>3. Dapatkan hasil prediksi dengan tingkat kepercayaan</p>
    </div>
    """, unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "📸 Upload Gambar Kucing atau Anjing",
    type=["jpg", "jpeg", "png"],
    help="Pilih gambar dengan format JPG, JPEG, atau PNG"
)

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Prediction logic
if uploaded_file is not None:
    try:
        # Create columns for image and results
        img_col, result_col = st.columns([1, 1])
        
        with img_col:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image, 
                caption="📷 Gambar yang diupload", 
                use_container_width=True,
                width=300
            )
        
        with result_col:
            st.success("✅ Model berhasil dimuat!")
            
            # Processing animation
            with st.spinner("🔍 Menganalisis gambar..."):
                # Preprocess and predict
                img_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()
            
            class_labels = ["🐱 Kucing", "🐶 Anjing"]
            class_emojis = ["🐱", "🐶"]
            confidence = probabilities[predicted_class].item()
            confidence_threshold = 0.70
            
            # Results display
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            if confidence < confidence_threshold:
                st.markdown(f"""
                <div class="emoji-large">🤔</div>
                <div class="prediction-text uncertain-result">Tidak Yakin</div>
                <div class="confidence-text">
                    Model tidak cukup yakin untuk mengklasifikasikan gambar ini
                </div>
                <div style="margin-top: 1rem; font-size: 1.2rem;">
                    Confidence: {confidence:.1%}
                </div>
                """, unsafe_allow_html=True)
            else:
                result_class = "cat-result" if predicted_class == 0 else "dog-result"
                st.markdown(f"""
                <div class="emoji-large">{class_emojis[predicted_class]}</div>
                <div class="prediction-text {result_class}">{class_labels[predicted_class]}</div>
                <div class="confidence-text">Confidence: {confidence:.1%}</div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Progress bars for both classes
            st.markdown("### 📊 Detail Prediksi:")
            
            cat_prob = probabilities[0].item()
            dog_prob = probabilities[1].item()
            
            st.markdown("🐱 **Kucing:**")
            st.progress(cat_prob)
            st.markdown(f"**{cat_prob:.1%}**")
            
            st.markdown("🐶 **Anjing:**")
            st.progress(dog_prob)
            st.markdown(f"**{dog_prob:.1%}**")
    
    except UnidentifiedImageError:
        st.error("❌ File yang diunggah bukan gambar yang valid atau rusak.")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan yang tidak terduga: {e}")

# Footer with stats
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stat-item">
        <div style="font-size: 2rem;">🤖</div>
        <div style="font-weight: bold;">Vision Transformer</div>
        <div>State-of-the-art AI</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-item">
        <div style="font-size: 2rem;">⚡</div>
        <div style="font-weight: bold;">Real-time</div>
        <div>Instant Classification</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-item">
        <div style="font-size: 2rem;">🎯</div>
        <div style="font-weight: bold;">High Accuracy</div>
        <div>Reliable Results</div>
    </div>
    """, unsafe_allow_html=True)

# Additional info
with st.expander("ℹ️ Tentang Model"):
    st.markdown("""
    **Vision Transformer (ViT)** adalah arsitektur deep learning yang menggunakan mekanisme attention 
    untuk mengklasifikasikan gambar. Model ini telah dilatih khusus untuk membedakan antara gambar 
    kucing dan anjing dengan akurasi tinggi.
    
    **Fitur:**
    - Menggunakan ViT base patch16_224
    - Preprocessing otomatis
    - Confidence threshold untuk hasil yang lebih akurat
    - Interface yang user-friendly
    """)

st.markdown("""
<div style="text-align: center; margin-top: 2rem; opacity: 0.7;">
    Made with ❤️ using Streamlit & PyTorch
</div>
""", unsafe_allow_html=True)
