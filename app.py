import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Classes ───────────────────────────────────────────────────────────────────
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

CLASS_ICONS = {
    'airplane': '✈️', 'automobile': '🚗', 'bird': '🐦',
    'cat': '🐱', 'deer': '🦌', 'dog': '🐶',
    'frog': '🐸', 'horse': '🐴', 'ship': '🚢', 'truck': '🚛'
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: #040d1a;
    color: #cce8ff;
}

#MainMenu, footer, header { visibility: hidden; }

/* Animated background grid */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(0,200,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 2.8rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #00c8ff 0%, #0066ff 60%, #7b2fff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #1a4a6e;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, #00c8ff, #0066ff, transparent);
    margin: 1.2rem 0;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #060f1f !important;
    border: 1px dashed #0a3a5e !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00c8ff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0066ff, #00c8ff) !important;
    color: #040d1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    letter-spacing: 1px !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #060f1f, #0a1a2e);
    border: 1px solid #0a3a6e;
    border-radius: 16px;
    padding: 1.8rem;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, #00c8ff, #0066ff);
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 2.2rem;
    color: #00c8ff;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.result-confidence {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #1a4a6e;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.4rem;
}

/* Stat boxes */
.stat-box {
    background: #060f1f;
    border: 1px solid #0a2a4e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-number {
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1.4rem;
    color: #00c8ff;
}
.stat-label {
    font-size: 0.7rem;
    color: #1a4a6e;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
    margin-top: 0.2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #030a14 !important;
    border-right: 1px solid #0a2a4e !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #0066ff, #00c8ff) !important;
}

/* Info box */
.info-box {
    background: #060f1f;
    border-left: 3px solid #00c8ff;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #1a6080;
    margin-top: 0.8rem;
    font-family: 'Space Mono', monospace;
    line-height: 1.6;
}

/* Top prediction bar */
.pred-bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #4a8aaa;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "cifar10_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model()
model_loaded = model is not None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-weight:700; font-size:1rem;
         color:#00c8ff; letter-spacing:2px; margin-bottom:0.3rem;'>
        CIFAR-10
    </div>
    <div style='font-family:Space Mono,monospace; font-size:0.65rem; color:#0a3a5e;
         text-transform:uppercase; letter-spacing:2px; margin-bottom:1.5rem;'>
        CNN CLASSIFIER · TILAL AHMED
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**10 Classes**")
    for cls in CLASSES:
        icon = CLASS_ICONS[cls]
        st.markdown(
            f"<div style='font-family:Space Mono,monospace; font-size:0.75rem; "
            f"color:#1a5a7a; padding:0.25rem 0;'>{icon} {cls.upper()}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:1px; background:linear-gradient(90deg,#00c8ff,transparent); margin:1rem 0;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:0.65rem; color:#0a2a4e; line-height:1.8;'>
        Architecture: CNN<br>
        Input: 32×32×3<br>
        Conv layers: 2<br>
        Training epochs: 10<br>
        Dataset: CIFAR-10<br>
        Framework: TensorFlow
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>Image<br>Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>CNN · TensorFlow · CIFAR-10 · Deep Learning</div>", unsafe_allow_html=True)
st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

# Stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='stat-box'><div class='stat-number'>10</div><div class='stat-label'>Classes</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='stat-box'><div class='stat-number'>60K</div><div class='stat-label'>Images</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='stat-box'><div class='stat-number'>32×32</div><div class='stat-label'>Input Size</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='stat-box'><div class='stat-number'>CNN</div><div class='stat-label'>Architecture</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([2, 3], gap="large")

with left:
    st.markdown("<div style='font-family:Space Mono,monospace; font-size:0.75rem; color:#1a4a6e; text-transform:uppercase; letter-spacing:2px; margin-bottom:0.5rem;'>Upload Image</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    classify_btn = st.button("CLASSIFY IMAGE →")

    st.markdown("""
    <div class='info-box'>
        Upload any image of:<br>
        ✈ airplane &nbsp; 🚗 car &nbsp; 🐦 bird<br>
        🐱 cat &nbsp; 🦌 deer &nbsp; 🐶 dog<br>
        🐸 frog &nbsp; 🐴 horse &nbsp; 🚢 ship &nbsp; 🚛 truck
    </div>
    """, unsafe_allow_html=True)

with right:
    if classify_btn:
        if not uploaded_file:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Analyzing..."):
                # Preprocess image exactly like training
                img = image.resize((32, 32))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                if model_loaded:
                    predictions = model.predict(img_array, verbose=0)[0]
                    predicted_class = CLASSES[np.argmax(predictions)]
                    confidence = float(np.max(predictions)) * 100
                else:
                    # Demo mode
                    predictions = np.random.dirichlet(np.ones(10))
                    predicted_class = CLASSES[np.argmax(predictions)]
                    confidence = float(np.max(predictions)) * 100

            icon = CLASS_ICONS[predicted_class]
            st.markdown(f"""
            <div class='result-card'>
                <div style='font-size:2.5rem; margin-bottom:0.5rem;'>{icon}</div>
                <div class='result-label'>{predicted_class}</div>
                <div class='result-confidence'>Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>**Top 5 Predictions**", unsafe_allow_html=True)

            top5_idx = np.argsort(predictions)[::-1][:5]
            for idx in top5_idx:
                cls_name = CLASSES[idx]
                prob = predictions[idx] * 100
                icon = CLASS_ICONS[cls_name]
                st.markdown(
                    f"<div class='pred-bar-label'>{icon} {cls_name.upper()}</div>",
                    unsafe_allow_html=True
                )
                st.progress(int(prob))
                st.markdown(
                    f"<div style='font-family:Space Mono,monospace; font-size:0.7rem; "
                    f"color:#0a4a6a; text-align:right; margin-top:-0.8rem; "
                    f"margin-bottom:0.5rem;'>{prob:.1f}%</div>",
                    unsafe_allow_html=True
                )

            if not model_loaded:
                st.info("⚠ Demo mode — place cifar10_model.h5 in the app folder for real predictions.")
    else:
        st.markdown("""
        <div style='background:#060f1f; border:1px solid #0a2a4e; border-radius:12px;
             padding:2rem; text-align:center; margin-top:1rem;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>🔭</div>
            <div style='font-family:Space Mono,monospace; font-size:0.8rem;
                 color:#0a3a5e; text-transform:uppercase; letter-spacing:2px;'>
                Upload an image<br>and click classify
            </div>
        </div>
        """, unsafe_allow_html=True)
