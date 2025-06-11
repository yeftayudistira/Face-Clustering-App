import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from huggingface_hub import snapshot_download
from io import BytesIO
import zipfile
import tempfile

# Setup Streamlit dengan tema Google Photos
st.set_page_config(
    page_title="Face Clustering - Photos", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk tema Google Photos
st.markdown("""
<style>
    /* Force light mode - override system dark mode */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Override Streamlit's dark mode elements */
    .stApp > header {
        background-color: transparent !important;
    }
    
    .stApp [data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }
    
    /* Main container styling */
    .main > div {
        padding: 2rem 1rem;
        background: #ffffff !important;
        min-height: 100vh;
        color: #000000 !important;
    }
    
    /* Override all text colors */
    .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
        color: white !important;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white !important;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem;
        margin-bottom: 0;
    }
    
    /* Upload section */
    .upload-section {
        background: white !important;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 2px solid #f0f0f0;
        transition: all 0.3s ease;
        color: #000000 !important;
    }
    
    .upload-section:hover {
        border-color: #4285f4;
        box-shadow: 0 8px 30px rgba(66,133,244,0.15);
    }
    
    /* Cluster section */
    .cluster-section {
        background: white !important;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        color: #000000 !important;
    }
    
    /* Cluster header */
    .cluster-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .cluster-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #202124 !important;
        margin: 0;
    }
    
    .cluster-count {
        background: linear-gradient(45deg, #4285f4, #34a853);
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Image grid styling */
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .face-image {
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        overflow: hidden;
    }
    
    .face-image:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #4285f4, #34a853) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(66,133,244,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(66,133,244,0.4);
        background: linear-gradient(45deg, #1a73e8, #137333) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: white !important;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 0.5rem;
        transition: all 0.3s ease;
        color: #000000 !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #4285f4;
        box-shadow: 0 0 0 3px rgba(66,133,244,0.1);
    }
    
    .stSelectbox label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed #4285f4;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(66,133,244,0.05) !important;
        transition: all 0.3s ease;
        color: #000000 !important;
    }
    
    .stFileUploader > div:hover {
        background: rgba(66,133,244,0.1) !important;
        border-color: #1a73e8;
    }
    
    .stFileUploader label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    /* Progress styling */
    .stProgress > div > div {
        background: linear-gradient(45deg, #4285f4, #34a853) !important;
        border-radius: 10px;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(52,168,83,0.1) !important;
        border: 1px solid #34a853;
        border-radius: 10px;
        color: #137333 !important;
    }
    
    .stError {
        background: rgba(234,67,53,0.1) !important;
        border: 1px solid #ea4335;
        border-radius: 10px;
        color: #d33b2c !important;
    }
    
    /* Override spinner */
    .stSpinner > div {
        color: #4285f4 !important;
    }
    
    /* Force white background for all containers */
    .block-container {
        background-color: #ffffff !important;
    }
    
    .stApp > div > div {
        background-color: #ffffff !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="main-header">
    <div class="main-title">📸 Face Clustering</div>
    <div class="main-subtitle">Organize your photos by faces, just like Google Photos</div>
</div>
""", unsafe_allow_html=True)

# ==== Inisialisasi model dan session state ====
@st.cache_resource
def load_face_app():
    try:
        snapshot_download(
            "fal/AuraFace-v1",
            local_dir="models/auraface",
        )
        face_app = FaceAnalysis(
            name="auraface",
            providers=["CPUExecutionProvider"],
            root=".",
        )
        face_app.prepare(ctx_id=0)
        return face_app
    except Exception as e:
        st.error(f"Error loading face analysis model: {str(e)}")
        return None

# Initialize face app
face_app = load_face_app()

if face_app is None:
    st.error("❌ Failed to load face analysis model. Please check your internet connection and try again.")
    st.stop()

# Initialize session state
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'total_faces' not in st.session_state:
    st.session_state.total_faces = 0

# ====================================

# Upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📤 Upload Your Photos")
st.markdown("Select multiple photos to find and group similar faces")

uploaded_files = st.file_uploader(
    "Choose photos", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    help="Upload multiple photos containing faces"
)
st.markdown('</div>', unsafe_allow_html=True)

# Check if we need to process new files
current_file_names = [file.name for file in uploaded_files] if uploaded_files else []
need_processing = (uploaded_files and 
                  (st.session_state.clusters is None or 
                   set(current_file_names) != set(st.session_state.processed_files)))

if uploaded_files and need_processing:
    # Processing with modern loading
    with st.spinner("🔍 Analyzing faces in your photos..."):
        embeddings, image_sources, face_images = [], [], []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {file.name}...")
                
                bytes_data = file.read()
                np_img = np.frombuffer(bytes_data, np.uint8)
                input_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                
                if input_image is None:
                    st.warning(f"⚠️ Could not read image: {file.name}")
                    continue
                    
                cv2_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                faces = face_app.get(cv2_image)
                
                for face in faces:
                    x1, y1, x2, y2 = [int(i) for i in face.bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(input_image.shape[1], x2), min(input_image.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        cropped = input_image[y1:y2, x1:x2]
                        face_images.append(cropped)
                        embeddings.append(face.normed_embedding)
                        image_sources.append(file.name)
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
            except Exception as e:
                st.warning(f"⚠️ Error processing {file.name}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
    
    if not embeddings:
        st.error("❌ No faces detected in the uploaded photos. Please try with different images.")
        st.stop()
    
    # Clustering
    with st.spinner("🧠 Grouping similar faces..."):
        try:
            X = np.array(embeddings)
            db = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(X)
            labels = db.labels_
            
            # Buat dictionary per cluster dan simpan ke session state
            clusters = {}
            for face_img, label in zip(face_images, labels):
                clusters.setdefault(label, []).append(face_img)
            
            # Save to session state
            st.session_state.clusters = clusters
            st.session_state.processed_files = current_file_names
            st.session_state.total_faces = len(embeddings)
            
        except Exception as e:
            st.error(f"❌ Error during clustering: {str(e)}")
            st.stop()

# Display results if we have processed data
if uploaded_files and st.session_state.clusters is not None:
    clusters = st.session_state.clusters
    
    # Success message
    unique_clusters = len([k for k in clusters.keys() if k != -1])
    st.success(f"✨ Found {st.session_state.total_faces} faces organized into {unique_clusters} groups!")
    
    # Cluster section
    st.markdown('<div class="cluster-section">', unsafe_allow_html=True)
        
    # Cluster dropdown
    cluster_options = sorted(clusters.keys())
    cluster_names = []
    for cluster_id in cluster_options:
        if cluster_id == -1:
            cluster_names.append(f"🔍 Unmatched faces ({len(clusters[cluster_id])} faces)")
        else:
            cluster_names.append(f"👥 Person {cluster_id + 1} ({len(clusters[cluster_id])} faces)")
        
    st.markdown("### 🎯 Select a Face Group")
    selected_index = st.selectbox(
        "Choose which group of faces to view:",
        range(len(cluster_options)),
        format_func=lambda x: cluster_names[x],
        help="Each group contains similar faces detected across your photos"
    )
        
    selected_cluster = cluster_options[selected_index]
        
    # Cluster header
    st.markdown(f"""
    <div class="cluster-header">
        <div class="cluster-title">{cluster_names[selected_index]}</div>
        <div class="cluster-count">{len(clusters[selected_cluster])} photos</div>
    </div>
    """, unsafe_allow_html=True)
        
    # Display faces in a responsive grid
    faces_in_cluster = clusters[selected_cluster]
        
    # Create responsive columns
    cols_per_row = 6
    for i in range(0, len(faces_in_cluster), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, img in enumerate(faces_in_cluster[i:i+cols_per_row]):
            with cols[j]:
                try:
                    st.image(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                        use_column_width=True,
                        caption=f"Face {i+j+1}"
                    )
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
        
    st.markdown('</div>', unsafe_allow_html=True)
        
    # Download section - Centered
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📦 Download This Face Group", use_container_width=True):
            with st.spinner("📁 Preparing download..."):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        zip_path = os.path.join(tmpdir, f"face_group_{selected_cluster}.zip")
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for idx, img in enumerate(clusters[selected_cluster]):
                                filename = f"face_{idx+1}.jpg"
                                file_path = os.path.join(tmpdir, filename)
                                cv2.imwrite(file_path, img)
                                zipf.write(file_path, arcname=filename)
                        
                        with open(zip_path, "rb") as f:
                            zip_data = f.read()
                    
                    # Show download button in center column
                    with col2:        
                        st.download_button(
                            label="⬇️ Download ZIP File",
                            data=zip_data,
                            file_name=f"face_group_{selected_cluster+1 if selected_cluster != -1 else 'unmatched'}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"❌ Error creating download: {str(e)}")

elif uploaded_files and st.session_state.clusters is None:
    st.info("⏳ Please wait while we process your photos...")

else:
    # Welcome message when no files uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #5f6368;">
        <h3>🚀 Get Started</h3>
        <p>Upload your photos above to automatically detect and group faces using AI</p>
        <p><strong>Features:</strong></p>
        <p>✨ Automatic face detection • 🧠 Smart grouping • 📱 Mobile-friendly • 📦 Easy downloads</p>
    </div>
    """, unsafe_allow_html=True)
    
# Add clear button to reset session state - Better centered
if st.session_state.clusters is not None:
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Process New Photos", type="secondary", use_container_width=True):
            st.session_state.clusters = None
            st.session_state.processed_files = []
            st.session_state.total_faces = 0
            st.experimental_rerun()
