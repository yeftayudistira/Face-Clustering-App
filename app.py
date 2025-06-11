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

# Setup Streamlit
st.set_page_config(page_title="Face Clustering App", layout="wide")
st.title("🧠 Face Clustering App")
st.markdown("Upload beberapa foto, lalu lihat hasil clustering wajah!")

# ==== Inisialisasi model di luar ====
@st.cache_resource
def load_face_app():
    model_dir = "models/auraface"
    if not os.path.exists(model_dir):
        snapshot_download("fal/AuraFace-v1", local_dir=model_dir)

    face_app = FaceAnalysis(name="auraface", root=".", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0)
    return face_app

face_app = load_face_app()
# ====================================

# Upload gambar
uploaded_files = st.file_uploader("Upload beberapa gambar wajah", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    embeddings, image_sources, face_images = [], [], []

    st.info("📸 Mendeteksi wajah...")
    progress = st.progress(0)

    for idx, file in enumerate(uploaded_files):
        bytes_data = file.read()
        np_img = np.frombuffer(bytes_data, np.uint8)
        input_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

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

        progress.progress((idx + 1) / len(uploaded_files))

    if not embeddings:
        st.error("❌ Tidak ada wajah terdeteksi.")
        st.stop()

    st.success(f"✅ Total wajah terdeteksi: {len(embeddings)}")

    # Clustering
    st.info("🔗 Melakukan clustering wajah...")
    X = np.array(embeddings)
    db = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(X)
    labels = db.labels_

    # Buat dictionary per cluster
    clusters = {}
    for face_img, label in zip(face_images, labels):
        clusters.setdefault(label, []).append(face_img)

    # Dropdown UI
    cluster_options = sorted(clusters.keys())
    selected_cluster = st.selectbox("🧩 Pilih Cluster", cluster_options)

    st.subheader(f"👥 Wajah dalam Cluster {selected_cluster}")
    cols = st.columns(5)
    for idx, img in enumerate(clusters[selected_cluster]):
        with cols[idx % 5]:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Download ZIP per cluster
    if st.button("📦 Download Cluster sebagai ZIP"):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, f"cluster_{selected_cluster}.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for idx, img in enumerate(clusters[selected_cluster]):
                    filename = f"face_{idx}.jpg"
                    file_path = os.path.join(tmpdir, filename)
                    cv2.imwrite(file_path, img)
                    zipf.write(file_path, arcname=filename)

            with open(zip_path, "rb") as f:
                st.download_button(
                    label="⬇️ Klik untuk download ZIP",
                    data=f,
                    file_name=f"cluster_{selected_cluster}.zip",
                    mime="application/zip"
                )
