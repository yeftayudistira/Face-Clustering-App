import streamlit as st
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
import zipfile
import tempfile

# Konfigurasi Streamlit
st.set_page_config(page_title="Clustering Wajah", layout="wide")
st.title("🧠 Clustering Wajah")
st.markdown("Upload gambar, deteksi wajah, lalu pilih cluster dari dropdown.")

# ==== Inisialisasi model di luar proses upload ====
@st.cache_resource
def load_face_app():
    model_dir = "models/auraface"
    if not os.path.exists(model_dir):
        snapshot_download("fal/AuraFace-v1", local_dir=model_dir)
    face_app = FaceAnalysis(name="auraface", root=".", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0)
    return face_app

face_app = load_face_app()

# ==== Upload gambar ====
uploaded_files = st.file_uploader("📁 Upload gambar wajah (jpg/png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    embeddings, face_images = [], []

    progress = st.progress(0, "⏳ Mendeteksi wajah...")
    for idx, file in enumerate(uploaded_files):
        img_bytes = file.read()
        np_img = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb)

        for face in faces:
            x1, y1, x2, y2 = [int(i) for i in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 > x1 and y2 > y1:
                face_img = image[y1:y2, x1:x2]
                embeddings.append(face.normed_embedding)
                face_images.append(face_img)
        progress.progress((idx + 1) / len(uploaded_files))

    if not embeddings:
        st.error("❌ Tidak ada wajah terdeteksi.")
        st.stop()

    # Clustering
    X = np.array(embeddings)
    db = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(X)
    labels = db.labels_

    # Buat dictionary cluster
    clusters = {}
    for label, img in zip(labels, face_images):
        clusters.setdefault(label, []).append(img)

    # Dropdown pemilihan cluster
    selected_label = st.selectbox("🧩 Pilih Cluster", sorted(clusters.keys()))
    st.subheader(f"👥 Wajah di Cluster {selected_label}")
    
    cols = st.columns(5)
    for idx, img in enumerate(clusters[selected_label]):
        with cols[idx % 5]:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Tombol download ZIP
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, f"cluster_{selected_label}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for idx, img in enumerate(clusters[selected_label]):
                face_path = os.path.join(tmpdir, f"face_{idx}.jpg")
                cv2.imwrite(face_path, img)
                zipf.write(face_path, arcname=f"face_{idx}.jpg")

        with open(zip_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Cluster sebagai ZIP",
                data=f,
                file_name=f"cluster_{selected_label}.zip",
                mime="application/zip"
            )
