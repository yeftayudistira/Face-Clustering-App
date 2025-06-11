import streamlit as st
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis
from huggingface_hub import snapshot_download
import tempfile
import base64
import zipfile
from io import BytesIO

# Setup
st.set_page_config(page_title="Face Clustering", layout="wide")
st.title("📸 Face Clustering with AuraFace + DBSCAN")

@st.cache_resource
def load_model():
    snapshot_download("fal/AuraFace-v1", local_dir="models/auraface")
    app = FaceAnalysis(name="auraface", providers=["CUDAExecutionProvider", "CPUExecutionProvider"], root=".")
    app.prepare(ctx_id=0)
    return app

face_app = load_model()

st.markdown("""
Upload beberapa foto wajah, dan aplikasi akan otomatis mendeteksi, mengelompokkan wajah serupa, serta menampilkan hasil klaster dengan pilihan unduh gambar asli.
""")

uploaded_files = st.file_uploader("📂 Unggah gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    embeddings, face_images, image_sources = [], [], []

    with st.spinner("🔍 Memproses gambar..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            for file in uploaded_files:
                img_bytes = file.read()
                np_img = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                filename = file.name

                faces = face_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                for face in faces:
                    x1, y1, x2, y2 = map(int, face.bbox)
                    h, w, _ = img.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        cropped_face = img[y1:y2, x1:x2]
                        face_images.append(cropped_face)
                        embeddings.append(face.normed_embedding)
                        image_sources.append((filename, img_bytes))

    if embeddings:
        st.success(f"✅ {len(embeddings)} wajah terdeteksi.")

        X = np.array(embeddings)
        db = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(X)
        labels = db.labels_

        clusters = {}
        for img, label, (filename, img_bytes) in zip(face_images, labels, image_sources):
            clusters.setdefault(label, []).append((img, filename, img_bytes))

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "a") as zip_file:
            for label, faces in clusters.items():
                for i, (_, filename, img_bytes) in enumerate(faces):
                    zip_file.writestr(f"cluster_{label}/{i}_{filename}", img_bytes)

        st.download_button(
            label="📦 Unduh Semua sebagai ZIP",
            data=zip_buffer.getvalue(),
            file_name="face_clusters.zip",
            mime="application/zip"
        )

        for label, faces in clusters.items():
            st.markdown(f"### 🧩 Cluster {label} - {len(faces)} wajah")
            cols = st.columns(min(len(faces), 5))

            for i, (img, filename, img_bytes) in enumerate(faces):
                with cols[i % len(cols)]:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=filename, use_column_width=True)
                    b64 = base64.b64encode(img_bytes).decode()
                    href = f'<a href="data:file/jpg;base64,{b64}" download="{filename}">⬇️ Unduh Asli</a>'
                    st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Tidak ada wajah terdeteksi.")
