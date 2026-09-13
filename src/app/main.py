"""Streamlit access-control app: enroll faces, then check access against them.

Run from the repo root:

    streamlit run src/app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingestion import (  # noqa: E402  (needs sys.path set first)
    FaceEnrollmentError,
    detect_faces,
    detect_single_face,
    generate_embedding,
)
from src.recognition import FaceRecognizer, RecognitionResult  # noqa: E402
from src.vector_store import FaceVectorStore  # noqa: E402

GRANTED_BGR = (0, 200, 0)
DENIED_BGR = (0, 0, 220)


@st.cache_resource
def get_store() -> FaceVectorStore:
    return FaceVectorStore()


def decode_upload(uploaded) -> np.ndarray:
    """Decode a Streamlit camera/file upload to a BGR image."""
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def photo_input(key_prefix: str):
    """A camera shot or file upload, whichever the user provided."""
    camera = st.camera_input("Take a photo", key=f"{key_prefix}_camera")
    upload = st.file_uploader(
        "...or upload a photo", type=["jpg", "jpeg", "png"], key=f"{key_prefix}_upload"
    )
    return camera or upload


def draw_result_box(image: np.ndarray, face, result: RecognitionResult) -> None:
    x1, y1, x2, y2 = face.bbox.astype(int)
    color = GRANTED_BGR if result.matched else DENIED_BGR
    label = (
        f"{result.name} {result.similarity:.2f}"
        if result.matched
        else f"unknown {result.similarity:.2f}"
    )
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        image, label, (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
    )


def render_enroll_tab(store: FaceVectorStore) -> None:
    st.subheader("Enroll a face")
    name = st.text_input("Name")
    photo = photo_input("enroll")

    if st.button("Enroll", disabled=not (name and photo)):
        image = decode_upload(photo)
        try:
            face = detect_single_face(image)
        except FaceEnrollmentError as exc:
            st.error(str(exc))
        else:
            store.add(name.strip(), generate_embedding(face))
            st.success(f"Enrolled '{name.strip()}'.")
            st.rerun()

    st.divider()
    names = store.list_names()
    st.write(f"**{store.count()} embedding(s) enrolled** for {len(names)} name(s):")
    for person_name in names:
        label_col, button_col = st.columns([4, 1])
        label_col.write(person_name)
        if button_col.button("Remove", key=f"remove_{person_name}"):
            store.delete_by_name(person_name)
            st.rerun()


def render_access_tab(store: FaceVectorStore, recognizer: FaceRecognizer) -> None:
    st.subheader("Check access")
    if store.count() == 0:
        st.info("No one is enrolled yet — add a face on the Enroll tab first.")

    photo = photo_input("access")
    if photo is None:
        return

    image = decode_upload(photo)
    faces = detect_faces(image)
    if not faces:
        st.warning("No face detected.")
        return

    annotated = image.copy()
    for face, result in recognizer.identify_faces(faces):
        draw_result_box(annotated, face, result)
        if result.matched:
            st.success(f"ACCESS GRANTED — {result.name} (similarity {result.similarity:.2f})")
        else:
            st.error(f"ACCESS DENIED (best similarity {result.similarity:.2f})")

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))


def main() -> None:
    st.set_page_config(page_title="Face Recognition Access Control", page_icon="🔐")
    st.title("Face Recognition Access Control")

    store = get_store()
    recognizer = FaceRecognizer(store)

    enroll_tab, access_tab = st.tabs(["Enroll", "Check access"])
    with enroll_tab:
        render_enroll_tab(store)
    with access_tab:
        render_access_tab(store, recognizer)


main()
