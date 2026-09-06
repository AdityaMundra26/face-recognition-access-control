"""Face ingestion and enrollment."""

from .enrollment import (
    EMBEDDING_DIM,
    FaceEnrollmentError,
    detect_faces,
    detect_single_face,
    enroll_face,
    generate_embedding,
    get_face_app,
    load_image,
)

__all__ = [
    "EMBEDDING_DIM",
    "FaceEnrollmentError",
    "detect_faces",
    "detect_single_face",
    "enroll_face",
    "generate_embedding",
    "get_face_app",
    "load_image",
]
