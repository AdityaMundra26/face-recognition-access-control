# face-recognition-access-control

A face-recognition-based access control system.

## Project structure

```
face-recognition-access-control/
├── src/
│   ├── ingestion/      # Capture and enroll face images
│   ├── recognition/    # Face detection and embedding/recognition
│   ├── vector_store/   # Storage and similarity search for face embeddings
│   └── app/            # Application entry point / API
├── data/
│   └── enrolled_faces/ # Enrolled face images and embeddings (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# Activate the virtual environment
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies (once requirements.txt is populated)
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run src/app/main.py
```

Opens a browser UI with two tabs: **Enroll** (name + photo -> stored in the
vector store) and **Check access** (photo -> recognized name/similarity per
face, or "unknown").

## Notes

`data/enrolled_faces/` is excluded from version control because it contains
personal face images and embeddings.
