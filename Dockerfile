FROM python:3.11-slim

WORKDIR /app

# opencv-python needs these even though it's the headless-free build;
# libgl1/libglib2.0-0 satisfy its shared-library imports on Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Enrolled faces, the vector store, accounts, and the audit log all live
# under data/enrolled_faces/ -- mount a volume there to persist them across
# container restarts, e.g.: -v facedata:/app/data/enrolled_faces
VOLUME ["/app/data/enrolled_faces"]

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "src/app/main.py", "--server.address=0.0.0.0"]
