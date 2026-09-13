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
# optional but recommended: gate the app behind a password
export APP_PASSWORD=change-me      # Linux/macOS
$env:APP_PASSWORD = "change-me"    # Windows PowerShell

streamlit run src/app/main.py
```

Opens a browser UI with two tabs: **Enroll** (name + photo -> stored in the
vector store) and **Check access** (a two-shot blink challenge — eyes open,
then blink — followed by recognized name/similarity, or "unknown"; a static
photo held up to the camera fails the blink check). Without `APP_PASSWORD`
set (or an `app_password` entry in `.streamlit/secrets.toml`), the app runs
without login protection and shows a warning — anyone who can open it can
enroll or remove faces.

### Accounts

Login starts as one shared `APP_PASSWORD`. Once logged in, use the
**Manage accounts** panel in the sidebar to add named username/password
accounts (bcrypt-hashed, stored in `data/enrolled_faces/users.json`) — as
soon as one exists, login switches to per-account for everyone, and an
account can be individually revoked without changing anyone else's
password.

### Lockout

After 5 failed access checks within 5 minutes, the Check access tab locks
out for a 2-minute cooldown (see `src/app/lockout.py` for the constants).

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `APP_PASSWORD` | none (open, with a warning) | Shared login password until a named account exists |
| `RECOGNITION_THRESHOLD` | `0.5` | Cosine similarity a match must clear to count as recognized |
| `ACCESS_WEBHOOK_URL` | none (log-only) | POSTed `{"event": "access_granted", "name": ...}` on a granted check — point this at a relay/door-controller webhook |

## Running with Docker

```bash
docker build -t face-access-control .
docker run -p 8501:8501 -v facedata:/app/data/enrolled_faces -e APP_PASSWORD=change-me face-access-control
```

The volume persists enrolled faces, accounts, and the audit log across
container restarts. Not verified in a real Docker environment (none was
available while building this) — if the build fails, it's most likely a
missing system library for `opencv-python` on the base image.

## Limitations

- **Liveness only defends against a static photo/screen replay**, not a
  video of the real person blinking — that needs a much harder anti-spoofing
  approach (depth sensing, remote-PPG, or a trained model) that this project
  doesn't attempt.
- **`RECOGNITION_THRESHOLD` is a reasonable default, not a tuned value** —
  real tuning needs false-accept/false-reject data from actual enrolled
  users, which doesn't exist yet.
- **The CI workflow (`.github/workflows/tests.yml`) hasn't run on GitHub** —
  its steps mirror the exact commands used to verify tests locally, but no
  GitHub Actions runner was available to confirm it end-to-end.

## Notes

`data/enrolled_faces/` is excluded from version control because it contains
personal face images and embeddings.
