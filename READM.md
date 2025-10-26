### ☁️ Cloud OCR Mode (no local GPU required)

1. Configure the remote provider in `.env`:
   cp .env.example .env
   # fill in REMOTE_OCR_API_KEY
2. Launch the cloud stack:
   docker compose -f docker-compose.cloud.yml up --build
   This will build the lightweight backend container (no CUDA) and proxy all OCR requests to the configured cloud API. Frontend access remains `http://localhost:3000` by default.
