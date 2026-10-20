# Chest Xpert — Deployment & Demo Guide

## Architecture

- **Backend:** FastAPI on the GPU server (Tesla V100, falls back to CPU when V100 is full)
- **Mobile:** React Native + Expo (TypeScript)
- **Communication:** Mobile app calls FastAPI backend over HTTP

## Backend URLs

| Access Method | URL | Notes |
|-------------|-----|-------|
| Local (same machine) | http://localhost:8000 | Direct access |
| Internal network | http://10.203.6.228:8000 | If on same network |
| SSH tunnel | `ssh -L 8002:localhost:8000 doli@<server-ip>` | Secure, recommended |
| External IP | http://103.159.214.203:8000 | May be firewalled |

**Important:** Do not expose port 8000 directly to the public internet without a reverse proxy and authentication. Use an SSH tunnel for secure access.

## Starting the Backend

```bash
source /home/doli/anaconda3/bin/activate hct-lsr
cd "/home/doli/Chest Xpert"

# Option 1: Use the management script
bash scripts/manage.sh start

# Option 2: Manual
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Check status
bash scripts/manage.sh status
curl http://localhost:8000/api/v1/health
```

## SSH Tunnel (Secure Professor Demo)

From your local machine:

```bash
ssh -L 8002:localhost:8000 doli@<server-ip>
# Then open: http://localhost:8000/docs (API docs)
# Mobile app: set EXPO_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Mobile App (Expo)

```bash
cd "/home/doli/Chest Xpert/mobile"
npm install
npx expo start
```

For the professor demo:
- **Expo Go:** Install Expo Go on a phone, scan the QR code from `npx expo start`
- **Expo Web:** `npx expo start --web` (runs in browser, easy to share via link)
- **Android build:** `npx expo build:android` (requires EAS credentials)

Set `EXPO_PUBLIC_API_BASE_URL` to your tunnel URL before starting.

## Resource Management

```bash
# Check GPU and processes
bash scripts/manage.sh gpu

# Check RAM
bash scripts/manage.sh ram

# Check disk and cache
bash scripts/manage.sh disk
bash scripts/manage.sh cache

# Clean expired artifacts (dry-run first)
bash scripts/manage.sh cleanup true
bash scripts/manage.sh cleanup false

# Run component tests
bash scripts/manage.sh test-classifier
bash scripts/manage.sh test-gradcam
bash scripts/manage.sh test-report
bash scripts/manage.sh test-e2e
```

## Security

- Authentication is disabled by default (`CHESTXPERT_REQUIRE_AUTH=false`)
- Enable for demos with sensitive uploads: set `CHESTXPERT_API_KEY` and `CHESTXPERT_REQUIRE_AUTH=true`
- The mobile app sends `X-API-Key` header when `EXPO_PUBLIC_API_KEY` is set
- Never commit `.env` files or API keys

## Limitations

- No training or fine-tuning performed
- Scores are raw model outputs, not calibrated probabilities
- Not a medical device — AI-generated research draft requiring radiologist review
- Tesla V100 is shared and often near capacity; CPU fallback is used when GPU is full
- Modality detection is a heuristic, not a validated clinical test

## Rollback

The original Streamlit app is preserved at `chest_xpert/app.py` (git commit `56ec74e`). To rollback:

```bash
git checkout 56ec74e -- chest_xpert/
streamlit run chest_xpert/app.py
```
