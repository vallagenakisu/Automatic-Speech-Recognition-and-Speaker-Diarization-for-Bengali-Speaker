# Bengali ASR & Speaker Diarization Web App

A web application for automatic speech recognition (ASR) and speaker diarization of Bengali audio/video. Upload a file, get speaker-labeled Bengali subtitles with a synced media player — entirely free to run on Kaggle with GPU.

---

## Demo

> Live at: [https://vikki-woody-plicately.ngrok-free.dev](https://vikki-woody-plicately.ngrok-free.dev) *(when the Kaggle notebook is running)*

Upload audio or video → processing → speaker-labeled subtitles with playback sync.

---

## Features

- **Bengali ASR** — Fine-tuned Whisper Medium (LoRA, CTranslate2) via `faster-whisper`
- **Speaker Diarization** — DiariZen pipeline (`BUT-FIT/diarizen-wavlm-large-s80-md-v2`) with custom fine-tuned segmentation head
- **Speaker Tagging** — Name each detected speaker; labels update live in player and subtitles
- **Synced Subtitles** — Active subtitle highlighted as media plays; click any line to seek
- **Short Readable Subtitles** — Long segments auto-split at word boundaries (max 5 s per subtitle)
- **Download** — Export SRT or VTT with custom speaker names
- **Supported formats** — WAV, MP3, FLAC, OGG, M4A, MP4, MKV, AVI, MOV, WebM (up to 500 MB)

---

## Architecture

```
webapp/
├── app.py            # FastAPI server (8 routes, background job processing)
├── pipeline.py       # Full pipeline: VAD → ASR → diarization → merge → SRT/VTT
├── config.py         # All settings via environment variables
├── static/
│   └── index.html    # Single-page frontend (vanilla HTML/CSS/JS)
├── requirements.txt
├── Dockerfile
├── deploy_kaggle.ipynb  # Kaggle deployment notebook (recommended)
└── deploy_colab.ipynb   # Google Colab deployment notebook
```

### Pipeline stages

1. **Audio extraction** (ffmpeg for video files)
2. **Noise reduction** (noisereduce)
3. **VAD** (Silero VAD — strips silence)
4. **ASR** (`BatchedInferencePipeline` from faster-whisper, Bengali, word timestamps)
5. **Diarization** (DiariZen with fine-tuned powerset segmentation + VBx clustering)
6. **Post-processing** — 6-stage Optuna-optimized pipeline:
   - Segment stitching (gap threshold 3.24 s)
   - Short segment filtering (min 0.78 s)
   - Neighbor-aware speaker reassignment
   - Backchannel suppression
   - Final gap merge
7. **Merge ASR + Diarization** (max-overlap speaker assignment)
8. **Segment splitting** (word-boundary split for subtitles > 5 s)
9. **SRT / VTT generation**

---

## Models

| Model | Source | Purpose |
|---|---|---|
| Whisper Medium (LoRA fine-tune) | [faysal314/whisper-md-lora-ep7-ct2](https://www.kaggle.com/datasets/faysal314/whisper-md-lora-ep7-ct2) on Kaggle | Bengali ASR |
| DiariZen base | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` on HuggingFace | Diarization backbone |
| DiariZen fine-tune | [mithilameowmeow/finetune-segmentation](https://www.kaggle.com/models/mithilameowmeow/finetune-segmentation) on Kaggle | Bengali-tuned segmentation head |
| Speaker embedding | `pyannote/wespeaker-voxceleb-resnet34-LM` on HuggingFace | Speaker embeddings for VBx clustering |

---

## Deployment on Kaggle (Recommended — Free GPU)

### Prerequisites

1. Kaggle account with internet access enabled
2. ngrok account (free) — get authtoken at [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken)
3. HuggingFace account — accept terms for [`pyannote/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)

### Setup

**Step 1 — Kaggle Secrets**

In your Kaggle notebook → **Add-ons → Secrets**, add:
| Name | Value |
|---|---|
| `NGROK_TOKEN` | Your ngrok authtoken |
| `HF_TOKEN` | Your HuggingFace token |

**Step 2 — Add Kaggle datasets/models as inputs**

In your notebook, add these as inputs:
- Dataset: `faysal314/whisper-md-lora-ep7-ct2`
- Model: `mithilameowmeow/finetune-segmentation` (pytorch / default / version 1)
- Dataset: `tanzirmannanturzo/bangla-webapp` (the 4 webapp files)

**Step 3 — The `bangla-webapp` dataset**

Create a Kaggle dataset named `bangla-webapp` and upload these 4 files **flat** (no subfolders):
- `config.py`
- `pipeline.py`
- `app.py`
- `index.html` *(the file from `webapp/static/index.html`)*

**Step 4 — Run the notebook**

Open `webapp/deploy_kaggle.ipynb`, set **GPU T4 x2 + Internet ON**, then run all cells:

| Cell | What it does |
|---|---|
| 1 | Installs ffmpeg + Python dependencies + DiariZen from GitHub |
| 2 | Creates app directories, sets model path env vars |
| 3 | Copies files from dataset, applies any compatibility patches |
| 4 | Debug — verifies pipeline.py looks correct |
| 5 | Starts FastAPI server + ngrok tunnel, prints public URL |

---

## Local Development

### Requirements

- Python 3.10+
- CUDA GPU recommended (CPU works but is slow)
- ffmpeg installed and on PATH

```bash
# Clone the repo
git clone https://github.com/<your-username>/Automatic-Speech-Recognition-and-Speaker-Diarization-for-Bengali-Speaker.git
cd Automatic-Speech-Recognition-and-Speaker-Diarization-for-Bengali-Speaker

# Install dependencies
pip install -r webapp/requirements.txt

# Install DiariZen
git clone https://github.com/BUTSpeechFIT/DiariZen.git
pip install -e DiariZen/pyannote-audio
pip install -e DiariZen
```

### Configure

```bash
export ASR_MODEL_PATH="/path/to/whisper_md_lora_e7_ct2"
export DIARIZEN_FINETUNED_PATH="/path/to/best_diarizen_retrained.pt"
export HF_TOKEN="hf_..."
```

### Run

```bash
cd webapp
uvicorn app:app --host 0.0.0.0 --port 7860
# Open http://localhost:7860
```

### Docker

```bash
cd webapp
docker build -t bengali-asr-diar .
docker run --gpus all -p 7860:7860 \
  -e ASR_MODEL_PATH=/models/whisper_md_lora_e7_ct2 \
  -e DIARIZEN_FINETUNED_PATH=/models/best_diarizen_retrained.pt \
  -e HF_TOKEN=hf_... \
  -v /path/to/models:/models \
  bengali-asr-diar
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serve the frontend SPA |
| `/api/upload` | POST | Upload a media file, returns `job_id` |
| `/api/status/{job_id}` | GET | Poll job status (`queued` / `processing` / `done` / `error`) |
| `/api/result/{job_id}` | GET | Get full results (segments, speakers, metadata) |
| `/api/media/{job_id}` | GET | Stream the original uploaded file |
| `/api/subtitle/{job_id}` | GET | Serve VTT subtitle file (for `<track>` element) |
| `/api/download/{job_id}` | GET | Download SRT or VTT (`?format=srt` or `?format=vtt`) |

---

## Fine-tuning Details

- **Training notebook**: `faster-whisper-e7-lora.ipynb`
- **Whisper**: Medium model, LoRA fine-tuned for 7 epochs on Bengali speech, exported to CTranslate2
- **Diarization**: DiariZen segmentation head fine-tuned with Powerset encoding (7 speakers, 2 simultaneous), optimized with Optuna for 6 post-processing hyperparameters (DER minimization)

---

## License

This project is released for research and educational use. Model components carry their own licenses:
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — MIT
- [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) — see MODEL_LICENSE
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) — MIT
