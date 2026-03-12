# Bengali ASR & Speaker Diarization Web App

A minimal web application for generating speaker-labeled Bengali subtitles from audio/video files.

## Features
- Upload audio (WAV, MP3, FLAC) or video (MP4, MKV, AVI, MOV, WebM)
- Automatic Speech Recognition (faster-whisper with fine-tuned LoRA model)
- Speaker Diarization (DiariZen pipeline with fine-tuned segmentation)
- Real-time synced subtitle display during playback
- Download subtitles in SRT/VTT format
- Speaker color-coding

## Architecture
```
Frontend (HTML/CSS/JS)  ←→  FastAPI Backend  ←→  Models (faster-whisper + DiariZen)
```

## Files
```
webapp/
├── app.py              # FastAPI server (routes, upload, status polling)
├── pipeline.py         # Processing pipeline (ASR + diarization + merge)
├── config.py           # Configuration (model paths, settings)
├── requirements.txt    # Python dependencies
├── Dockerfile          # For HuggingFace Spaces / Docker deployment
├── static/
│   └── index.html      # Frontend (single-page app)
├── deploy_colab.ipynb  # Google Colab deployment notebook
└── deploy_kaggle.ipynb # Kaggle deployment notebook
```

## Deployment Options (Free, with GPU)

### Option 1: Google Colab (Recommended for testing)
1. Upload `deploy_colab.ipynb` to Google Colab
2. Set Runtime → Change runtime type → **T4 GPU**
3. Upload `config.py`, `pipeline.py`, `app.py`, `static/index.html` to `/content/webapp/`
4. Get free ngrok token at https://ngrok.com
5. Run all cells — you'll get a public URL

### Option 2: Kaggle Notebooks
1. Create new Kaggle notebook, enable **GPU T4 x2**
2. Add your model datasets as inputs
3. Upload `deploy_kaggle.ipynb` or paste the cells
4. Upload webapp files as a Kaggle dataset
5. Run all cells

### Option 3: HuggingFace Spaces (Docker)
1. Create new Space: https://huggingface.co/new-space
2. Choose **Docker** SDK, select **T4 small** (free with ZeroGPU)
3. Upload all files from `webapp/` folder
4. Set Secrets in Space settings:
   - `HF_TOKEN`: Your HuggingFace token
   - `ASR_MODEL_PATH`: `faysal314/whisper-md-lora-ep7-ct2`
   - `DIARIZEN_FINETUNED_PATH`: Path to your fine-tuned .pt file

### Option 4: Local (if you have GPU)
```bash
cd webapp
pip install -r requirements.txt
# Install DiariZen
git clone https://github.com/BUTSpeechFIT/DiariZen.git
pip install -e DiariZen/pyannote-audio
pip install -e DiariZen

# Set environment variables
export ASR_MODEL_PATH="faysal314/whisper-md-lora-ep7-ct2"
export HF_TOKEN="your_token"

# Run
python app.py
# Open http://localhost:7860
```

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| `ASR_MODEL_PATH` | `faysal314/whisper-md-lora-ep7-ct2` | HF repo ID or local path to CT2 model |
| `ASR_COMPUTE_TYPE` | `float16` | `float16` (GPU), `int8` (CPU), `int8_float16` |
| `ASR_DEVICE` | `auto` | `auto`, `cuda`, `cpu` |
| `DIARIZEN_REPO_ID` | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` | HF repo for DiariZen base |
| `DIARIZEN_FINETUNED_PATH` | (empty) | Path to fine-tuned .pt file |
| `HF_TOKEN` | (empty) | HuggingFace API token |
| `MAX_UPLOAD_SIZE_MB` | `500` | Max upload file size |

## Running without GPU
The app can run on CPU but will be significantly slower:
```bash
export ASR_COMPUTE_TYPE="int8"  # Use int8 quantization for CPU
export ASR_DEVICE="cpu"
```
Expect ~10-30x slower inference on CPU vs T4 GPU.

## Models Used
- **ASR**: Whisper Medium (Bengali fine-tuned with LoRA, CTranslate2 format)
- **Diarization**: DiariZen (WavLM-Large backbone, fine-tuned segmentation head)
