# Bengali ASR & Speaker Diarization Web App

A web application for automatic speech recognition (ASR) and speaker diarization of Bengali audio/video. Upload a file, get speaker-labeled Bengali subtitles with a synced media player — entirely free to run on Kaggle with GPU.
We mainly fine-tuned and developed the whole architecture for BUET DL Sprint 4.0. We were one of the finalist of the competition where our Diarization model placed 2nd on the private leaderboard. We build the web application just to test our model and appreciate it.

---


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

## DEMO 
<img width="1017" height="480" alt="Image" src="https://github.com/user-attachments/assets/dda70ad6-3d3f-49bd-973f-8be0b47456e6" />
<img width="878" height="707" alt="Image" src="https://github.com/user-attachments/assets/8ec9557c-f970-46a1-8ea8-e1e9efe82617" />
<img width="904" height="641" alt="Image" src="https://github.com/user-attachments/assets/ea1944f9-c04d-43b9-8e2d-13697f28f0d1" />



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
