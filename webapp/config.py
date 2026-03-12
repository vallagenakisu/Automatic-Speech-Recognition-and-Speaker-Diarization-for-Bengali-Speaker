"""
Bengali ASR + Speaker Diarization Web Application
Configuration settings
"""
import os
from pathlib import Path

# ── Base directories ──
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── Model paths (configurable via environment variables) ──
# ASR: faster-whisper CTranslate2 model
# Set ASR_MODEL_PATH to a local directory with CT2 model, or a HuggingFace repo ID
ASR_MODEL_PATH = os.environ.get(
    "ASR_MODEL_PATH",
    "faysal314/whisper-md-lora-ep7-ct2"  # HuggingFace repo ID (default)
)
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "float16")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "auto")  # "auto", "cuda", "cpu"
ASR_BATCH_SIZE = int(os.environ.get("ASR_BATCH_SIZE", "16"))

# Diarization: DiariZen pipeline
DIARIZEN_REPO_ID = os.environ.get(
    "DIARIZEN_REPO_ID",
    "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
)
DIARIZEN_FINETUNED_PATH = os.environ.get("DIARIZEN_FINETUNED_PATH", "")

# HuggingFace token (for gated models)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── Processing settings ──
MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "500"))
SAMPLE_RATE = 16000
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# ── Diarization post-processing (Optuna-optimized) ──
DIARIZATION_POST_CONFIG = {
    "stitch_gap_threshold": 3.236,
    "min_segment_duration": 0.781,
    "reassign_window_size": 10,
    "reassign_agreement_threshold": 0.828,
    "backchannel_min_duration": 0.870,
    "backchannel_context_window": 4.618,
    "final_merge_gap": 1.234,
}
