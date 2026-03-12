"""
Bengali ASR + Speaker Diarization Processing Pipeline
Handles audio preprocessing, transcription, diarization, and subtitle generation.
"""
import gc
import os
import time
import warnings
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
import soundfile as sf

from config import (
    ASR_MODEL_PATH, ASR_COMPUTE_TYPE, ASR_DEVICE, ASR_BATCH_SIZE,
    DIARIZEN_REPO_ID, DIARIZEN_FINETUNED_PATH, HF_TOKEN,
    SAMPLE_RATE, DIARIZATION_POST_CONFIG,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── NumPy 2.0 compatibility ──
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "NAN"):
    np.NAN = np.nan

# ── torchaudio compatibility ──
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []

# ── PyTorch 2.6+ weights_only fix ──
if not getattr(torch.load, "_patched", False):
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)
    _patched_load._patched = True
    torch.load = _patched_load

# ── HuggingFace auth ──
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception:
        pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Singleton model holders
# =============================================================================
_asr_model = None
_diar_pipeline = None


def get_asr_model():
    """Lazy-load faster-whisper model."""
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    from faster_whisper import WhisperModel, BatchedInferencePipeline

    device = ASR_DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[ASR] Loading model from {ASR_MODEL_PATH} on {device} ({ASR_COMPUTE_TYPE})...")
    raw = WhisperModel(
        ASR_MODEL_PATH,
        device=device,
        compute_type=ASR_COMPUTE_TYPE,
    )
    _asr_model = BatchedInferencePipeline(model=raw)
    print("[ASR] Model loaded.")
    return _asr_model


def get_diar_pipeline():
    """Lazy-load DiariZen diarization pipeline."""
    global _diar_pipeline
    if _diar_pipeline is not None:
        return _diar_pipeline

    from pyannote.audio.utils.powerset import Powerset
    from diarizen.pipelines.inference import DiariZenPipeline
    import torch.nn as nn

    print(f"[DIAR] Loading pipeline from {DIARIZEN_REPO_ID}...")
    pipeline = DiariZenPipeline.from_pretrained(DIARIZEN_REPO_ID)
    print("[DIAR] Base pipeline loaded.")

    # Load fine-tuned segmentation weights if provided
    if DIARIZEN_FINETUNED_PATH and os.path.exists(DIARIZEN_FINETUNED_PATH):
        print(f"[DIAR] Loading fine-tuned weights from {DIARIZEN_FINETUNED_PATH}...")
        if hasattr(pipeline._segmentation, "model_"):
            raw_model = pipeline._segmentation.model_
        elif hasattr(pipeline._segmentation, "model"):
            raw_model = pipeline._segmentation.model
        else:
            raw_model = pipeline._segmentation

        state_dict = torch.load(DIARIZEN_FINETUNED_PATH, map_location=DEVICE)

        # Auto-detect output size and resize classifier
        ckpt_out = None
        for key in state_dict:
            if "classifier" in key and "weight" in key:
                ckpt_out = state_dict[key].shape[0]
                break

        if ckpt_out is not None:
            matched = False
            for max_spk in range(2, 15):
                for max_spf in range(1, max_spk + 1):
                    ps = Powerset(num_classes=max_spk, max_set_size=max_spf)
                    if ps.num_powerset_classes == ckpt_out:
                        raw_model.powerset = ps
                        matched = True
                        matched_max_spk, matched_max_spf = max_spk, max_spf
                        break
                if matched:
                    break

            if not matched:
                raw_model.powerset = Powerset(num_classes=7, max_set_size=2)
                matched_max_spk, matched_max_spf = 7, 2

            # Patch decoder specifications
            specs_holder = raw_model
            if hasattr(specs_holder, "specifications"):
                specs = specs_holder.specifications
                if hasattr(specs, "powerset_max_classes"):
                    specs.powerset_max_classes = ckpt_out
                if hasattr(specs, "classes"):
                    new_classes = [str(i) for i in range(matched_max_spk)]
                    if isinstance(specs.classes, list) and specs.classes and isinstance(specs.classes[0], list):
                        specs.classes = [new_classes]
                    else:
                        specs.classes = new_classes

            # Replace cached powerset converter
            new_conversion = Powerset(matched_max_spk, matched_max_spf).to(DEVICE)
            pipeline._segmentation.conversion = new_conversion

            # Resize classifier if needed
            if hasattr(raw_model, "classifier"):
                cls = raw_model.classifier
                if isinstance(cls, nn.Linear) and cls.out_features != ckpt_out:
                    raw_model.classifier = nn.Linear(cls.in_features, ckpt_out).to(DEVICE)
                elif isinstance(cls, nn.Sequential):
                    last = cls[-1]
                    if last.out_features != ckpt_out:
                        cls[-1] = nn.Linear(last.in_features, ckpt_out).to(DEVICE)

        raw_model.load_state_dict(state_dict, strict=False)
        print("[DIAR] Fine-tuned weights loaded.")

    pipeline._segmentation.batch_size = 4
    _diar_pipeline = pipeline
    print("[DIAR] Pipeline ready.")
    return _diar_pipeline


# =============================================================================
# Audio utilities
# =============================================================================

def extract_audio_from_video(video_path: str, output_path: str) -> str:
    """Extract audio from video using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-y", output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def load_audio(path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load audio, convert to mono float32, resample."""
    waveform, orig_sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy().astype(np.float32), sr


def denoise_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Adaptive spectral gating noise reduction."""
    import noisereduce as nr
    rms = np.sqrt(np.mean(audio ** 2))
    noise_est = np.percentile(np.abs(audio), 10)
    snr = 20 * np.log10(rms / (noise_est + 1e-8))
    if snr < 20:
        audio = nr.reduce_noise(
            y=audio, sr=sr, stationary=False,
            prop_decrease=0.85, n_fft=2048, hop_length=512,
        )
    return audio.astype(np.float32)


# =============================================================================
# VAD (Voice Activity Detection)
# =============================================================================
_vad_model = None


def _get_vad():
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad()
    return _vad_model


def run_vad(audio: np.ndarray, sr: int = SAMPLE_RATE) -> List[Dict[str, float]]:
    """Detect speech regions using Silero VAD."""
    from silero_vad import get_speech_timestamps
    model = _get_vad()
    tensor = torch.from_numpy(audio).float()
    return get_speech_timestamps(
        tensor, model,
        threshold=0.40,
        min_speech_duration_ms=250,
        min_silence_duration_ms=200,
        max_speech_duration_s=20.0,
        speech_pad_ms=400,
        window_size_samples=512,
        sampling_rate=sr,
        return_seconds=True,
    )


# =============================================================================
# ASR Transcription
# =============================================================================

def transcribe_audio(audio_path: str, progress_callback=None) -> List[Dict]:
    """
    Transcribe audio using faster-whisper.
    Returns list of {"start": float, "end": float, "text": str}
    """
    model = get_asr_model()

    if progress_callback:
        progress_callback("Transcribing audio...")

    segments, info = model.transcribe(
        audio_path,
        language="bn",
        beam_size=5,
        batch_size=ASR_BATCH_SIZE,
        word_timestamps=True,
        no_speech_threshold=0.9,
        log_prob_threshold=-2.0,
        compression_ratio_threshold=3.0,
    )

    results = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            words = [
                {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                for w in (seg.words or [])
            ]
            results.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": text,
                "words": words,
            })

    # Post-process text
    results = _postprocess_transcription(results)

    if progress_callback:
        progress_callback(f"Transcription complete: {len(results)} segments")

    return results


def _postprocess_transcription(segments: List[Dict]) -> List[Dict]:
    """Clean up Bengali transcription."""
    import unicodedata
    try:
        from bnunicodenormalizer import Normalizer
        bn_norm = Normalizer()
    except ImportError:
        bn_norm = None

    cleaned = []
    for seg in segments:
        text = seg["text"]
        # Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)
        # Remove zero-width chars
        text = text.replace("\u200c", "").replace("\u200d", "").replace("\u200b", "")
        # Bengali unicode normalization (normalize word-by-word to avoid multi-word warning)
        if bn_norm:
            words = text.split(" ")
            normalized_words = []
            for word in words:
                if word:
                    result = bn_norm(word)
                    normalized_words.append(result["normalized"] if result and result["normalized"] else word)
            text = " ".join(normalized_words)
        text = text.strip()
        if text:
            cleaned.append({**seg, "text": text})
    return cleaned


# =============================================================================
# Speaker Diarization
# =============================================================================

def diarize_audio(audio_path: str, progress_callback=None) -> List[Dict]:
    """
    Run speaker diarization.
    Returns list of {"start": float, "end": float, "speaker": str}
    """
    pipeline = get_diar_pipeline()

    if progress_callback:
        progress_callback("Running speaker diarization...")

    annotation = pipeline(audio_path)

    # Convert to segments
    from pyannote.core import Annotation, Segment
    raw_segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        raw_segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    # Apply post-processing
    segments = _postprocess_diarization(raw_segments)

    if progress_callback:
        progress_callback(f"Diarization complete: {len(segments)} segments")

    return segments


def _postprocess_diarization(segments: List[Dict]) -> List[Dict]:
    """Apply post-processing pipeline to diarization segments."""
    cfg = DIARIZATION_POST_CONFIG

    # Stitch close segments of same speaker
    segments = _stitch_segments(segments, gap_threshold=cfg["stitch_gap_threshold"])
    # Filter short segments
    segments = [s for s in segments if (s["end"] - s["start"]) >= cfg["min_segment_duration"]]
    # Reassign isolated segments
    segments = _reassign_segments(segments, cfg["reassign_window_size"], cfg["reassign_agreement_threshold"])
    # Filter backchannel
    segments = _filter_backchannel(segments, cfg["backchannel_min_duration"], cfg["backchannel_context_window"])
    # Final merge
    segments = _stitch_segments(segments, gap_threshold=cfg["final_merge_gap"])

    # Rename speakers to sequential SPEAKER_1, SPEAKER_2, etc.
    speaker_map = {}
    counter = 1
    for seg in segments:
        if seg["speaker"] not in speaker_map:
            speaker_map[seg["speaker"]] = f"SPEAKER_{counter}"
            counter += 1
        seg["speaker"] = speaker_map[seg["speaker"]]

    return segments


def _stitch_segments(segments: List[Dict], gap_threshold: float) -> List[Dict]:
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda x: x["start"])
    result = [sorted_segs[0].copy()]
    for seg in sorted_segs[1:]:
        prev = result[-1]
        if seg["speaker"] == prev["speaker"] and (seg["start"] - prev["end"]) < gap_threshold:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            result.append(seg.copy())
    return result


def _reassign_segments(segments: List[Dict], window: int, threshold: float) -> List[Dict]:
    if len(segments) < 3:
        return segments
    result = [s.copy() for s in segments]
    for i in range(len(result)):
        left = [result[j]["speaker"] for j in range(max(0, i - window), i)]
        right = [result[j]["speaker"] for j in range(i + 1, min(len(result), i + window + 1))]
        neighbors = left + right
        if not neighbors:
            continue
        counts = Counter(neighbors)
        best, count = counts.most_common(1)[0]
        if (result[i]["speaker"] != best and count / len(neighbors) >= threshold
                and (result[i]["end"] - result[i]["start"]) < 2.0):
            result[i]["speaker"] = best
    return result


def _filter_backchannel(segments: List[Dict], min_dur: float, context_window: float) -> List[Dict]:
    if not segments:
        return segments
    result = []
    for i, seg in enumerate(segments):
        dur = seg["end"] - seg["start"]
        if dur >= min_dur:
            result.append(seg)
            continue
        is_bc = False
        if 0 < i < len(segments) - 1:
            prev, nxt = segments[i - 1], segments[i + 1]
            if (prev["speaker"] != seg["speaker"] and nxt["speaker"] == prev["speaker"]
                    and (seg["start"] - prev["end"]) < context_window
                    and (nxt["start"] - seg["end"]) < context_window):
                is_bc = True
        if not is_bc:
            result.append(seg)
    return result


# =============================================================================
# Merge ASR + Diarization → Speaker-labeled subtitles
# =============================================================================

def merge_asr_diarization(
    asr_segments: List[Dict],
    diar_segments: List[Dict],
) -> List[Dict]:
    """
    Merge ASR transcription with diarization to produce speaker-labeled subtitles.
    Returns list of {"start": float, "end": float, "text": str, "speaker": str}
    """
    if not diar_segments:
        # No diarization — return ASR segments with unknown speaker
        return [{"speaker": "SPEAKER_1", **seg} for seg in asr_segments]

    result = []
    for asr_seg in asr_segments:
        # Find the diarization segment with maximum overlap
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for diar_seg in diar_segments:
            overlap_start = max(asr_seg["start"], diar_seg["start"])
            overlap_end = min(asr_seg["end"], diar_seg["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg["speaker"]

        result.append({
            "start": asr_seg["start"],
            "end": asr_seg["end"],
            "text": asr_seg["text"],
            "speaker": best_speaker,
            "words": asr_seg.get("words", []),
        })

    return result


# =============================================================================
# Segment splitting for readable subtitles
# =============================================================================

MAX_SUBTITLE_DURATION = 5.0  # seconds


def _split_long_segments(segments: List[Dict], max_duration: float = MAX_SUBTITLE_DURATION) -> List[Dict]:
    """
    Split segments longer than max_duration into shorter chunks using word timestamps.
    Strips 'words' from output so the API response stays clean.
    """
    result = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        words = seg.get("words", [])

        if duration <= max_duration or len(words) < 2:
            result.append({k: v for k, v in seg.items() if k != "words"})
            continue

        current_words = []
        chunk_start = seg["start"]

        for word in words:
            current_words.append(word)
            if word["end"] - chunk_start >= max_duration:
                text = " ".join(w["word"].strip() for w in current_words if w["word"].strip())
                if text:
                    result.append({
                        "start": chunk_start,
                        "end": word["end"],
                        "text": text,
                        "speaker": seg.get("speaker", "UNKNOWN"),
                    })
                chunk_start = word["end"]
                current_words = []

        # Flush remaining words
        if current_words:
            text = " ".join(w["word"].strip() for w in current_words if w["word"].strip())
            if text:
                result.append({
                    "start": chunk_start,
                    "end": seg["end"],
                    "text": text,
                    "speaker": seg.get("speaker", "UNKNOWN"),
                })

    return result


# =============================================================================
# Subtitle format generation
# =============================================================================

def generate_srt(segments: List[Dict]) -> str:
    """Generate SRT subtitle content."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_time_srt(seg["start"])
        end = _format_time_srt(seg["end"])
        speaker = seg.get("speaker", "")
        text = seg["text"]
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def generate_vtt(segments: List[Dict]) -> str:
    """Generate WebVTT subtitle content."""
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        start = _format_time_vtt(seg["start"])
        end = _format_time_vtt(seg["end"])
        speaker = seg.get("speaker", "")
        text = seg["text"]
        if speaker:
            text = f"<v {speaker}>{text}"
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _format_time_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_time_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# =============================================================================
# Main processing function
# =============================================================================

def process_media(
    file_path: str,
    job_id: str,
    results_dir: str,
    progress_callback=None,
) -> Dict:
    """
    Full pipeline: audio extraction → ASR → diarization → merge → subtitles.
    Returns dict with results metadata.
    """
    results_path = Path(results_dir) / job_id
    results_path.mkdir(parents=True, exist_ok=True)

    ext = Path(file_path).suffix.lower()
    is_video = ext in {".mp4", ".mkv", ".avi", ".mov", ".webm"}

    # Step 1: Extract audio if video
    if is_video:
        if progress_callback:
            progress_callback("Extracting audio from video...")
        audio_path = str(results_path / "audio.wav")
        extract_audio_from_video(file_path, audio_path)
    else:
        audio_path = file_path

    # Step 2: ASR transcription
    if progress_callback:
        progress_callback("Starting transcription...")
    asr_segments = transcribe_audio(audio_path, progress_callback)

    # Step 3: Speaker diarization
    if progress_callback:
        progress_callback("Starting diarization...")
    try:
        diar_segments = diarize_audio(audio_path, progress_callback)
    except Exception as e:
        print(f"[WARN] Diarization failed: {e}. Proceeding with ASR only.")
        diar_segments = []

    # Step 4: Merge
    if progress_callback:
        progress_callback("Merging transcription with speaker labels...")
    merged = merge_asr_diarization(asr_segments, diar_segments)

    # Step 4.5: Split long segments into readable subtitle chunks
    merged = _split_long_segments(merged)

    # Step 5: Generate subtitle files
    srt_content = generate_srt(merged)
    vtt_content = generate_vtt(merged)

    srt_path = results_path / "subtitles.srt"
    vtt_path = results_path / "subtitles.vtt"
    srt_path.write_text(srt_content, encoding="utf-8")
    vtt_path.write_text(vtt_content, encoding="utf-8")

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "job_id": job_id,
        "segments": merged,
        "num_segments": len(merged),
        "num_speakers": len(set(s.get("speaker", "") for s in merged)),
        "srt_path": str(srt_path),
        "vtt_path": str(vtt_path),
        "audio_path": audio_path,
    }
