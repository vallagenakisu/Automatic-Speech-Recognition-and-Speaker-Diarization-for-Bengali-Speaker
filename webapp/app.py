"""
Bengali ASR + Speaker Diarization Web Application
FastAPI Backend
"""
import uuid
import asyncio
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config import (
    UPLOAD_DIR, RESULTS_DIR, MAX_UPLOAD_SIZE_MB, ALLOWED_EXTENSIONS,
)

app = FastAPI(title="Bengali ASR & Speaker Diarization", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── In-memory job tracking ──
jobs: dict = {}


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload audio/video file and start processing."""
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    # Generate job ID
    job_id = uuid.uuid4().hex[:12]

    # Save uploaded file
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = f"input{ext}"
    file_path = job_upload_dir / safe_filename

    # Stream-save with size check
    size = 0
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                # Clean up and reject
                f.close()
                shutil.rmtree(job_upload_dir, ignore_errors=True)
                raise HTTPException(413, f"File too large. Max: {MAX_UPLOAD_SIZE_MB}MB")
            f.write(chunk)

    # Initialize job status
    jobs[job_id] = {
        "status": "queued",
        "progress": "Upload complete. Processing queued...",
        "file_path": str(file_path),
        "original_filename": file.filename,
        "result": None,
        "error": None,
    }

    # Start processing in background
    background_tasks.add_task(_process_job, job_id)

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get processing status for a job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
    }
    if job["status"] == "error":
        response["error"] = job["error"]
    return response


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """Get processing results (subtitle segments)."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, f"Job not ready. Status: {job['status']}")
    result = job["result"]
    return {
        "job_id": job_id,
        "original_filename": job["original_filename"],
        "segments": result["segments"],
        "num_segments": result["num_segments"],
        "num_speakers": result["num_speakers"],
    }


@app.get("/api/media/{job_id}")
async def get_media(job_id: str):
    """Stream the uploaded media file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    file_path = jobs[job_id]["file_path"]
    if not Path(file_path).exists():
        raise HTTPException(404, "Media file not found")
    return FileResponse(file_path, filename=Path(file_path).name)


@app.get("/api/download/{job_id}")
async def download_subtitle(job_id: str, format: str = "srt"):
    """Download subtitle file in SRT or VTT format."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Job not ready")

    result = job["result"]
    if format == "vtt":
        path = result["vtt_path"]
        media_type = "text/vtt"
    else:
        path = result["srt_path"]
        media_type = "application/x-subrip"

    if not Path(path).exists():
        raise HTTPException(404, "Subtitle file not found")

    stem = Path(job["original_filename"]).stem
    return FileResponse(path, filename=f"{stem}.{format}", media_type=media_type)


@app.get("/api/subtitle/{job_id}")
async def get_subtitle_vtt(job_id: str):
    """Get VTT subtitle for the HTML5 video/audio player <track> element."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Job not ready")
    vtt_path = job["result"]["vtt_path"]
    if not Path(vtt_path).exists():
        raise HTTPException(404, "VTT file not found")
    return FileResponse(vtt_path, media_type="text/vtt")


# =============================================================================
# Background processing
# =============================================================================

def _process_job(job_id: str):
    """Run the full processing pipeline for a job."""
    job = jobs[job_id]
    job["status"] = "processing"

    def update_progress(msg: str):
        job["progress"] = msg

    try:
        from pipeline import process_media
        result = process_media(
            file_path=job["file_path"],
            job_id=job_id,
            results_dir=str(RESULTS_DIR),
            progress_callback=update_progress,
        )
        job["result"] = result
        job["status"] = "done"
        job["progress"] = "Processing complete!"
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] Job {job_id}: {tb}")
        job["status"] = "error"
        job["error"] = str(e)
        job["progress"] = f"Error: {e}"


# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup():
    print("=" * 60)
    print("Bengali ASR + Speaker Diarization Web App")
    print("=" * 60)
    print(f"  Upload dir:  {UPLOAD_DIR}")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  GPU:         {('Yes' if __import__('torch').cuda.is_available() else 'No')}")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
