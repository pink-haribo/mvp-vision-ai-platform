"""
Image Tools API

Provides simple, training-free image processing tools:
- Super-resolution (2x, 4x upscaling)
- Background removal (coming soon)
- Image enhancement (coming soon)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
import subprocess
import json
import os
import shutil
import uuid
from typing import Optional

router = APIRouter(prefix="/image-tools", tags=["Image Tools"])


@router.post("/super-resolution")
async def super_resolution(
    image: UploadFile = File(...),
    scale: int = Query(2, ge=2, le=4, description="Upscale factor (2 or 4)"),
):
    """
    Upscale image using AI-based super-resolution.

    Supports 2x and 4x upscaling using Swin2SR models.

    Args:
        image: Image file to upscale
        scale: Upscale factor (2 or 4)

    Returns:
        JSON with result_url pointing to upscaled image
    """
    # Validate image type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )

    # Create temporary directory
    session_id = str(uuid.uuid4())
    from app.core.config import settings
    temp_dir = Path(settings.UPLOAD_DIR) / "image_tools_temp" / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded image
    image_path = temp_dir / f"input{Path(image.filename).suffix}"
    try:
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save image: {str(e)}"
        )

    try:
        # Determine model based on scale
        model_name = f"caidas/swin2SR-classical-sr-x{scale}-64"

        # Path to training venv python
        backend_dir = Path(__file__).parent.parent.parent
        project_root = backend_dir.parent
        training_venv_python = project_root / "training" / "venv" / "Scripts" / "python.exe"
        inference_script = project_root / "training" / "run_quick_inference.py"

        if not training_venv_python.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Training environment not found at {training_venv_python}"
            )

        if not inference_script.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Inference script not found at {inference_script}"
            )

        # Build command for super-resolution
        # We create a dummy training job ID (0) since this is standalone inference
        cmd = [
            str(training_venv_python),
            str(inference_script),
            "--training_job_id", "0",
            "--image_path", str(image_path),
            "--framework", "huggingface",
            "--model_name", model_name,
            "--task_type", "super_resolution",
            "--num_classes", "0",
            "--dataset_path", "",
            "--output_dir", str(temp_dir),
            "--use_pretrained"
        ]

        print(f"[INFO] Running super-resolution with model: {model_name}")

        # Run inference
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180  # 3 min timeout (allow model download)
        )

        if result.returncode != 0:
            error_msg = result.stderr or "Super-resolution failed"
            print(f"[ERROR] Super-resolution failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Super-resolution failed: {error_msg}"
            )

        # Parse result
        try:
            output_lines = result.stdout.strip().split('\n')
            json_line = output_lines[-1]
            result_dict = json.loads(json_line)

            # Get upscaled image path
            upscaled_path = result_dict.get('upscaled_image_path')
            if not upscaled_path or not Path(upscaled_path).exists():
                raise HTTPException(
                    status_code=500,
                    detail="Upscaled image not generated"
                )

            # Convert to URL
            filename = Path(upscaled_path).name
            result_url = f"/image-tools/result/{session_id}/{filename}"

            return {
                "status": "success",
                "result_url": result_url,
                "session_id": session_id,
                "scale": scale,
                "model": model_name
            }

        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse output: {result.stdout}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse result: {str(e)}"
            )

    except subprocess.TimeoutExpired:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail="Processing timeout (3 minutes)"
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] Super-resolution error: {e}")
        traceback.print_exc()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@router.post("/background-removal")
async def background_removal(
    image: UploadFile = File(...),
):
    """
    Remove background from image.

    Coming soon!

    Args:
        image: Image file

    Returns:
        JSON with result_url
    """
    raise HTTPException(
        status_code=501,
        detail="Background removal is coming soon!"
    )


@router.post("/enhancement")
async def image_enhancement(
    image: UploadFile = File(...),
    denoise: bool = Query(True, description="Apply denoising"),
    sharpen: bool = Query(True, description="Apply sharpening"),
):
    """
    Enhance image quality.

    Coming soon!

    Args:
        image: Image file
        denoise: Apply denoising filter
        sharpen: Apply sharpening filter

    Returns:
        JSON with result_url
    """
    raise HTTPException(
        status_code=501,
        detail="Image enhancement is coming soon!"
    )


@router.get("/result/{session_id}/{filename}")
async def get_result_image(
    session_id: str,
    filename: str
):
    """
    Serve processed result images.

    Args:
        session_id: Session ID
        filename: Image filename

    Returns:
        FileResponse with the image
    """
    from app.core.config import settings

    # Try multiple possible locations
    # 1. Direct in session dir (for future tools)
    image_path = Path(settings.UPLOAD_DIR) / "image_tools_temp" / session_id / filename

    # 2. In inference_results subdirectory (for super-resolution)
    if not image_path.exists():
        image_path = Path(settings.UPLOAD_DIR) / "image_tools_temp" / session_id / "inference_results" / filename

    if not image_path.exists():
        print(f"[ERROR] Image not found at: {image_path}")
        raise HTTPException(
            status_code=404,
            detail=f"Result image not found: {filename}"
        )

    # Verify file is within allowed directory (security check)
    try:
        image_path.resolve().relative_to(Path(settings.UPLOAD_DIR).resolve())
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )

    return FileResponse(
        path=image_path,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*"
        }
    )


@router.delete("/session/{session_id}")
async def cleanup_session(session_id: str):
    """
    Clean up session files.

    Args:
        session_id: Session ID to clean up

    Returns:
        Status message
    """
    try:
        from app.core.config import settings
        session_dir = Path(settings.UPLOAD_DIR) / "image_tools_temp" / session_id

        if session_dir.exists() and session_dir.is_dir():
            # Safety check
            if "image_tools_temp" in str(session_dir):
                shutil.rmtree(session_dir)
                print(f"[INFO] Cleaned up session: {session_id}")
                return {"status": "cleaned", "session_id": session_id}
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid session directory"
                )
        else:
            return {"status": "not_found", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] Cleanup failed for session {session_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )
