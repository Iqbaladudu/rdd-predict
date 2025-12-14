import shutil
import uuid
import os
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from PIL.Image import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from ultralytics import YOLO
from utils.boto import upload_file
from utils.cloudinary_uploader import upload_to_cloudinary

# Set device - use GPU if available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files for serving results
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
# This will download yolov8n.pt if not present on first use
try:
    model = YOLO("YOLOv8_Small_RDD.pt") 
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback or re-raise
    raise

@app.get("/")
def read_root():
    return {"message": "RDD Predict API is running"}

@app.post("/predict")
async def predict_media(file: UploadFile = File(...)):
    # Generate unique ID for this request
    result_type = None
    request_id = str(uuid.uuid4())
    filename = file.filename
    # Handle cases where filename might be empty or missing extension
    if not filename or "." not in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    extension = filename.split(".")[-1].lower()
    
    # Paths
    input_path = Path(f"uploads/{request_id}.{extension}")
    output_filename = f"{request_id}_processed.{extension}"
    output_path = Path(f"static/{output_filename}")
    
    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    results_data = []
    video_metadata = {}
    
    try:
        if extension in ["jpg", "jpeg", "png", "bmp", "webp"]:
            result_type = "image"
            # Process Image
            results = model(input_path, device=DEVICE)
            result = results[0]
            
            # Save annotated image
            annotated_frame = result.plot()
            cv2.imwrite(str(output_path), annotated_frame)
            
            # Upload to S3
            s3_url = upload_file(str(output_path), output_filename)
            
            # Upload to Cloudinary
            cloudinary_result = upload_to_cloudinary(str(output_path))
            
            # Extract data
            for box in result.boxes:
                c = int(box.cls)
                class_name = model.names[c]
                conf = float(box.conf)
                xyxy = box.xyxy.tolist()[0]
                results_data.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": xyxy
                })
                
        elif extension in ["mp4", "avi", "mov", "mkv", "webm"]:
            result_type = "video"
            # Process Video
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open video file")
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            video_metadata = {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames
            }
            
            # Use mp4v for compatibility
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_idx = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                    
                # Run YOLO
                results = model(frame, device=DEVICE)
                result = results[0]
                
                # Write frame with annotations
                annotated_frame = result.plot()
                out.write(annotated_frame)
                
                # Extract data
                frame_detections = []
                for box in result.boxes:
                    c = int(box.cls)
                    class_name = model.names[c]
                    conf = float(box.conf)
                    xyxy = box.xyxy.tolist()[0]
                    frame_detections.append({
                        "class": class_name,
                        "confidence": conf,
                        "bbox": xyxy
                    })
                
                if frame_detections:
                    # Save frame as image and upload to Cloudinary
                    frame_filename = f"{request_id}_frame_{frame_idx}.jpg"
                    frame_path = Path(f"static/{frame_filename}")
                    cv2.imwrite(str(frame_path), annotated_frame)

                    # Upload frame to Cloudinary
                    frame_cloudinary_result = upload_to_cloudinary(str(frame_path))

                    # Clean up the frame file to save space
                    if os.path.exists(frame_path):
                        os.remove(frame_path)

                    results_data.append({
                        "frame": frame_idx,
                        "timestamp": frame_idx / fps if fps > 0 else 0,
                        "detections": frame_detections,
                        "frame_url": frame_cloudinary_result.get("url"),
                        "frame_public_id": frame_cloudinary_result.get("public_id")
                    })
                
                frame_idx += 1
                
            cap.release()
            out.release()

            # Upload processed video to Cloudinary
            cloudinary_result = upload_to_cloudinary(str(output_path))
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use image (jpg, png) or video (mp4, avi).")
            
    except Exception as e:
        # Log error?
        print(f"Error processing: {e}")
        # Cleanup output if it exists and is partial?
        if os.path.exists(output_path):
             os.remove(output_path)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)

    return {
        "status": "success",
        "file_url": f"/static/{output_filename}",
        result_type: result_type,
        # "s3_url": s3_url,
        "cloudinary_url": cloudinary_result.get("url"),
        "cloudinary_public_id": cloudinary_result.get("public_id"),
        "filename": output_filename,
        "metadata": video_metadata if video_metadata else {"type": "image"},
        "data_summary": f"Found {len(results_data)} frames/items with detections",
        "data": results_data
    }