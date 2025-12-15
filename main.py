import shutil
import uuid
import os
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from PIL.Image import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from ultralytics import YOLO
from utils.boto import upload_file
from utils.cloudinary_uploader import upload_to_cloudinary
from utils.stream_utils import encode_frame_to_base64, decode_base64_to_frame
import requests
import time

# Device detection - prioritize GPU if available
def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"   CUDA version: {torch.version.cuda}")
        return device
    else:
        print("⚠️  No GPU detected, using CPU")
        print("   TensorRT models will NOT be available")
        return 'cpu'

DEVICE = get_device()
HAS_GPU = DEVICE == 'cuda'
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

# Model configurations
# Format: {endpoint_suffix: {"url": url, "local_path": path, "description": desc, "requires_gpu": bool}}

# Check TensorFlow availability for TFLite models
try:
    import tensorflow
    HAS_TENSORFLOW = True
    print(f"✅ TensorFlow available: {tensorflow.__version__}")
except ImportError:
    HAS_TENSORFLOW = False
    print("⚠️  TensorFlow not installed - TFLite models will NOT be available")

MODEL_CONFIGS = {
    "tfrt-32": {
        "url": "https://pub-0ccce103f38e4902912534cdb3973783.r2.dev/YOLOv8_Small_RDD_float32.engine",
        "local_path": "models/YOLOv8_Small_RDD_float32.engine",
        "description": "TensorRT Float32",
        "requires_gpu": True,
        "requires_tensorflow": False
    },
    "tfrt-16": {
        "url": "https://pub-0ccce103f38e4902912534cdb3973783.r2.dev/YOLOv8_Small_RDD_float16.engine",
        "local_path": "models/YOLOv8_Small_RDD_float16.engine",
        "description": "TensorRT Float16",
        "requires_gpu": True,
        "requires_tensorflow": False
    },
    "tflite-32": {
        "url": "https://pub-0ccce103f38e4902912534cdb3973783.r2.dev/YOLOv8_Small_RDD_float32.tflite",
        "local_path": "models/YOLOv8_Small_RDD_float32.tflite",
        "description": "TFLite Float32",
        "requires_gpu": False,
        "requires_tensorflow": True
    },
    "tflite-16": {
        "url": "https://pub-0ccce103f38e4902912534cdb3973783.r2.dev/YOLOv8_Small_RDD_float16.tflite",
        "local_path": "models/YOLOv8_Small_RDD_float16.tflite",
        "description": "TFLite Float16",
        "requires_gpu": False,
        "requires_tensorflow": True
    },
    "pytorch": {
        "url": None,  # Local file
        "local_path": "YOLOv8_Small_RDD.pt",
        "description": "PyTorch Original",
        "requires_gpu": False,  # Can run on CPU or GPU
        "requires_tensorflow": False
    }
}

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Fungsi untuk download dengan progres bar
def download_file(url: str, filename: str) -> bool:
    """Download file with progress bar. Returns True if successful."""
    if os.path.exists(filename):
        print(f"[Download] {filename} sudah ada, skip download.")
        return True
    
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded_size = 0

            print(f"[Download] Mengunduh {filename} dari {url}...")
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                        print(f"\r[Download] Progres: {progress:.2f}% ({downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)", end="")
            print("\n[Download] Download selesai!")
            return True
    except Exception as e:
        print(f"\n[Download] Error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

# Dictionary to store loaded models
models: dict = {}

def load_all_models():
    """Download and load all models at startup."""
    print("\n" + "="*60)
    print("LOADING ALL MODELS")
    print("="*60 + "\n")
    
    for model_key, config in MODEL_CONFIGS.items():
        url = config["url"]
        local_path = config["local_path"]
        description = config["description"]
        requires_gpu = config.get("requires_gpu", False)
        requires_tensorflow = config.get("requires_tensorflow", False)
        
        print(f"\n[Model] Loading {model_key} ({description})...")
        
        # Skip GPU-only models if no GPU available
        if requires_gpu and not HAS_GPU:
            print(f"[Model] SKIP: {model_key} requires GPU (TensorRT)")
            continue
        
        # Skip TFLite models if TensorFlow not available
        if requires_tensorflow and not HAS_TENSORFLOW:
            print(f"[Model] SKIP: {model_key} requires TensorFlow (TFLite)")
            continue
        
        # Download if URL is provided and file doesn't exist
        if url and not os.path.exists(local_path):
            success = download_file(url, local_path)
            if not success:
                print(f"[Model] SKIP: Failed to download {model_key}")
                continue
        
        # Check if local file exists
        if not os.path.exists(local_path):
            print(f"[Model] SKIP: {local_path} not found")
            continue
        
        # Load model
        try:
            model = YOLO(local_path)
            models[model_key] = model
            print(f"[Model] SUCCESS: {model_key} loaded from {local_path}")
        except Exception as e:
            print(f"[Model] ERROR: Failed to load {model_key}: {e}")
    
    print("\n" + "="*60)
    print(f"LOADED MODELS: {list(models.keys())}")
    if HAS_GPU:
        print(f"GPU READY: TensorRT models available for acceleration")
    print("="*60 + "\n")

# Load all models at startup
load_all_models()

# Default model untuk backward compatibility
model = models.get("pytorch") or (list(models.values())[0] if models else None)
if model is None:
    raise RuntimeError("No models could be loaded!")

@app.get("/")
def read_root():
    return {"message": "RDD Predict API is running"}

@app.get("/ping")
async def health_check():
    return {"status": "healthy"}


@app.get("/models")
async def list_models():
    """List all loaded models and their endpoints."""
    model_list = []
    for model_key in models.keys():
        config = MODEL_CONFIGS.get(model_key, {})
        model_list.append({
            "key": model_key,
            "description": config.get("description", "Unknown"),
            "stream_endpoint": f"/predict/stream/{model_key}",
            "requires_gpu": config.get("requires_gpu", False),
            "loaded": True
        })
    return {
        "device": DEVICE,
        "has_gpu": HAS_GPU,
        "loaded_models": model_list,
        "total_loaded": len(model_list),
        "default_model": "pytorch" if "pytorch" in models else (list(models.keys())[0] if models else None)
    }


async def handle_stream_prediction(websocket: WebSocket, model_key: str, selected_model):
    """
    Shared handler for real-time video streaming using WebSocket.
    
    Protocol:
    - Client sends: Base64 encoded JPEG frame
    - Server returns: JSON with processed frame and detections
    
    Response format:
    {
        "status": "success",
        "model": str,
        "frame_index": int,
        "timestamp_ms": int,
        "processing_latency_ms": float,
        "processed_frame": "data:image/jpeg;base64,...",
        "detections": [...],
        "detection_count": int
    }
    """
    await websocket.accept()
    frame_index = 0
    
    print(f"[Stream:{model_key}] Client connected")
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            start_time = time.time()
            
            try:
                # Decode base64 frame
                frame = decode_base64_to_frame(data)
                
                # Run YOLO prediction with selected model
                results = selected_model(frame, device=DEVICE, verbose=False)
                result = results[0]
                
                # Get annotated frame
                annotated_frame = result.plot()
                
                # Encode processed frame to base64
                processed_frame_b64 = encode_frame_to_base64(annotated_frame)
                
                # Extract detection data
                detections = []
                for box in result.boxes:
                    c = int(box.cls)
                    class_name = selected_model.names[c]
                    conf = float(box.conf)
                    xyxy = box.xyxy.tolist()[0]
                    detections.append({
                        "class": class_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(x, 2) for x in xyxy]
                    })
                
                # Calculate processing latency
                processing_latency = (time.time() - start_time) * 1000
                
                # Send response
                response = {
                    "status": "success",
                    "model": model_key,
                    "frame_index": frame_index,
                    "timestamp_ms": int(time.time() * 1000),
                    "processing_latency_ms": round(processing_latency, 2),
                    "processed_frame": processed_frame_b64,
                    "detections": detections,
                    "detection_count": len(detections)
                }
                
                # Log to terminal
                print(f"[Stream:{model_key}] Frame {frame_index:04d} | "
                      f"Detections: {len(detections):2d} | "
                      f"Latency: {processing_latency:6.2f}ms | "
                      f"Classes: {[d['class'] for d in detections]}")
                
                await websocket.send_json(response)
                frame_index += 1
                
            except ValueError as e:
                # Invalid frame data
                await websocket.send_json({
                    "status": "error",
                    "model": model_key,
                    "frame_index": frame_index,
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        print(f"[Stream:{model_key}] WebSocket disconnected after {frame_index} frames")
    except Exception as e:
        print(f"[Stream:{model_key}] WebSocket error: {e}")
        try:
            await websocket.send_json({
                "status": "error",
                "model": model_key,
                "error": str(e)
            })
        except:
            pass


# Default stream endpoint (uses default model)
@app.websocket("/predict/stream")
async def predict_stream_default(websocket: WebSocket):
    """Default stream endpoint using the default model (pytorch or first available)."""
    await handle_stream_prediction(websocket, "default", model)


# Dynamic stream endpoints for each loaded model
@app.websocket("/predict/stream/{model_key}")
async def predict_stream_model(websocket: WebSocket, model_key: str):
    """
    Model-specific stream endpoint.
    
    Available models:
    - /predict/stream/tfrt-32 : TensorRT Float32
    - /predict/stream/tfrt-16 : TensorRT Float16
    - /predict/stream/tflite-32 : TFLite Float32
    - /predict/stream/tflite-16 : TFLite Float16
    - /predict/stream/pytorch : PyTorch Original
    """
    if model_key not in models:
        await websocket.accept()
        await websocket.send_json({
            "status": "error",
            "error": f"Model '{model_key}' not loaded. Available models: {list(models.keys())}"
        })
        await websocket.close()
        return
    
    selected_model = models[model_key]
    await handle_stream_prediction(websocket, model_key, selected_model)


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