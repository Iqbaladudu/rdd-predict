import shutil
import uuid
import os
import logging
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
import secrets

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Device detection - prioritize GPU if available
def get_device():
    """Detect and return the best available device."""
    logger.info("[STEP] Detecting available device...")
    if torch.cuda.is_available():
        # Check for minimum required CUDA version for TensorRT
        cuda_version_str = torch.version.cuda
        if cuda_version_str and float(cuda_version_str.split('.')[0]) < 11:
            logger.warning(f"⚠️  PyTorch CUDA version {cuda_version_str} is too old for TensorRT models.")
            logger.warning("   Falling back to CPU. Please upgrade NVIDIA drivers and CUDA.")
            return 'cpu'
            
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"   CUDA version: {torch.version.cuda}")
        logger.debug(f"[STEP] Device detection complete: {device}")
        return device
    else:
        logger.warning("⚠️  No GPU detected, using CPU")
        logger.warning("   TensorRT models will NOT be available")
        logger.debug("[STEP] Device detection complete: cpu")
        return 'cpu'

DEVICE = get_device()
HAS_GPU = DEVICE == 'cuda'
logger.info(f"[STEP] Using device: {DEVICE}")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Streaming configuration from environment
STREAM_CONFIG = {
    "target_fps": int(os.getenv("STREAM_TARGET_FPS", "30")),
    "max_pending_frames": int(os.getenv("STREAM_MAX_PENDING_FRAMES", "5")),
    "camera_width": int(os.getenv("STREAM_CAMERA_WIDTH", "640")),
    "camera_height": int(os.getenv("STREAM_CAMERA_HEIGHT", "480")),
    "jpeg_quality_client": int(os.getenv("STREAM_JPEG_QUALITY_CLIENT", "50")),
    "jpeg_quality_server": int(os.getenv("STREAM_JPEG_QUALITY_SERVER", "70")),
    "yolo_imgsz": int(os.getenv("STREAM_YOLO_IMGSZ", "480")),
}
logger.info(f"[CONFIG] Streaming settings: {STREAM_CONFIG}")

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

# Check ai_edge_litert availability for TFLite models
logger.info("[STEP] Checking ai_edge_litert availability...")
try:
    from ai_edge_litert.interpreter import Interpreter
    HAS_TENSORFLOW = True
    logger.info("✅ ai_edge_litert available for TFLite models")
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("⚠️  ai_edge_litert not installed - TFLite models will NOT be available")

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
    logger.info(f"[STEP] download_file() called for: {filename}")
    logger.debug(f"[STEP] Checking if file exists: {filename}")
    
    if os.path.exists(filename):
        logger.info(f"[Download] {filename} sudah ada, skip download.")
        return True
    
    try:
        logger.debug(f"[STEP] Starting HTTP request to: {url}")
        with requests.get(url, stream=True, timeout=30) as r:
            logger.debug(f"[STEP] HTTP response status: {r.status_code}")
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded_size = 0

            logger.info(f"[Download] Mengunduh {filename} dari {url}...")
            logger.debug(f"[STEP] Total file size: {total_size / (1024*1024):.2f} MB")
            
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                        # Log every 10% progress
                        if int(progress) % 10 == 0:
                            logger.debug(f"[Download] Progress: {progress:.2f}%")
                        print(f"\r[Download] Progres: {progress:.2f}% ({downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)", end="")
            
            print("\n[Download] Download selesai!")
            logger.info(f"[STEP] Download complete: {filename}")
            return True
    except requests.exceptions.Timeout as e:
        logger.error(f"[Download] Timeout error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False
    except requests.exceptions.HTTPError as e:
        logger.error(f"[Download] HTTP error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False
    except Exception as e:
        logger.exception(f"[Download] Unexpected error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

# Dictionary to store loaded models
models: dict = {}

def load_all_models():
    """Download and load all models at startup."""
    logger.info("[STEP] " + "="*50)
    logger.info("[STEP] LOADING ALL MODELS")
    logger.info("[STEP] " + "="*50)
    
    for model_key, config in MODEL_CONFIGS.items():
        url = config["url"]
        local_path = config["local_path"]
        description = config["description"]
        requires_gpu = config.get("requires_gpu", False)
        requires_tensorflow = config.get("requires_tensorflow", False)
        
        logger.info(f"[STEP] Processing model: {model_key} ({description})")
        logger.debug(f"[STEP] Model config: url={url}, path={local_path}, gpu={requires_gpu}, tf={requires_tensorflow}")
        
        # Skip GPU-only models if no GPU available
        if requires_gpu and not HAS_GPU:
            logger.warning(f"[Model] SKIP: {model_key} requires GPU (TensorRT)")
            continue
        
        # Skip TFLite models if TensorFlow not available
        if requires_tensorflow and not HAS_TENSORFLOW:
            logger.warning(f"[Model] SKIP: {model_key} requires TensorFlow (TFLite)")
            continue
        
        # Download if URL is provided and file doesn't exist
        if url and not os.path.exists(local_path):
            logger.info(f"[STEP] Downloading model: {model_key}")
            success = download_file(url, local_path)
            if not success:
                logger.error(f"[Model] SKIP: Failed to download {model_key}")
                continue
        
        # Check if local file exists
        if not os.path.exists(local_path):
            logger.warning(f"[Model] SKIP: {local_path} not found")
            continue
        
        # Load model
        try:
            logger.info(f"[STEP] Loading YOLO model from: {local_path}")
            model = YOLO(local_path)
            models[model_key] = model
            logger.info(f"[Model] SUCCESS: {model_key} loaded from {local_path}")
        except Exception as e:
            # Handle TensorRT/CUDA initialization failures gracefully
            error_msg = str(e).lower()
            if "cuda" in error_msg or "tensorrt" in error_msg or "segmentation fault" in error_msg:
                logger.error(f"[Model] SKIP: {model_key} - CUDA/TensorRT initialization failed")
                logger.error(f"         This usually means GPU drivers are incompatible or missing.")
                logger.exception(f"         Error details: {e}")
            else:
                logger.exception(f"[Model] ERROR: Failed to load {model_key}: {e}")
    
    logger.info("[STEP] " + "="*50)
    logger.info(f"[STEP] LOADED MODELS: {list(models.keys())}")
    if HAS_GPU:
        logger.info("[STEP] GPU READY: TensorRT models available for acceleration")
    logger.info("[STEP] " + "="*50)

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


@app.get("/stream/config")
async def get_stream_config():
    """Get streaming configuration for the frontend."""
    return STREAM_CONFIG


async def handle_stream_prediction(websocket: WebSocket, model_key: str, selected_model, device_id: str, session_id: str):
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
    logger.info(f"[STEP] handle_stream_prediction() called for model: {model_key}")
    logger.debug(f"[STEP] Accepting WebSocket connection...")
    await websocket.accept()
    frame_index = 0
    
    logger.info(f"[Stream:{model_key}] Client connected")
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            start_time = time.time()
            
            try:
                # Decode base64 frame
                frame = decode_base64_to_frame(data)
                
                # Run YOLO prediction in thread pool (non-blocking)
                import asyncio
                
                def process_frame_sync(frame_data):
                    """Synchronous frame processing for thread pool execution."""
                    results = selected_model(frame_data, device=DEVICE, verbose=False, imgsz=STREAM_CONFIG['yolo_imgsz'])
                    result = results[0]
                    annotated_frame = result.plot()
                    
                    # Extract detections
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
                    
                    return annotated_frame, detections
                
                # Run inference in thread pool
                annotated_frame, detections = await asyncio.to_thread(process_frame_sync, frame)
                
                # Encode processed frame to base64 (quality from config)
                processed_frame_b64 = encode_frame_to_base64(annotated_frame, quality=STREAM_CONFIG['jpeg_quality_server'])
                
                # Calculate processing latency
                processing_latency = (time.time() - start_time) * 1000
                
                # Send response
                response = {
                    "device_id": device_id,
                    "session_id": session_id,
                    "status": "success",
                    "model": model_key,
                    "frame_index": frame_index,
                    "timestamp_ms": int(time.time() * 1000),
                    "processing_latency_ms": round(processing_latency, 2),
                    "processed_frame": processed_frame_b64,
                    "detections": detections,
                    "detection_count": len(detections)
                }
                
                # Log to terminal (less verbose for performance)
                if frame_index % 30 == 0:  # Log every 30 frames
                    logger.info(f"[Stream:{model_key}] Frame {frame_index:04d} | "
                          f"Detections: {len(detections):2d} | "
                          f"Latency: {processing_latency:6.2f}ms")
                
                await websocket.send_json(response)
                frame_index += 1
                
            except ValueError as e:
                logger.error(f"[STEP] ValueError processing frame: {e}")
                # Invalid frame data
                await websocket.send_json({
                    "device_id": device_id,
                    "session_id": session_id,
                    "status": "error",
                    "model": model_key,
                    "frame_index": frame_index,
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        logger.info(f"[Stream:{model_key}] WebSocket disconnected after {frame_index} frames")
    except Exception as e:
        logger.exception(f"[Stream:{model_key}] WebSocket error: {e}")
        try:
            await websocket.send_json({
                "device_id": device_id,
                "session_id": session_id,
                "status": "error",
                "model": model_key,
                "error": str(e)
            })
        except Exception as send_error:
            logger.error(f"[STEP] Failed to send error response: {send_error}")


def decode_bytes_to_frame(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw JPEG bytes to OpenCV frame.
    
    Args:
        image_bytes: Raw JPEG bytes
    
    Returns:
        OpenCV BGR frame (numpy array)
    
    Raises:
        ValueError: If decoding fails
    """
    try:
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image")
        
        return frame
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")


async def handle_binary_stream_prediction(websocket: WebSocket, model_key: str, selected_model, device_id: str, session_id: str):
    """
    Binary protocol handler for high-performance video streaming.
    
    Protocol:
    - Client sends: Raw JPEG bytes (binary)
    - Server returns: Binary message with format:
        - First 4 bytes: JSON header length (uint32, little-endian)
        - Next N bytes: JSON header (status, detections, latency, etc.)
        - Remaining bytes: JPEG frame data
    
    This provides ~33% bandwidth reduction compared to base64.
    """
    logger.info(f"[STEP] handle_binary_stream_prediction() called for model: {model_key}")
    await websocket.accept()
    frame_index = 0
    
    logger.info(f"[BinaryStream:{model_key}] Client connected (binary protocol)")
    
    try:
        while True:
            # Receive binary frame from client
            data = await websocket.receive_bytes()
            start_time = time.time()
            
            try:
                # Decode raw JPEG bytes
                frame = decode_bytes_to_frame(data)
                
                # Run YOLO prediction in thread pool (non-blocking)
                import asyncio
                
                def process_frame_sync(frame_data):
                    """Synchronous frame processing for thread pool execution."""
                    results = selected_model(frame_data, device=DEVICE, verbose=False, imgsz=STREAM_CONFIG['yolo_imgsz'])
                    result = results[0]
                    annotated_frame = result.plot()
                    
                    # Extract detections
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
                    
                    return annotated_frame, detections
                
                # Run inference in thread pool
                annotated_frame, detections = await asyncio.to_thread(process_frame_sync, frame)
                
                # Encode processed frame to JPEG bytes (not base64)
                from utils.stream_utils import encode_frame_to_bytes
                frame_bytes = encode_frame_to_bytes(annotated_frame, quality=STREAM_CONFIG['jpeg_quality_server'])
                
                # Calculate processing latency
                processing_latency = (time.time() - start_time) * 1000
                
                # Create JSON header (without frame data)
                import json
                header = {
                    "device_id": device_id,
                    "session_id": session_id,
                    "status": "success",
                    "model": model_key,
                    "frame_index": frame_index,
                    "timestamp_ms": int(time.time() * 1000),
                    "processing_latency_ms": round(processing_latency, 2),
                    "detections": detections,
                    "detection_count": len(detections)
                }
                header_json = json.dumps(header).encode('utf-8')
                header_length = len(header_json)
                
                # Build binary response: [4 bytes header length] + [header JSON] + [JPEG bytes]
                import struct
                response = struct.pack('<I', header_length) + header_json + frame_bytes
                
                # Log to terminal (less verbose for better performance)
                if frame_index % 30 == 0:  # Log every 30 frames
                    logger.info(f"[BinaryStream:{model_key}] Frame {frame_index:04d} | "
                          f"Detections: {len(detections):2d} | "
                          f"Latency: {processing_latency:6.2f}ms | "
                          f"Size: {len(response)} bytes")
                
                await websocket.send_bytes(response)
                frame_index += 1
                
            except ValueError as e:
                logger.error(f"[BinaryStream] ValueError processing frame: {e}")
                # Send error as JSON text (fallback)
                import json
                await websocket.send_text(json.dumps({
                    "device_id": device_id,
                    "session_id": session_id,
                    "status": "error",
                    "model": model_key,
                    "frame_index": frame_index,
                    "error": str(e)
                }))
                
    except WebSocketDisconnect:
        logger.info(f"[BinaryStream:{model_key}] WebSocket disconnected after {frame_index} frames")
    except Exception as e:
        logger.exception(f"[BinaryStream:{model_key}] WebSocket error: {e}")
        try:
            import json
            await websocket.send_text(json.dumps({
                "device_id": device_id,
                "session_id": session_id,
                "status": "error",
                "model": model_key,
                "error": str(e)
            }))
        except Exception as send_error:
            logger.error(f"[STEP] Failed to send error response: {send_error}")


# Default stream endpoint (uses default model)
@app.websocket("/predict/stream/{device_id}")
async def predict_stream_default(websocket: WebSocket, device_id: str):
    """Default stream endpoint using the default model (pytorch or first available)."""
    logger.info("[STEP] predict_stream_default() endpoint called")
    session_id = secrets.token_hex(16)
    logger.info(f"[STEP] Created session_id: {session_id} for device_id: {device_id}")
    await handle_stream_prediction(websocket, "default", model, device_id, session_id)


# Dynamic stream endpoints for each loaded model
@app.websocket("/predict/stream/{model_key}/{device_id}")
async def predict_stream_model(websocket: WebSocket, model_key: str, device_id: str):
    """
    Model-specific stream endpoint.
    
    Available models:
    - /predict/stream/tfrt-32/{device_id} : TensorRT Float32
    - /predict/stream/tfrt-16/{device_id} : TensorRT Float16
    - /predict/stream/tflite-32/{device_id} : TFLite Float32
    - /predict/stream/tflite-16/{device_id} : TFLite Float16
    - /predict/stream/pytorch/{device_id} : PyTorch Original
    """
    session_id = secrets.token_hex(16)
    logger.info(f"[STEP] Created session_id: {session_id} for device_id: {device_id}")
    logger.info(f"[STEP] predict_stream_model() endpoint called for model: {model_key}")
    
    if model_key not in models:
        logger.error(f"[STEP] Model not found: {model_key}. Available: {list(models.keys())}")
        await websocket.accept()
        await websocket.send_json({
            "device_id": device_id,
            "session_id": session_id,
            "status": "error",
            "error": f"Model '{model_key}' not loaded. Available models: {list(models.keys())}"
        })
        await websocket.close()
        return
    
    selected_model = models[model_key]
    logger.debug(f"[STEP] Selected model: {model_key}")
    await handle_stream_prediction(websocket, model_key, selected_model, device_id, session_id)


# Binary stream endpoints (high-performance, ~33% bandwidth reduction)
@app.websocket("/predict/stream-binary/{device_id}")
async def predict_stream_binary_default(websocket: WebSocket, device_id: str):
    """
    Default binary stream endpoint for high-performance video streaming.
    Uses raw JPEG bytes instead of base64, reducing bandwidth by ~33%.
    """
    logger.info("[STEP] predict_stream_binary_default() endpoint called")
    session_id = secrets.token_hex(16)
    logger.info(f"[STEP] Created session_id: {session_id} for device_id: {device_id}")
    await handle_binary_stream_prediction(websocket, "default", model, device_id, session_id)


@app.websocket("/predict/stream-binary/{model_key}/{device_id}")
async def predict_stream_binary_model(websocket: WebSocket, model_key: str, device_id: str):
    """
    Model-specific binary stream endpoint for high-performance video streaming.
    
    Binary Protocol:
    - Client sends: Raw JPEG bytes
    - Server returns: [4-byte header length] + [JSON header] + [JPEG bytes]
    
    Available models:
    - /predict/stream-binary/tfrt-32/{device_id} : TensorRT Float32
    - /predict/stream-binary/tfrt-16/{device_id} : TensorRT Float16
    - /predict/stream-binary/tflite-32/{device_id} : TFLite Float32
    - /predict/stream-binary/tflite-16/{device_id} : TFLite Float16
    - /predict/stream-binary/pytorch/{device_id} : PyTorch Original
    """
    session_id = secrets.token_hex(16)
    logger.info(f"[STEP] Created session_id: {session_id} for device_id: {device_id}")
    logger.info(f"[STEP] predict_stream_binary_model() endpoint called for model: {model_key}")
    
    if model_key not in models:
        logger.error(f"[STEP] Model not found: {model_key}. Available: {list(models.keys())}")
        await websocket.accept()
        import json
        await websocket.send_text(json.dumps({
            "device_id": device_id,
            "session_id": session_id,
            "status": "error",
            "error": f"Model '{model_key}' not loaded. Available models: {list(models.keys())}"
        }))
        await websocket.close()
        return
    
    selected_model = models[model_key]
    logger.debug(f"[STEP] Selected model: {model_key}")
    await handle_binary_stream_prediction(websocket, model_key, selected_model, device_id, session_id)


@app.post("/predict")
async def predict_media(file: UploadFile = File(...)):
    logger.info("[STEP] predict_media() endpoint called")
    
    # Generate unique ID for this request
    result_type = None
    request_id = str(uuid.uuid4())
    filename = file.filename
    logger.info(f"[STEP] Processing file: {filename}, request_id: {request_id}")
    
    # Handle cases where filename might be empty or missing extension
    if not filename or "." not in filename:
        logger.error(f"[STEP] Invalid filename: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    extension = filename.split(".")[-1].lower()
    logger.debug(f"[STEP] File extension: {extension}")
    
    # Paths
    input_path = Path(f"uploads/{request_id}.{extension}")
    output_filename = f"{request_id}_processed.{extension}"
    output_path = Path(f"static/{output_filename}")
    logger.debug(f"[STEP] Input path: {input_path}, Output path: {output_path}")
    
    # Save uploaded file
    logger.info(f"[STEP] Saving uploaded file to: {input_path}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    logger.debug(f"[STEP] File saved successfully")
        
    results_data = []
    video_metadata = {}
    
    try:
        if extension in ["jpg", "jpeg", "png", "bmp", "webp"]:
            logger.info(f"[STEP] Processing as IMAGE: {extension}")
            result_type = "image"
            # Process Image
            logger.debug(f"[STEP] Running YOLO model on image...")
            results = model(input_path, device=DEVICE)
            result = results[0]
            logger.debug(f"[STEP] YOLO prediction complete, detections: {len(result.boxes)}")
            
            # Save annotated image
            logger.debug(f"[STEP] Plotting and saving annotated image...")
            annotated_frame = result.plot()
            cv2.imwrite(str(output_path), annotated_frame)
            logger.info(f"[STEP] Annotated image saved to: {output_path}")
            
            # Upload to S3
            logger.info(f"[STEP] Uploading to S3...")
            s3_url = upload_file(str(output_path), output_filename)
            logger.debug(f"[STEP] S3 upload complete: {s3_url}")
            
            # Upload to Cloudinary
            logger.info(f"[STEP] Uploading to Cloudinary...")
            cloudinary_result = upload_to_cloudinary(str(output_path))
            logger.debug(f"[STEP] Cloudinary upload complete: {cloudinary_result.get('url')}")
            
            # Extract data
            logger.debug(f"[STEP] Extracting detection data...")
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
            logger.info(f"[STEP] Image processing complete, {len(results_data)} detections")
                
        elif extension in ["mp4", "avi", "mov", "mkv", "webm"]:
            logger.info(f"[STEP] Processing as VIDEO: {extension}")
            result_type = "video"
            # Process Video
            logger.debug(f"[STEP] Opening video capture...")
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                logger.error(f"[STEP] Could not open video file: {input_path}")
                raise HTTPException(status_code=400, detail="Could not open video file")
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"[STEP] Video metadata: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            video_metadata = {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames
            }
            
            # Use mp4v for compatibility
            logger.debug(f"[STEP] Initializing video writer...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_idx = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    logger.debug(f"[STEP] Video read complete at frame {frame_idx}")
                    break
                    
                # Run YOLO
                if frame_idx % 100 == 0:
                    logger.debug(f"[STEP] Processing frame {frame_idx}/{total_frames}")
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
                    logger.debug(f"[STEP] Frame {frame_idx}: {len(frame_detections)} detections found")
                    # Save frame as image and upload to Cloudinary
                    frame_filename = f"{request_id}_frame_{frame_idx}.jpg"
                    frame_path = Path(f"static/{frame_filename}")
                    cv2.imwrite(str(frame_path), annotated_frame)

                    # Upload frame to Cloudinary
                    logger.debug(f"[STEP] Uploading frame {frame_idx} to Cloudinary...")
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
            logger.info(f"[STEP] Video processing complete, {frame_idx} frames processed")

            # Upload processed video to Cloudinary
            logger.info(f"[STEP] Uploading processed video to Cloudinary...")
            cloudinary_result = upload_to_cloudinary(str(output_path))
            logger.debug(f"[STEP] Video Cloudinary upload complete: {cloudinary_result.get('url')}")
            
        else:
            logger.error(f"[STEP] Unsupported file type: {extension}")
            raise HTTPException(status_code=400, detail="Unsupported file type. Use image (jpg, png) or video (mp4, avi).")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[STEP] Error processing file: {e}")
        # Cleanup output if it exists and is partial?
        if os.path.exists(output_path):
             os.remove(output_path)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Cleanup input file
        logger.debug(f"[STEP] Cleaning up input file: {input_path}")
        if os.path.exists(input_path):
            os.remove(input_path)
            logger.debug(f"[STEP] Input file cleaned up")

    logger.info(f"[STEP] Request {request_id} completed successfully")
    logger.info(f"[STEP] Results: {len(results_data)} detections, Cloudinary URL: {cloudinary_result.get('url')}")
    
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