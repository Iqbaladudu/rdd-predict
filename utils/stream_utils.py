"""
Streaming utilities for real-time video processing.
"""
import base64
import numpy as np
import cv2


def encode_frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """
    Encode OpenCV frame to base64 JPEG string.
    
    Args:
        frame: OpenCV BGR frame (numpy array)
        quality: JPEG quality (1-100)
    
    Returns:
        Base64 encoded JPEG string with data URI prefix
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"


def decode_base64_to_frame(base64_str: str) -> np.ndarray:
    """
    Decode base64 string to OpenCV frame.
    
    Args:
        base64_str: Base64 encoded image string (with or without data URI prefix)
    
    Returns:
        OpenCV BGR frame (numpy array)
    
    Raises:
        ValueError: If decoding fails
    """
    # Remove data URI prefix if present
    if base64_str.startswith('data:'):
        base64_str = base64_str.split(',', 1)[1]
    
    try:
        image_bytes = base64.b64decode(base64_str)
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image")
        
        return frame
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")


def encode_frame_to_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    """
    Encode OpenCV frame to JPEG bytes.
    
    Args:
        frame: OpenCV BGR frame (numpy array)
        quality: JPEG quality (1-100)
    
    Returns:
        JPEG bytes
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    return buffer.tobytes()
