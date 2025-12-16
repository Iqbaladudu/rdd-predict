"""
WebRTC handler for real-time video streaming with YOLO inference.

This module provides WebRTC-based video processing using aiortc library,
following industry best practices for low-latency real-time streaming.

OPTIMIZATIONS:
- ThreadPoolExecutor for non-blocking YOLO inference
- Frame skipping when processing is slow
- Reduced logging overhead
- Direct numpy array manipulation
"""

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Set, Any

import cv2
import numpy as np
from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from aiortc.contrib.media import MediaRelay

logger = logging.getLogger(__name__)

# Global MediaRelay for efficient multi-client streaming
relay = MediaRelay()

# Track active peer connections
peer_connections: Set[RTCPeerConnection] = set()

# Thread pool for YOLO inference (non-blocking)
inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yolo_inference")


def run_yolo_inference(model, img, device, imgsz):
    """Run YOLO inference in thread pool (CPU-bound operation)."""
    results = model(img, device=device, verbose=False, imgsz=imgsz)
    return results[0]


class VideoProcessorTrack(MediaStreamTrack):
    """
    Optimized video track that processes frames through YOLO model.
    
    OPTIMIZATIONS APPLIED:
    - Non-blocking inference using ThreadPoolExecutor
    - Frame skipping when processing is slow
    - Minimal logging to reduce overhead
    - Direct frame manipulation without copies
    """
    
    kind = "video"
    
    def __init__(
        self,
        track: MediaStreamTrack,
        model: Any,
        model_key: str = "default",
        device: str = "cpu",
        imgsz: int = 320,  # Reduced default for faster inference
        skip_frames: int = 0  # Process every Nth frame (0 = process all)
    ):
        super().__init__()
        self.track = track
        self.model = model
        self.model_key = model_key
        self.device = device
        self.imgsz = imgsz
        self.skip_frames = skip_frames
        
        # DataChannel for sending detection results
        self.datachannel: Optional[RTCDataChannel] = None
        
        # Metrics
        self.frame_count = 0
        self.processed_count = 0
        self.total_detections = 0
        self.start_time = time.time()
        
        # Processing state
        self._processing = False
        self._pending_result = None
        self._last_annotated = None
        self._last_detections = []
        
        # Performance tracking
        self._last_log_time = time.time()
        self._latency_sum = 0
        self._latency_count = 0
        
        logger.info(f"[WebRTC:{model_key}] VideoProcessorTrack created | device:{device} | imgsz:{imgsz}")
    
    def set_datachannel(self, channel: RTCDataChannel):
        """Set the data channel for sending detection results."""
        self.datachannel = channel
        logger.info(f"[WebRTC:{self.model_key}] DataChannel attached")
    
    async def recv(self) -> VideoFrame:
        """
        Receive a frame with optimized processing pipeline.
        Uses non-blocking inference and frame skipping.
        """
        frame = await self.track.recv()
        self.frame_count += 1
        
        # Get frame as numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Determine if we should process this frame
        should_process = (
            not self._processing and  # Not already processing
            (self.skip_frames == 0 or self.frame_count % (self.skip_frames + 1) == 0)  # Skip logic
        )
        
        if should_process:
            self._processing = True
            start_time = time.time()
            
            try:
                # Run YOLO inference in thread pool (non-blocking)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    inference_executor,
                    run_yolo_inference,
                    self.model, img, self.device, self.imgsz
                )
                
                # Get annotated frame
                annotated = result.plot()
                
                # Extract detections (minimal processing)
                detections = []
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        c = int(box.cls)
                        detections.append({
                            "class": self.model.names[c],
                            "confidence": round(float(box.conf), 3),
                            "bbox": [round(x, 1) for x in box.xyxy.tolist()[0]]
                        })
                
                # Update cached results
                self._last_annotated = annotated
                self._last_detections = detections
                self.processed_count += 1
                self.total_detections += len(detections)
                
                # Track latency
                latency = (time.time() - start_time) * 1000
                self._latency_sum += latency
                self._latency_count += 1
                
                # Send detection results via DataChannel
                if self.datachannel and self.datachannel.readyState == "open":
                    try:
                        self.datachannel.send(json.dumps({
                            "status": "success",
                            "model": self.model_key,
                            "frame_index": self.processed_count,
                            "timestamp_ms": int(time.time() * 1000),
                            "processing_latency_ms": round(latency, 1),
                            "detections": detections,
                            "detection_count": len(detections)
                        }))
                    except Exception:
                        pass  # Ignore send errors
                
            except Exception as e:
                logger.error(f"[WebRTC:{self.model_key}] Inference error: {e}")
            finally:
                self._processing = False
        
        # Log stats periodically (every 5 seconds)
        now = time.time()
        if now - self._last_log_time >= 5.0:
            elapsed = now - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            avg_latency = self._latency_sum / self._latency_count if self._latency_count > 0 else 0
            logger.info(
                f"[WebRTC:{self.model_key}] FPS:{fps:.1f} | "
                f"Frames:{self.frame_count} | "
                f"Processed:{self.processed_count} | "
                f"AvgLatency:{avg_latency:.1f}ms | "
                f"Detections:{self.total_detections}"
            )
            self._last_log_time = now
        
        # Return annotated frame if available, otherwise original
        if self._last_annotated is not None:
            new_frame = VideoFrame.from_ndarray(self._last_annotated, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        
        return frame
    
    def stop(self):
        """Stop the track and cleanup resources."""
        super().stop()
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        logger.info(
            f"[WebRTC:{self.model_key}] Track stopped | "
            f"Frames:{self.frame_count} | "
            f"Processed:{self.processed_count} | "
            f"AvgFPS:{avg_fps:.1f} | "
            f"Detections:{self.total_detections}"
        )


class WebRTCSessionManager:
    """
    Manages WebRTC peer connections and their lifecycle.
    
    Responsibilities:
    - Create and configure peer connections
    - Track active sessions
    - Cleanup disconnected sessions
    - Provide session statistics
    """
    
    def __init__(self, models: Dict[str, Any], default_model: Any, device: str = "cpu", stream_config: dict = None):
        self.models = models
        self.default_model = default_model
        self.device = device
        self.stream_config = stream_config or {}
        self.sessions: Dict[str, RTCPeerConnection] = {}
        self.session_tracks: Dict[str, VideoProcessorTrack] = {}
        
        logger.info(f"[WebRTC] SessionManager initialized with {len(models)} models")
    
    async def create_session(
        self,
        offer: RTCSessionDescription,
        model_key: Optional[str] = None
    ) -> tuple[RTCSessionDescription, str]:
        """
        Create a new WebRTC session from an SDP offer.
        
        Args:
            offer: SDP offer from client
            model_key: Optional model key for inference (uses default if None)
        
        Returns:
            Tuple of (SDP answer, session_id)
        """
        # Generate session ID
        session_id = str(uuid.uuid4())[:8]
        
        # Select model
        if model_key and model_key in self.models:
            model = self.models[model_key]
        else:
            model = self.default_model
            model_key = "default"
        
        logger.info(f"[WebRTC] Creating session {session_id} with model: {model_key}")
        
        # Create peer connection (no STUN/TURN needed for VPS with public IP)
        pc = RTCPeerConnection()
        self.sessions[session_id] = pc
        peer_connections.add(pc)
        
        # Track to store VideoProcessorTrack for datachannel attachment
        processor_track: Optional[VideoProcessorTrack] = None
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"[WebRTC:{session_id}] Connection state: {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self.close_session(session_id)
        
        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            nonlocal processor_track
            logger.info(f"[WebRTC:{session_id}] Received track: {track.kind}")
            
            if track.kind == "video":
                # Create processor track with YOLO inference (settings from ENV)
                processor_track = VideoProcessorTrack(
                    track=relay.subscribe(track),
                    model=model,
                    model_key=model_key,
                    device=self.device,
                    imgsz=self.stream_config.get("yolo_imgsz", 480),
                    skip_frames=self.stream_config.get("skip_frames", 0)
                )
                self.session_tracks[session_id] = processor_track
                
                # Add processed track to send back to client
                pc.addTrack(processor_track)
                
                @track.on("ended")
                async def on_ended():
                    logger.info(f"[WebRTC:{session_id}] Track ended")
                    await self.close_session(session_id)
        
        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel):
            logger.info(f"[WebRTC:{session_id}] DataChannel opened: {channel.label}")
            
            # Attach datachannel to processor track
            if processor_track:
                processor_track.set_datachannel(channel)
            elif session_id in self.session_tracks:
                self.session_tracks[session_id].set_datachannel(channel)
            
            @channel.on("message")
            def on_message(message):
                # Handle incoming messages from client if needed
                logger.debug(f"[WebRTC:{session_id}] Received message: {message}")
        
        # Set remote description (client's offer)
        await pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        logger.info(f"[WebRTC:{session_id}] Session created successfully")
        
        return pc.localDescription, session_id
    
    async def close_session(self, session_id: str):
        """Close and cleanup a WebRTC session."""
        if session_id in self.sessions:
            pc = self.sessions.pop(session_id)
            peer_connections.discard(pc)
            
            if session_id in self.session_tracks:
                track = self.session_tracks.pop(session_id)
                track.stop()
            
            await pc.close()
            logger.info(f"[WebRTC:{session_id}] Session closed")
    
    async def close_all_sessions(self):
        """Close all active sessions (for shutdown)."""
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            await self.close_session(session_id)
        logger.info("[WebRTC] All sessions closed")
    
    def get_stats(self) -> dict:
        """Get statistics about active sessions."""
        return {
            "active_sessions": len(self.sessions),
            "session_ids": list(self.sessions.keys())
        }


async def cleanup_peer_connections():
    """Cleanup all peer connections on shutdown."""
    coros = [pc.close() for pc in peer_connections]
    await asyncio.gather(*coros)
    peer_connections.clear()
    logger.info("[WebRTC] All peer connections cleaned up")
