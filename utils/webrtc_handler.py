"""
WebRTC handler for real-time video streaming with YOLO inference.

This module provides WebRTC-based video processing using aiortc library,
following industry best practices for low-latency real-time streaming.
"""

import asyncio
import json
import logging
import time
import uuid
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


class VideoProcessorTrack(MediaStreamTrack):
    """
    A video track that processes frames through YOLO model.
    
    This track receives video frames from the client, runs YOLO inference,
    and returns annotated frames. Detection results are sent via DataChannel.
    
    Best Practices Applied:
    - Uses native video frames (no base64 encoding overhead)
    - Separates video stream from detection data (DataChannel)
    - Supports H.264/VP8 for low-latency encoding
    """
    
    kind = "video"
    
    def __init__(
        self,
        track: MediaStreamTrack,
        model: Any,
        model_key: str = "default",
        device: str = "cpu",
        imgsz: int = 480,
        jpeg_quality: int = 70
    ):
        super().__init__()
        self.track = track
        self.model = model
        self.model_key = model_key
        self.device = device
        self.imgsz = imgsz
        self.jpeg_quality = jpeg_quality
        
        # DataChannel for sending detection results
        self.datachannel: Optional[RTCDataChannel] = None
        
        # Metrics
        self.frame_count = 0
        self.total_detections = 0
        self.start_time = time.time()
        
        logger.info(f"[WebRTC:{model_key}] VideoProcessorTrack created on device: {device}")
    
    def set_datachannel(self, channel: RTCDataChannel):
        """Set the data channel for sending detection results."""
        self.datachannel = channel
        logger.info(f"[WebRTC:{self.model_key}] DataChannel attached")
    
    async def recv(self) -> VideoFrame:
        """
        Receive a frame, process it with YOLO, and return annotated frame.
        Detection results are sent via DataChannel.
        """
        frame = await self.track.recv()
        start_time = time.time()
        
        try:
            # Convert frame to numpy array (BGR format for OpenCV)
            img = frame.to_ndarray(format="bgr24")
            
            # Run YOLO inference
            results = self.model(img, device=self.device, verbose=False, imgsz=self.imgsz)
            result = results[0]
            
            # Get annotated frame
            annotated = result.plot()
            
            # Extract detections
            detections = []
            for box in result.boxes:
                c = int(box.cls)
                class_name = self.model.names[c]
                conf = float(box.conf)
                xyxy = box.xyxy.tolist()[0]
                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 2) for x in xyxy]
                })
            
            # Calculate processing latency
            processing_latency = (time.time() - start_time) * 1000
            self.frame_count += 1
            self.total_detections += len(detections)
            
            # Send detection results via DataChannel
            if self.datachannel and self.datachannel.readyState == "open":
                detection_data = {
                    "status": "success",
                    "model": self.model_key,
                    "frame_index": self.frame_count,
                    "timestamp_ms": int(time.time() * 1000),
                    "processing_latency_ms": round(processing_latency, 2),
                    "detections": detections,
                    "detection_count": len(detections)
                }
                try:
                    self.datachannel.send(json.dumps(detection_data))
                except Exception as e:
                    logger.warning(f"[WebRTC:{self.model_key}] Failed to send detection data: {e}")
            
            # Log frame processing
            if self.frame_count % 30 == 0 or len(detections) > 0:
                logger.info(
                    f"[WebRTC:{self.model_key}] Frame {self.frame_count:04d} | "
                    f"Detections: {len(detections):2d} | "
                    f"Latency: {processing_latency:6.2f}ms"
                )
            
            # Convert back to VideoFrame
            new_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            
            return new_frame
            
        except Exception as e:
            logger.error(f"[WebRTC:{self.model_key}] Error processing frame: {e}")
            # Return original frame if processing fails
            return frame
    
    def stop(self):
        """Stop the track and cleanup resources."""
        super().stop()
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        logger.info(
            f"[WebRTC:{self.model_key}] Track stopped | "
            f"Frames: {self.frame_count} | "
            f"Avg FPS: {avg_fps:.2f} | "
            f"Total detections: {self.total_detections}"
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
                # Create processor track with YOLO inference
                processor_track = VideoProcessorTrack(
                    track=relay.subscribe(track),
                    model=model,
                    model_key=model_key,
                    device=self.device,
                    imgsz=self.stream_config.get("yolo_imgsz", 480),
                    jpeg_quality=self.stream_config.get("jpeg_quality_server", 70)
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
