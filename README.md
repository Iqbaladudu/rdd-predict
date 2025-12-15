# 🛣️ RDD-Predict - Road Damage Detection API

**RDD-Predict** adalah API berbasis **FastAPI** untuk mendeteksi kerusakan jalan secara real-time menggunakan model **YOLOv8**. API ini mendukung berbagai format model (PyTorch, TensorRT, TFLite) dan menyediakan endpoint untuk pemrosesan gambar/video serta streaming real-time via WebSocket.

---

## 📋 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Model dan Klasifikasi](#-model-dan-klasifikasi)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [WebSocket Streaming](#-websocket-streaming)
- [Web Dashboard](#-web-dashboard)
- [Command Line Client](#-command-line-client)
- [Docker Deployment](#-docker-deployment)
- [Konfigurasi Environment](#-konfigurasi-environment)
- [Struktur Direktori](#-struktur-direktori)

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 🎯 **Multi-Model Support** | PyTorch (.pt), TensorRT (.engine), TFLite (.tflite) |
| 🎬 **Real-time Streaming** | WebSocket endpoint untuk streaming video real-time |
| 📸 **Image/Video Processing** | Upload dan proses gambar atau video |
| 🖥️ **Web Dashboard** | Interface browser untuk live detection |
| ☁️ **Cloud Storage** | Upload otomatis ke Cloudflare R2 dan Cloudinary |
| 🚀 **GPU Acceleration** | Dukungan CUDA/cuDNN untuk akselerasi GPU |
| 🐳 **Docker Ready** | Containerized deployment dengan NVIDIA CUDA |

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Client Layer                                      │
├─────────────────┬─────────────────────────┬─────────────────────────────────┤
│   Web Browser   │   Python CLI Client     │      External Applications       │
│  (stream.html)  │ (test_stream_client.py) │        (REST/WebSocket)          │
└────────┬────────┴────────────┬────────────┴───────────────┬─────────────────┘
         │                     │                            │
         │  WebSocket (ws://)  │     WebSocket (ws://)      │   HTTP (REST)
         │                     │                            │
         ▼                     ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Server (main.py)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────────┐  │
│  │  REST Endpoints  │  │ WebSocket Stream │  │    Static File Server    │  │
│  │                  │  │                  │  │                           │  │
│  │  GET  /          │  │ /predict/stream  │  │  GET /static/*            │  │
│  │  GET  /ping      │  │ /predict/stream/ │  │                           │  │
│  │  GET  /models    │  │   {model_key}    │  │                           │  │
│  │  POST /predict   │  │                  │  │                           │  │
│  └──────────────────┘  └──────────────────┘  └───────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
        │   PyTorch     │  │   TensorRT    │  │    TFLite     │
        │  (.pt model)  │  │ (.engine FP16 │  │  (.tflite)    │
        │               │  │   / FP32)     │  │               │
        └───────────────┘  └───────────────┘  └───────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────┐
        │                   YOLOv8 Inference                  │
        │              (Ultralytics + OpenCV)                 │
        └─────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
        │ Cloudflare R2 │  │   Cloudinary  │  │  Local Static │
        │   (S3 API)    │  │               │  │     Files     │
        └───────────────┘  └───────────────┘  └───────────────┘
```

---

## 🎯 Model dan Klasifikasi

### Model yang Tersedia

| Model Key | Tipe | Deskripsi | Requires GPU | Requires TensorFlow |
|-----------|------|-----------|--------------|---------------------|
| `pytorch` | PyTorch | Model original (.pt) | ❌ (dapat CPU/GPU) | ❌ |
| `tfrt-32` | TensorRT | Float32 precision | ✅ | ❌ |
| `tfrt-16` | TensorRT | Float16 precision (lebih cepat) | ✅ | ❌ |
| `tflite-32` | TFLite | Float32 untuk edge devices | ❌ | ✅ |
| `tflite-16` | TFLite | Float16 untuk edge devices | ❌ | ✅ |

### Klasifikasi Kerusakan Jalan

| Kode | Nama | Deskripsi |
|------|------|-----------|
| **D00** | Longitudinal Crack | Retakan memanjang sepanjang jalur roda |
| **D10** | Transverse Crack | Retakan melintang tegak lurus jalan |
| **D20** | Alligator Crack | Retakan fatigue berbentuk kulit buaya |
| **D40** | Pothole | Lubang pada permukaan jalan |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- CUDA 12.x (opsional, untuk GPU)
- [uv](https://docs.astral.sh/uv/) package manager

### Instalasi

```bash
# Clone repository
git clone <repository-url>
cd rdd-predict

# Install dependencies dengan uv
uv sync

# Atau dengan pip (alternatif)
pip install -e .
```

### Menjalankan Server

```bash
# Dengan FastAPI (development)
fastapi dev

# Atau dengan uvicorn langsung
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server akan berjalan di `http://localhost:8000`

---

## 📡 API Reference

### 1. Root Endpoint

**`GET /`**

Cek apakah API berjalan.

**Response:**
```json
{
  "message": "RDD Predict API is running"
}
```

---

### 2. Health Check

**`GET /ping`**

Cek status kesehatan API.

**Response:**
```json
{
  "status": "healthy"
}
```

---

### 3. List Models

**`GET /models`**

Dapatkan daftar model yang tersedia.

**Response:**
```json
{
  "device": "cuda",
  "has_gpu": true,
  "loaded_models": [
    {
      "key": "pytorch",
      "description": "PyTorch Original",
      "stream_endpoint": "/predict/stream/pytorch",
      "requires_gpu": false,
      "loaded": true
    },
    {
      "key": "tfrt-32",
      "description": "TensorRT Float32",
      "stream_endpoint": "/predict/stream/tfrt-32",
      "requires_gpu": true,
      "loaded": true
    }
  ],
  "total_loaded": 2,
  "default_model": "pytorch"
}
```

---

### 4. Predict Media (Image/Video)

**`POST /predict`**

Upload dan proses gambar atau video.

**Request:**
- `Content-Type: multipart/form-data`
- Body: `file` - File gambar (jpg, png, bmp, webp) atau video (mp4, avi, mov, mkv, webm)

**cURL Example - Image:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@road_image.jpg"
```

**cURL Example - Video:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@road_video.mp4"
```

**Response (Image):**
```json
{
  "status": "success",
  "file_url": "/static/uuid_processed.jpg",
  "image": "image",
  "cloudinary_url": "https://res.cloudinary.com/.../rdd-predict/...",
  "cloudinary_public_id": "rdd-predict/xxx",
  "filename": "uuid_processed.jpg",
  "metadata": {
    "type": "image"
  },
  "data_summary": "Found 3 frames/items with detections",
  "data": [
    {
      "class": "D00",
      "confidence": 0.8542,
      "bbox": [120.5, 230.2, 450.8, 380.1]
    },
    {
      "class": "D40",
      "confidence": 0.7891,
      "bbox": [550.0, 290.5, 680.3, 420.7]
    }
  ]
}
```

**Response (Video):**
```json
{
  "status": "success",
  "file_url": "/static/uuid_processed.mp4",
  "video": "video",
  "cloudinary_url": "https://res.cloudinary.com/.../rdd-predict/...",
  "cloudinary_public_id": "rdd-predict/xxx",
  "filename": "uuid_processed.mp4",
  "metadata": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "total_frames": 450
  },
  "data_summary": "Found 127 frames/items with detections",
  "data": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "detections": [
        {
          "class": "D20",
          "confidence": 0.9123,
          "bbox": [100.0, 200.0, 300.0, 400.0]
        }
      ],
      "frame_url": "https://res.cloudinary.com/.../frame_0.jpg",
      "frame_public_id": "rdd-predict/frame_0"
    }
  ]
}
```

---

## 🔌 WebSocket Streaming

### Flow Diagram

```
Client                                         Server
  │                                              │
  │  ──────── WebSocket Connect ────────────▶   │
  │           ws://localhost:8000/predict/stream │
  │                                              │
  │  ◀──────── Connection Accepted ─────────    │
  │                                              │
  ├──────────────── Loop ────────────────────────┤
  │                                              │
  │  ──────── Base64 JPEG Frame ─────────────▶  │
  │           (raw base64 string)                │
  │                                              │
  │           ┌────────────────────────────┐     │
  │           │  1. Decode Base64          │     │
  │           │  2. YOLO Inference         │     │
  │           │  3. Annotate Frame         │     │
  │           │  4. Extract Detections     │     │
  │           │  5. Encode to Base64       │     │
  │           └────────────────────────────┘     │
  │                                              │
  │  ◀──────── JSON Response ─────────────────  │
  │           (processed frame + detections)     │
  │                                              │
  └──────────────────────────────────────────────┘
  │                                              │
  │  ──────── WebSocket Disconnect ──────────▶  │
  │                                              │
```

### Endpoint WebSocket

| Endpoint | Deskripsi |
|----------|-----------|
| `ws://localhost:8000/predict/stream` | Default model (pytorch) |
| `ws://localhost:8000/predict/stream/pytorch` | PyTorch model |
| `ws://localhost:8000/predict/stream/tfrt-32` | TensorRT FP32 |
| `ws://localhost:8000/predict/stream/tfrt-16` | TensorRT FP16 |
| `ws://localhost:8000/predict/stream/tflite-32` | TFLite FP32 |
| `ws://localhost:8000/predict/stream/tflite-16` | TFLite FP16 |

### Request Format

Kirim frame sebagai **Base64 encoded JPEG string** (tanpa prefix):

```
/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwME...
```

Atau dengan data URI prefix:

```
data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMD...
```

### Response Format

```json
{
  "status": "success",
  "model": "pytorch",
  "frame_index": 42,
  "timestamp_ms": 1702658400000,
  "processing_latency_ms": 23.45,
  "processed_frame": "data:image/jpeg;base64,/9j/4AAQSkZ...",
  "detections": [
    {
      "class": "D00",
      "confidence": 0.8542,
      "bbox": [120.5, 230.2, 450.8, 380.1]
    },
    {
      "class": "D40",
      "confidence": 0.7891,
      "bbox": [550.0, 290.5, 680.3, 420.7]
    }
  ],
  "detection_count": 2
}
```

### Error Response

```json
{
  "status": "error",
  "model": "pytorch",
  "frame_index": 42,
  "error": "Invalid base64 image data: ..."
}
```

### JavaScript Client Example

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/predict/stream/pytorch');

ws.onopen = () => {
  console.log('Connected!');
  startStreaming();
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  
  if (response.status === 'success') {
    // Display processed frame
    document.getElementById('resultImage').src = response.processed_frame;
    
    // Log detections
    console.log(`Frame ${response.frame_index}: ${response.detection_count} detections`);
    response.detections.forEach(det => {
      console.log(`  - ${det.class}: ${(det.confidence * 100).toFixed(1)}%`);
    });
  }
};

// Send frame from video/canvas
function sendFrame(canvas) {
  canvas.toBlob(blob => {
    const reader = new FileReader();
    reader.onloadend = () => {
      // Send base64 without prefix
      const base64 = reader.result.split(',')[1];
      ws.send(base64);
    };
    reader.readAsDataURL(blob);
  }, 'image/jpeg', 0.85);
}
```

### Python Client Example

```python
import asyncio
import base64
import cv2
import websockets
import json

async def stream_video():
    uri = "ws://localhost:8000/predict/stream/pytorch"
    
    async with websockets.connect(uri) as ws:
        cap = cv2.VideoCapture(0)  # Webcam
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame
            await ws.send(frame_b64)
            
            # Receive response
            response = json.loads(await ws.recv())
            
            if response['status'] == 'success':
                print(f"Frame {response['frame_index']}: {response['detection_count']} detections")
                for det in response['detections']:
                    print(f"  - {det['class']}: {det['confidence']:.2%}")

asyncio.run(stream_video())
```

---

## 🌐 Web Dashboard

Akses web dashboard untuk real-time detection di browser:

```
http://localhost:8000/static/stream.html
```

### Fitur Dashboard

- 📹 **Live Camera Feed** - Tampilkan feed dari webcam/kamera
- 🎯 **Real-time Detection** - Hasil deteksi dengan anotasi
- 📊 **Statistics** - Frames analyzed, latency, FPS, total detections
- 📈 **Detection Timeline** - Chart real-time jumlah deteksi per frame
- 🔍 **Damage Classification** - Counter per kategori (D00, D10, D20, D40)
- 📝 **Detection Log** - Log detail setiap deteksi
- 🤖 **Model Selector** - Pilih model yang ingin digunakan
- 💾 **Export Data** - Export hasil ke JSON atau CSV

### Screenshot Layout

```
┌─────────────────────────────────────────────────────────────┐
│ 🛣️ RDD Analysis   [Model: ▼]  [●Offline] [▶Start] [⏹Stop]   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ 📊 Frames: 0     │  │ ⚡ Latency: 0ms  │                │
│  └──────────────────┘  └──────────────────┘                │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ 🎯 Detections: 0 │  │ 📹 FPS: 0        │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                             │
│  ┌─────────────────────┐ ┌─────────────────────┐           │
│  │ 📹 Camera Feed     │ │ 🎯 Detection Result │           │
│  │      [LIVE]        │ │    [AI PROCESSED]   │           │
│  │                    │ │                     │           │
│  │    (video feed)    │ │  (annotated feed)   │           │
│  └─────────────────────┘ └─────────────────────┘           │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │ 📈 Detection Timeline                    [Reset] │      │
│  │              (chart here)                        │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Command Line Client

### Instalasi Dependensi Tambahan

```bash
pip install websockets aiohttp opencv-python
```

### Usage

```bash
# Lihat bantuan
python test_stream_client.py --help

# Stream dari video file
python test_stream_client.py video.mp4

# Stream dari video file dengan model tertentu
python test_stream_client.py video.mp4 --model tfrt-32

# Stream dari webcam (default camera)
python test_stream_client.py --webcam

# Stream dari webcam dengan device tertentu
python test_stream_client.py --webcam 2

# Stream dari webcam dan simpan hasil
python test_stream_client.py --webcam --save output.mp4

# Stream dengan model TensorRT
python test_stream_client.py --webcam --model tfrt-16

# List video devices
python test_stream_client.py --list

# List available models dari server
python test_stream_client.py --list-models
```

### Output Example

```
Video: road_video.mp4
FPS: 30.0, Total frames: 450
Connecting to ws://localhost:8000/predict/stream/pytorch...
Connected! Streaming frames...

[Frame 0/450] Latency: 45.2ms | Detections: 2
  - D00: 85.42% @ [120.5, 230.2, 450.8, 380.1]
  - D40: 78.91% @ [550.0, 290.5, 680.3, 420.7]

[Frame 1/450] Latency: 42.1ms | Detections: 1
  - D20: 92.33% @ [200.0, 150.0, 400.0, 350.0]

...

Completed! Processed 450 frames in 15.32s
Average FPS: 29.37
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t rdd-predict .
```

### Run Container (GPU)

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/models:/code/models \
  --name rdd-predict \
  rdd-predict
```

### Run Container (CPU Only)

```bash
docker run -d \
  -p 8000:8000 \
  -v /path/to/models:/code/models \
  --name rdd-predict \
  rdd-predict
```

### Docker Compose

```yaml
version: '3.8'
services:
  rdd-predict:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/code/models
      - ./uploads:/code/uploads
      - ./static:/code/static
    environment:
      - CLOUDINARY_URL=cloudinary://API_KEY:API_SECRET@CLOUD_NAME
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## ⚙️ Konfigurasi Environment

Buat file `.env` di root directory:

```env
# Cloudinary Configuration
CLOUDINARY_URL=cloudinary://api_key:api_secret@cloud_name

# Atau individual values:
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

### Environment Variables

| Variable | Deskripsi | Required |
|----------|-----------|----------|
| `CLOUDINARY_URL` | Cloudinary connection URL | ✅ (untuk upload) |
| `AWS_ACCESS_KEY_ID` | R2/S3 access key | ✅ (untuk S3) |
| `AWS_SECRET_ACCESS_KEY` | R2/S3 secret key | ✅ (untuk S3) |

---

## 📁 Struktur Direktori

```
rdd-predict/
├── main.py                    # FastAPI application entry point
├── test_stream_client.py      # CLI client untuk testing
├── pyproject.toml             # Python dependencies (uv/pip)
├── uv.lock                    # Lock file untuk uv
├── Dockerfile                 # Docker configuration
├── .env                       # Environment variables
├── YOLOv8_Small_RDD.pt       # PyTorch model weights
│
├── models/                    # Downloaded model files
│   ├── YOLOv8_Small_RDD_float32.engine    # TensorRT FP32
│   ├── YOLOv8_Small_RDD_float16.engine    # TensorRT FP16
│   ├── YOLOv8_Small_RDD_float32.tflite    # TFLite FP32
│   └── YOLOv8_Small_RDD_float16.tflite    # TFLite FP16
│
├── static/                    # Static files served by FastAPI
│   ├── stream.html            # Web dashboard
│   └── (processed files)      # Generated result files
│
├── uploads/                   # Temporary upload directory
│
└── utils/                     # Utility modules
    ├── __init__.py
    ├── stream_utils.py        # Base64 encode/decode utilities
    ├── boto.py                # Cloudflare R2/S3 upload
    └── cloudinary_uploader.py # Cloudinary upload
```

---

## 📊 Performance Benchmarks

| Model | Device | Latency (avg) | FPS |
|-------|--------|---------------|-----|
| PyTorch | CPU | ~150ms | ~6-7 |
| PyTorch | GPU (RTX 3080) | ~25ms | ~40 |
| TensorRT FP32 | GPU (RTX 3080) | ~15ms | ~65 |
| TensorRT FP16 | GPU (RTX 3080) | ~10ms | ~100 |
| TFLite FP32 | CPU | ~120ms | ~8 |

*Benchmark pada gambar 640x640px*

---

## 🔧 Troubleshooting

### Model tidak ter-load

```
[Model] SKIP: tfrt-32 requires GPU (TensorRT)
```

**Solusi:** Pastikan NVIDIA GPU dan CUDA driver terinstall dengan benar.

### WebSocket connection refused

```
Connection error: [Errno 111] Connection refused
```

**Solusi:** Pastikan server berjalan di `localhost:8000`.

### TFLite models tidak tersedia

```
[Model] SKIP: tflite-32 requires TensorFlow (TFLite)
```

**Solusi:** Install TensorFlow: `pip install tensorflow`

### Out of GPU memory

**Solusi:** Gunakan model FP16 atau kurangi ukuran frame input.

---

## 📄 License

MIT License

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📞 Contact

For questions or support, please open an issue on GitHub.
