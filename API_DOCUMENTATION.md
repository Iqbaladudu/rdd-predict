# 📡 RDD-Predict API Documentation

> Dokumentasi lengkap endpoint API untuk integrasi frontend.

---

## 📋 Daftar Isi

- [Base URL](#base-url)
- [REST Endpoints](#rest-endpoints)
  - [GET /](#get-)
  - [GET /ping](#get-ping)
  - [GET /models](#get-models)
  - [GET /stream/config](#get-streamconfig)
  - [POST /predict](#post-predict)
- [WebSocket Endpoints](#websocket-endpoints)
  - [Base64 Protocol](#base64-protocol-standar)
  - [Binary Protocol](#binary-protocol-high-performance)
- [Data Types](#data-types)
- [Error Handling](#error-handling)

---

## Base URL

```
Development: http://localhost:8000
Production:  https://your-domain.com
```

---

## REST Endpoints

### GET /

Health check dasar untuk memastikan API berjalan.

**Request:**
```http
GET / HTTP/1.1
Host: localhost:8000
```

**Response:**
```json
{
  "message": "RDD Predict API is running"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `message` | `string` | Status message |

---

### GET /ping

Health check singkat untuk monitoring.

**Request:**
```http
GET /ping HTTP/1.1
Host: localhost:8000
```

**Response:**
```json
{
  "status": "healthy"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | Status kesehatan: `"healthy"` |

---

### GET /models

Mendapatkan daftar model yang tersedia beserta endpoint streaming-nya.

**Request:**
```http
GET /models HTTP/1.1
Host: localhost:8000
```

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
    },
    {
      "key": "tfrt-16",
      "description": "TensorRT Float16",
      "stream_endpoint": "/predict/stream/tfrt-16",
      "requires_gpu": true,
      "loaded": true
    },
    {
      "key": "tflite-32",
      "description": "TFLite Float32",
      "stream_endpoint": "/predict/stream/tflite-32",
      "requires_gpu": false,
      "loaded": true
    },
    {
      "key": "tflite-16",
      "description": "TFLite Float16",
      "stream_endpoint": "/predict/stream/tflite-16",
      "requires_gpu": false,
      "loaded": true
    }
  ],
  "total_loaded": 5,
  "default_model": "pytorch"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `device` | `string` | Device yang digunakan: `"cuda"` atau `"cpu"` |
| `has_gpu` | `boolean` | Apakah GPU tersedia |
| `loaded_models` | `array` | Daftar model yang telah dimuat |
| `loaded_models[].key` | `string` | Key model untuk digunakan di endpoint |
| `loaded_models[].description` | `string` | Deskripsi model |
| `loaded_models[].stream_endpoint` | `string` | Endpoint WebSocket untuk model ini |
| `loaded_models[].requires_gpu` | `boolean` | Apakah model memerlukan GPU |
| `loaded_models[].loaded` | `boolean` | Status model terload |
| `total_loaded` | `number` | Jumlah model yang berhasil dimuat |
| `default_model` | `string` | Model default yang digunakan |

**JavaScript Example:**
```javascript
const response = await fetch('/models');
const data = await response.json();

// Populate model selector dropdown
const select = document.getElementById('modelSelect');
data.loaded_models.forEach(model => {
  const option = document.createElement('option');
  option.value = model.key;
  option.textContent = `${model.description} ${model.requires_gpu ? '(GPU)' : ''}`;
  select.appendChild(option);
});
```

---

### GET /stream/config

Mendapatkan konfigurasi streaming untuk optimasi frontend.

**Request:**
```http
GET /stream/config HTTP/1.1
Host: localhost:8000
```

**Response:**
```json
{
  "target_fps": 30,
  "max_pending_frames": 5,
  "camera_width": 640,
  "camera_height": 480,
  "jpeg_quality_client": 50,
  "jpeg_quality_server": 70,
  "yolo_imgsz": 480
}
```

| Field | Type | Description |
|-------|------|-------------|
| `target_fps` | `number` | Target FPS yang disarankan |
| `max_pending_frames` | `number` | Maksimal frame pending sebelum throttling |
| `camera_width` | `number` | Lebar resolusi kamera yang disarankan |
| `camera_height` | `number` | Tinggi resolusi kamera yang disarankan |
| `jpeg_quality_client` | `number` | Kualitas JPEG untuk encode di client (1-100) |
| `jpeg_quality_server` | `number` | Kualitas JPEG dari server |
| `yolo_imgsz` | `number` | Ukuran gambar input YOLO |

**JavaScript Example:**
```javascript
const response = await fetch('/stream/config');
const config = await response.json();

// Apply config to camera settings
const constraints = {
  video: {
    width: { ideal: config.camera_width },
    height: { ideal: config.camera_height }
  }
};

// Apply to canvas encoding
const quality = config.jpeg_quality_client / 100;
canvas.toBlob(handleBlob, 'image/jpeg', quality);
```

---

### POST /predict

Upload dan proses gambar atau video untuk deteksi kerusakan jalan.

**Request:**
```http
POST /predict HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data

file=@road_image.jpg
```

**Supported File Types:**
- **Image:** `jpg`, `jpeg`, `png`, `bmp`, `webp`
- **Video:** `mp4`, `avi`, `mov`, `mkv`, `webm`

#### Response untuk Image

```json
{
  "status": "success",
  "file_url": "/static/a1b2c3d4-5678-90ab-cdef-1234567890ab_processed.jpg",
  "image": "image",
  "cloudinary_url": "https://res.cloudinary.com/your-cloud/image/upload/v1234567890/rdd-predict/abc123.jpg",
  "cloudinary_public_id": "rdd-predict/abc123",
  "filename": "a1b2c3d4-5678-90ab-cdef-1234567890ab_processed.jpg",
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
    },
    {
      "class": "D20",
      "confidence": 0.6234,
      "bbox": [200.0, 100.0, 350.0, 250.0]
    }
  ]
}
```

#### Response untuk Video

```json
{
  "status": "success",
  "file_url": "/static/a1b2c3d4-5678-90ab-cdef-1234567890ab_processed.mp4",
  "video": "video",
  "cloudinary_url": "https://res.cloudinary.com/your-cloud/video/upload/v1234567890/rdd-predict/xyz789.mp4",
  "cloudinary_public_id": "rdd-predict/xyz789",
  "filename": "a1b2c3d4-5678-90ab-cdef-1234567890ab_processed.mp4",
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
    },
    {
      "frame": 15,
      "timestamp": 0.5,
      "detections": [
        {
          "class": "D00",
          "confidence": 0.8756,
          "bbox": [150.0, 180.0, 380.0, 420.0]
        },
        {
          "class": "D10",
          "confidence": 0.7234,
          "bbox": [500.0, 300.0, 700.0, 500.0]
        }
      ],
      "frame_url": "https://res.cloudinary.com/.../frame_15.jpg",
      "frame_public_id": "rdd-predict/frame_15"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | Status proses: `"success"` |
| `file_url` | `string` | URL lokal file hasil proses |
| `image` / `video` | `string` | Tipe hasil: `"image"` atau `"video"` |
| `cloudinary_url` | `string` | URL Cloudinary untuk file hasil |
| `cloudinary_public_id` | `string` | Public ID Cloudinary |
| `filename` | `string` | Nama file hasil |
| `metadata` | `object` | Metadata file (dimensi, FPS untuk video) |
| `data_summary` | `string` | Ringkasan hasil deteksi |
| `data` | `array` | Detail deteksi (lihat tabel dibawah) |

**Image Detection Object:**

| Field | Type | Description |
|-------|------|-------------|
| `class` | `string` | Kelas kerusakan: `D00`, `D10`, `D20`, `D40` |
| `confidence` | `number` | Confidence score (0-1) |
| `bbox` | `array` | Bounding box `[x1, y1, x2, y2]` |

**Video Detection Object:**

| Field | Type | Description |
|-------|------|-------------|
| `frame` | `number` | Nomor frame |
| `timestamp` | `number` | Timestamp dalam detik |
| `detections` | `array` | Array detection objects |
| `frame_url` | `string` | URL Cloudinary frame annotated |
| `frame_public_id` | `string` | Public ID Cloudinary frame |

**JavaScript Example:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();

if (result.status === 'success') {
  // Tampilkan hasil
  resultImage.src = result.cloudinary_url;
  
  // Proses deteksi
  result.data.forEach(detection => {
    console.log(`${detection.class}: ${(detection.confidence * 100).toFixed(1)}%`);
  });
}
```

---

## WebSocket Endpoints

### Base64 Protocol (Standar)

Standard streaming protocol menggunakan Base64 encoding.

#### Available Endpoints

| Endpoint | Description |
|----------|-------------|
| `ws://host/predict/stream/{device_id}` | Default model (pytorch) |
| `ws://host/predict/stream/{model_key}/{device_id}` | Model spesifik |

**Model Keys:** `pytorch`, `tfrt-32`, `tfrt-16`, `tflite-32`, `tflite-16`

#### Client → Server (Request)

Kirim frame sebagai Base64 encoded JPEG string:

```
/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwME...
```

Atau dengan data URI prefix:

```
data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMD...
```

#### Server → Client (Response)

**Success Response:**
```json
{
  "device_id": "abc123",
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

| Field | Type | Description |
|-------|------|-------------|
| `device_id` | `string` | Device ID dari URL parameter |
| `status` | `string` | `"success"` atau `"error"` |
| `model` | `string` | Model yang digunakan |
| `frame_index` | `number` | Index frame (incremental) |
| `timestamp_ms` | `number` | Unix timestamp dalam milliseconds |
| `processing_latency_ms` | `number` | Waktu proses di server (ms) |
| `processed_frame` | `string` | Frame dengan anotasi (Base64 JPEG) |
| `detections` | `array` | Array objek deteksi |
| `detection_count` | `number` | Jumlah deteksi |

**Error Response:**
```json
{
  "status": "error",
  "model": "pytorch",
  "frame_index": 42,
  "error": "Invalid base64 image data: Incorrect padding"
}
```

#### JavaScript Example (Base64)

```javascript
class RDDStreamClient {
  constructor(modelKey = 'pytorch', deviceId = 'device-001') {
    this.modelKey = modelKey;
    this.deviceId = deviceId;
    this.ws = null;
    this.frameIndex = 0;
  }

  connect() {
    const wsUrl = `ws://${location.host}/predict/stream/${this.modelKey}/${this.deviceId}`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('Connected to RDD stream');
      this.startStreaming();
    };

    this.ws.onmessage = (event) => {
      const response = JSON.parse(event.data);
      this.handleResponse(response);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket closed');
    };
  }

  handleResponse(response) {
    if (response.status === 'success') {
      // Display annotated frame
      document.getElementById('resultImage').src = response.processed_frame;

      // Update statistics
      document.getElementById('latency').textContent = 
        `${response.processing_latency_ms.toFixed(1)}ms`;
      document.getElementById('detectionCount').textContent = 
        response.detection_count;

      // Process detections
      this.updateDetectionChart(response.detections);
    } else {
      console.error('Stream error:', response.error);
    }
  }

  sendFrame(canvas, quality = 0.5) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;

    canvas.toBlob(blob => {
      const reader = new FileReader();
      reader.onloadend = () => {
        // Send base64 without prefix
        const base64 = reader.result.split(',')[1];
        this.ws.send(base64);
      };
      reader.readAsDataURL(blob);
    }, 'image/jpeg', quality);
  }

  disconnect() {
    this.ws?.close();
  }
}

// Usage
const client = new RDDStreamClient('pytorch', 'my-device-001');
client.connect();
```

---

### Binary Protocol (High Performance)

Binary streaming untuk performa lebih tinggi (~33% bandwidth reduction).

#### Available Endpoints

| Endpoint | Description |
|----------|-------------|
| `ws://host/predict/stream-binary` | Default model |
| `ws://host/predict/stream-binary/{model_key}` | Model spesifik |

#### Client → Server (Request)

Kirim frame sebagai **raw JPEG bytes** (bukan Base64):

```javascript
// Dapatkan Blob dari canvas
canvas.toBlob(blob => {
  blob.arrayBuffer().then(buffer => {
    ws.send(buffer);  // Kirim raw bytes
  });
}, 'image/jpeg', 0.5);
```

#### Server → Client (Response)

Response dalam format binary:
```
[4 bytes: header length (uint32 LE)] + [JSON header] + [JPEG bytes]
```

**Parsing di JavaScript:**
```javascript
ws.onmessage = async (event) => {
  const buffer = await event.data.arrayBuffer();
  const view = new DataView(buffer);
  
  // Read header length (first 4 bytes, little-endian)
  const headerLength = view.getUint32(0, true);
  
  // Extract JSON header
  const headerBytes = new Uint8Array(buffer, 4, headerLength);
  const headerStr = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerStr);
  
  // Extract JPEG frame
  const jpegBytes = new Uint8Array(buffer, 4 + headerLength);
  const blob = new Blob([jpegBytes], { type: 'image/jpeg' });
  const frameUrl = URL.createObjectURL(blob);
  
  document.getElementById('resultImage').src = frameUrl;
};
```

**Header JSON Format:**
```json
{
  "status": "success",
  "model": "pytorch",
  "frame_index": 42,
  "timestamp_ms": 1702658400000,
  "processing_latency_ms": 18.32,
  "detections": [
    {
      "class": "D00",
      "confidence": 0.8542,
      "bbox": [120.5, 230.2, 450.8, 380.1]
    }
  ],
  "detection_count": 1
}
```

#### JavaScript Example (Binary)

```javascript
class RDDBinaryStreamClient {
  constructor(modelKey = 'pytorch') {
    this.modelKey = modelKey;
    this.ws = null;
  }

  connect() {
    const wsUrl = `ws://${location.host}/predict/stream-binary/${this.modelKey}`;
    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      console.log('Binary stream connected');
    };

    this.ws.onmessage = async (event) => {
      await this.handleBinaryResponse(event.data);
    };
  }

  async handleBinaryResponse(data) {
    const buffer = data;
    const view = new DataView(buffer);
    
    // Parse header length (first 4 bytes, little-endian)
    const headerLength = view.getUint32(0, true);
    
    // Parse JSON header
    const headerBytes = new Uint8Array(buffer, 4, headerLength);
    const header = JSON.parse(new TextDecoder().decode(headerBytes));
    
    if (header.status === 'success') {
      // Extract and display JPEG frame
      const jpegBytes = new Uint8Array(buffer, 4 + headerLength);
      const blob = new Blob([jpegBytes], { type: 'image/jpeg' });
      
      // Revoke previous URL to prevent memory leak
      if (this.currentFrameUrl) {
        URL.revokeObjectURL(this.currentFrameUrl);
      }
      
      this.currentFrameUrl = URL.createObjectURL(blob);
      document.getElementById('resultImage').src = this.currentFrameUrl;
      
      // Update stats
      this.updateStats(header);
    }
  }

  sendFrame(canvas, quality = 0.5) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;

    canvas.toBlob(blob => {
      blob.arrayBuffer().then(buffer => {
        this.ws.send(buffer);  // Send raw bytes
      });
    }, 'image/jpeg', quality);
  }

  disconnect() {
    if (this.currentFrameUrl) {
      URL.revokeObjectURL(this.currentFrameUrl);
    }
    this.ws?.close();
  }
}
```

---

## Data Types

### Detection Classes

Kelas kerusakan jalan yang dapat dideteksi:

| Code | Name | Description (ID) | Description (EN) |
|------|------|------------------|------------------|
| `D00` | Longitudinal Crack | Retakan memanjang sepanjang jalur roda | Crack along wheel path |
| `D10` | Transverse Crack | Retakan melintang tegak lurus jalan | Crack perpendicular to road |
| `D20` | Alligator Crack | Retakan fatigue berbentuk kulit buaya | Fatigue cracking pattern |
| `D40` | Pothole | Lubang pada permukaan jalan | Hole in road surface |

### Bounding Box Format

Bounding box menggunakan format `[x1, y1, x2, y2]`:

```
(x1, y1) ─────────────────┐
    │                     │
    │     DETECTION       │
    │                     │
    └───────────────── (x2, y2)
```

- `x1, y1`: Koordinat sudut kiri atas
- `x2, y2`: Koordinat sudut kanan bawah
- Koordinat dalam pixel relatif terhadap ukuran frame

---

## Error Handling

### HTTP Error Responses

| Status Code | Description |
|-------------|-------------|
| `400` | Bad Request - Invalid filename atau file type tidak didukung |
| `500` | Internal Server Error - Error saat memproses |

**Example Error Response:**
```json
{
  "detail": "Unsupported file type. Use image (jpg, png) or video (mp4, avi)."
}
```

### WebSocket Error Responses

```json
{
  "status": "error",
  "model": "pytorch",
  "frame_index": 42,
  "error": "Error message description"
}
```

**Common Errors:**
- `"Invalid base64 image data"` - Frame tidak valid
- `"Model 'xxx' not loaded"` - Model tidak tersedia
- `"Failed to decode image"` - JPEG corrupt

### JavaScript Error Handling Example

```javascript
// REST API
try {
  const response = await fetch('/predict', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }
  
  const result = await response.json();
} catch (error) {
  console.error('API Error:', error.message);
}

// WebSocket
ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  
  if (response.status === 'error') {
    console.error(`Frame ${response.frame_index} error:`, response.error);
    // Optionally retry or skip frame
    return;
  }
  
  // Process successful response
};

ws.onerror = (event) => {
  console.error('WebSocket connection error');
};

ws.onclose = (event) => {
  if (!event.wasClean) {
    console.error('Connection lost, attempting reconnect...');
    setTimeout(() => connect(), 3000);
  }
};
```

---

## Quick Reference

### Endpoint Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/ping` | Health status |
| GET | `/models` | List available models |
| GET | `/stream/config` | Streaming configuration |
| POST | `/predict` | Upload & process media |
| WS | `/predict/stream/{device_id}` | Default stream |
| WS | `/predict/stream/{model_key}/{device_id}` | Model-specific stream |
| WS | `/predict/stream-binary` | Binary default stream |
| WS | `/predict/stream-binary/{model_key}` | Binary model-specific stream |

### Model Keys

| Key | Type | GPU Required |
|-----|------|--------------|
| `pytorch` | PyTorch | ❌ |
| `tfrt-32` | TensorRT FP32 | ✅ |
| `tfrt-16` | TensorRT FP16 | ✅ |
| `tflite-32` | TFLite FP32 | ❌ |
| `tflite-16` | TFLite FP16 | ❌ |
