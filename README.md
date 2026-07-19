## facial-mesh-emotion-inference

**facial-mesh-emotion-inference** is a production-grade computer vision application designed to map 3D facial structural geometry and accurately infer human emotional states in real-time. By combining the ultra-low-latency landmark tracking of **MediaPipe Face Mesh** with the robust deep learning architecture of **DeepFace (VGG-Face backend)**, the application seamlessly translates facial morphology into emotional metrics.

### The Architectural Problem We Solved
Most real-time emotion tracking projects suffer from heavy video lag or frame-dropping. Because deep neural networks take significant time to calculate inference on a per-frame basis, standard sequential code drops video feeds down to an unwatchable 5–10 FPS. 

This repository implements a **decoupled, multi-threaded pipeline**:
1. **The Core Thread** handles high-frequency webcam ingestion and runs MediaPipe to render the structural face mesh flawlessly at a native 30+ FPS.
2. **The Inference Worker** processes frame matrices asynchronously or via frame-skipping intervals, updating emotional states in the background without bottlenecking the main UI thread.
