[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_NOTEBOOK_LINK_HERE)

know you can also use in th evs code 

# 🎯 Face Detection & Recognition System

A complete end-to-end **Face Detection and Recognition** pipeline built in Python, running on **Google Colab** with GPU acceleration. The system can detect faces in images and videos, register known individuals into a persistent database, and identify them in new photos — all using pre-trained deep learning models with zero training required.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Cell-by-Cell Run Order](#cell-by-cell-run-order)
- [Tuning Guide](#tuning-guide)
- [Demo](#demo)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

---

## 📖 Overview

This project demonstrates a two-stage face AI pipeline:

1. **Face Detection** — Finds where faces are in an image using OpenCV's pre-trained SSD + ResNet-10 DNN model.
2. **Face Recognition** — Identifies *who* the detected faces belong to by comparing 128-dimensional face embeddings stored in a local database.

The entire pipeline runs inside **Google Colab** (free tier supported; T4 GPU recommended for speed).

---

## ✨ Features

- 🔍 **Face Detection** using OpenCV DNN (SSD ResNet-10 Caffe model)
- 🧠 **Face Recognition** using the `face_recognition` library (dlib-based 128D embeddings)
- 🗃️ **Persistent Face Database** — saves and auto-reloads registered faces via `.pkl` file (survives Colab disconnections)
- 📸 **Register Anyone** — load a person's photo from a URL or upload directly from your device
- 🎥 **Video Processing** — run the full pipeline on video files frame-by-frame
- 📊 **Confidence Scores** — every detection and recognition result includes a confidence/distance score
- 🎛️ **Tunable Parameters** — adjust `confidence_threshold` and `tolerance` to balance accuracy vs. recall
- 🖼️ **Inline Visualization** — annotated images displayed directly inside the notebook

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `opencv-python-headless` | Face detection (DNN), image processing, video I/O |
| `face_recognition` | 128D face embedding generation & matching |
| `dlib` | Backbone for `face_recognition` |
| `numpy` | Array operations |
| `matplotlib` | Inline image display inside Colab |
| `Pillow` | Image loading and format handling |
| `pickle` | Persistent face database serialization |
| `cmake` | Required build dependency for `dlib` |

**Pre-trained Model:**
- `res10_300x300_ssd_iter_140000.caffemodel` — OpenCV's ResNet-10 SSD face detector (~10 MB)
- `deploy.prototxt` — Network architecture blueprint

---

## 📁 Project Structure

```
face_project/
│
├── models/
│   ├── deploy.prototxt                          # DNN model architecture
│   └── res10_300x300_ssd_iter_140000.caffemodel # Pre-trained DNN weights
│
├── known_faces/                                 # (Optional) Store reference photos here
│
├── test_images/                                 # Test images downloaded or uploaded
│
└── face_database.pkl                            # Auto-saved face embeddings database
```

---

## ⚙️ How It Works

### Stage 1 — Face Detection

```
Input Image
     │
     ▼
Resize to 300×300  ──▶  Normalize pixels  ──▶  Create blob
     │
     ▼
Pass through SSD ResNet-10 DNN
     │
     ▼
Get bounding boxes + confidence scores
     │
     ▼
Filter by confidence_threshold (default: 0.5)
     │
     ▼
Return: [(x, y, w, h, confidence), ...]
```

### Stage 2 — Face Recognition

```
Detected Face Region
     │
     ▼
Generate 128D face embedding  (via dlib / face_recognition)
     │
     ▼
Compare against all stored embeddings in database
     │
     ▼
Find closest match using Euclidean distance
     │
     ▼
If distance < tolerance  →  Return person's name + distance
If distance ≥ tolerance  →  Return "Unknown"
```

### Database (Persistence)

The face database is stored as a `.pkl` file. Every time you register a new person, the database is saved to disk. On the next Colab session, it **auto-loads** — you don't need to re-register everyone from scratch.

---

## 🚀 Getting Started

### 1. Open in Google Colab

Upload `Face_Detection___Recognition.ipynb` to [Google Colab](https://colab.research.google.com/) or open it directly from your Google Drive.

> **Recommended:** Enable GPU — `Runtime → Change runtime type → T4 GPU`

### 2. Install Dependencies (Cell 1)

```python
!pip install face_recognition
!pip install cmake
!pip install opencv-python-headless
!pip install numpy
!pip install matplotlib
!pip install pillow
```

> ⚠️ `dlib` (installed as a dependency of `face_recognition`) requires `cmake` to be installed first.

### 3. Run Cells in Order

Follow the [Cell-by-Cell Run Order](#cell-by-cell-run-order) below.

---

## 📋 Cell-by-Cell Run Order

| Cell | Purpose |
|------|---------|
| **Cell 1** | Install all required libraries *(run once per session)* |
| **Cell 2** | Import libraries and create project folder structure |
| **Cell 3** | Download pre-trained DNN model files from OpenCV GitHub |
| **Cell 4** | Load DNN model into memory as a `cv2.dnn.Net` object |
| **Cell 5** | Define `show_image()` helper for inline display |
| **Cell 6** | Define `detect_faces()` — the core detection function |
| **Cell 7** | Define `draw_boxes()` — draws bounding boxes and labels on images |
| **Cell 8** | Sanity check — test the detector on a sample image |
| **Cell 9** | Define `FaceRecognizer` class with register/identify methods |
| **Cell 10** | Register known people (Barack Obama, Elon Musk, etc.) from URLs |
| **Cell 11** | Define and run `process_image()` — the full detection + recognition pipeline |
| **Cell 12** | Test recognition on a different photo of a registered person |
| **Cell 13** | Upload your own photo and register yourself into the database |
| **Cell 14** | Test recognition on your own photo |
| **Cell 15** | *(Optional)* Process a full video file frame-by-frame |
| **Cell 16** | Tuning reference guide |

> 💡 **After a Colab disconnect:** Re-run Cells 1 → 9. The database auto-loads from the `.pkl` file — your registered faces are preserved!

---

## 🎛️ Tuning Guide

Fine-tune these two parameters to get optimal results for your use case:

### `confidence_threshold` — controls face *detection* sensitivity

| Problem | Fix |
|---------|-----|
| Face not being detected (no box drawn) | **Lower** the threshold → try `0.4`, `0.3`, `0.2` |
| Too many false detections (boxes on non-faces) | **Raise** the threshold → try `0.6`, `0.7` |

### `tolerance` — controls face *recognition* strictness

| Problem | Fix |
|---------|-----|
| Known person showing as "Unknown" | **Raise** tolerance → try `0.55`, `0.6` |
| Wrong person being identified | **Lower** tolerance → try `0.45`, `0.4` |
| Poor accuracy in general | Register **more photos** per person (3–5 ideal); use varied angles and lighting |

**Recommended defaults:**
```python
confidence_threshold = 0.5   # detection sensitivity
tolerance = 0.55             # recognition strictness
```

---

## 🧪 Demo

The notebook includes a built-in demo with publicly available photos of well-known individuals used purely for testing purposes:

- **Barack Obama** — registered with 2 photos, tested with a 3rd
- **Elon Musk** — registered and verified
- **Custom user** — upload your own photo and register yourself (Cell 13 & 14)

Registering yourself only takes a few seconds — the system encodes your face and saves it to the database permanently.

---

## ⚠️ Known Limitations

- Runs best on **Google Colab** (some cells use Colab-specific APIs like `google.colab.files` for uploads)
- Face detection accuracy can drop with **very small faces**, **extreme angles**, or **poor lighting**
- The `face_recognition` library's `dlib` backend can be slow to install on some systems
- Database is stored locally in Colab's `/content/` directory — it **does not persist** across Colab sessions unless you save the `.pkl` file to Google Drive

---

## 🔮 Future Improvements

- [ ] Save `face_database.pkl` directly to Google Drive for full persistence
- [ ] Build a real-time webcam recognition stream
- [ ] Add a web UI using Gradio or Streamlit
- [ ] Support batch registration from a folder of images
- [ ] Integrate anti-spoofing to detect printed photos
- [ ] Export results as annotated video with embedded labels

---

## 📄 License

This project is for educational purposes. The pre-trained model files are sourced from the [OpenCV GitHub repository](https://github.com/opencv/opencv) and are subject to their respective licenses.

---

## 🙏 Acknowledgements

- [OpenCV DNN Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn) — SSD ResNet-10 Caffe model
- [face_recognition by ageitgey](https://github.com/ageitgey/face_recognition) — dlib-based face embedding library
- [Google Colab](https://colab.research.google.com/) — free cloud GPU environment

---

*Built with ❤️ using Python, OpenCV, and dlib*
