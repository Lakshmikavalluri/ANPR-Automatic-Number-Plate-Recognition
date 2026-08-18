# ANPR – Automatic Number Plate Recognition

A real-time **Automatic Number Plate Recognition (ANPR)** system that detects vehicle license plates using **YOLOv8**, extracts plate numbers using **OCR**, and validates detected plates against a predefined list.

## 🚀 Project Overview

This project implements an end-to-end ANPR pipeline for detecting and recognizing vehicle license plates from images and video streams.

The system combines **YOLOv8 object detection** with **OCR-based text recognition** to automatically identify license plates and validate them against pre-booked/authorized plate numbers.

## ✨ Features

* Real-time license plate detection using **YOLOv8**
* License plate text extraction using **OCR**
* Image preprocessing for improved OCR results
* Plate number validation using a predefined whitelist
* Support for image/video-based detection
* End-to-end detection → OCR → validation pipeline
* API integration using **Flask**
* Optimized for reliable detection under varying conditions

## 🛠️ Technologies Used

* **Python**
* **YOLOv8 / Ultralytics**
* **OpenCV**
* **Tesseract OCR / EasyOCR**
* **Flask**
* **NumPy**
* **Regular Expressions (Regex)**
* **Git & GitHub**

## 🧠 System Workflow

```text
Input Image / Video / Camera
            ↓
     Image Preprocessing
            ↓
       YOLOv8 Detection
            ↓
      License Plate Crop
            ↓
       OCR Processing
            ↓
    Text Cleaning & Regex
            ↓
     Plate Number Validation
            ↓
   Authorized / Unauthorized
```

## 📊 Model Performance

The YOLOv8 model was trained on an annotated license plate dataset and achieved approximately **85% detection accuracy**.

The model was optimized to provide reliable license plate detection under different image and environmental conditions.

## 📁 Project Structure

```text
ANPR-Automatic-Number-Plate-Recognition/
│
├── dataset/
│   └── Training and validation dataset
│
├── runs/
│   └── detect/
│       └── Model training results
│
├── api.py
│   └── Flask API for ANPR processing
│
├── detect.py
│   └── License plate detection and OCR
│
├── normal.py
│   └── Image/video processing utilities
│
├── dataset.txt
│   └── Dataset configuration
│
├── prebooked_plates.txt
│   └── Authorized plate numbers
│
├── requirements.txt
│   └── Python dependencies
│
└── yolov8n.pt
    └── YOLOv8 pretrained/model weights
```

## 🔍 How It Works

### 1. License Plate Detection

The trained **YOLOv8 model** identifies license plate regions in the input image or video.

### 2. Image Preprocessing

The detected license plate is cropped and processed using image-processing techniques to improve the quality of the input provided to the OCR system.

### 3. OCR Recognition

OCR is applied to the cropped license plate to extract the characters from the plate.

### 4. Text Processing

The extracted text is cleaned and processed using **Regex-based validation** to improve recognition reliability and handle unwanted OCR characters.

### 5. Plate Validation

The recognized plate number is compared against the entries in `prebooked_plates.txt`.

```text
Detected Plate
      ↓
OCR Text
      ↓
Text Cleaning
      ↓
Compare with Whitelist
      ↓
Authorized → Access Granted
Unauthorized → Access Denied
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Lakshmikavalluri/ANPR-Automatic-Number-Plate-Recognition.git
```

Navigate to the project directory:

```bash
cd ANPR-Automatic-Number-Plate-Recognition
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

For detection:

```bash
python detect.py
```

For the Flask API:

```bash
python api.py
```

Make sure the required model weights and configuration files are available in the project directory.

## 📌 Applications

* Smart parking systems
* Automated vehicle entry systems
* Toll collection systems
* Parking access control
* Security and surveillance
* Vehicle monitoring systems

## 🔮 Future Improvements

* Improve detection accuracy with a larger and more diverse dataset
* Add real-time CCTV/IP camera support
* Improve OCR accuracy for different plate conditions
* Deploy the system on a cloud server
* Integrate automatic barrier control
* Add database storage for detected vehicles
* Develop a web-based monitoring dashboard

## 👩‍💻 Author

**Lakshmika Valluri**

GitHub: [Lakshmikavalluri](https://github.com/Lakshmikavalluri)

## ⭐ Project Highlights

* Built a complete **YOLOv8 + OCR ANPR pipeline**
* Achieved approximately **85% license plate detection accuracy**
* Implemented **real-time plate detection and recognition**
* Integrated **Flask API** for application-level access
* Added **authorized plate validation**
* Optimized the pipeline for practical usage under varying conditions
