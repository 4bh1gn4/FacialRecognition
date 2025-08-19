# Face Recognition with OpenCV (LBPH)

A simple face recognition app built with Python and OpenCV using the **LBPH (Local Binary Patterns Histograms)** algorithm.  
This app allows you to **enroll faces** and **recognize people in real-time** from your webcam.

## 🚀 Features
- Face enrollment with automatic image capture (30 samples per person by default)
- Real-time face detection and recognition
- Adjustable confidence threshold for recognition strictness
- Saves and loads trained models (`lbph_model.xml`)
- Simple storage of face images and labels

## 🛠️ Technologies Used
- **Python 3**
- **OpenCV** (cv2)
- **NumPy**
- **JSON** for label mapping

## 📂 Project Structure
```
faces_db/          # Stores captured face images (per person)
lbph_model.xml     # Trained LBPH model file
labels.json        # Maps numeric labels to names
face_recognizer.py # Main script
```

## ⚙️ Setup Instructions
1. Clone this repository and install dependencies:
   ```bash
   pip install opencv-python opencv-contrib-python numpy
   ```

2. Run the app:
   ```bash
   python face_recognizer.py
   ```

3. Use the keyboard controls:
   - **N** → Enroll a new person  
   - **+ / -** → Adjust recognition threshold  
   - **Q** → Quit

## 🎯 How It Works
1. **Enrollment:** Captures 30 samples of a person's face, stores them in `faces_db/`, and retrains the LBPH model.  
2. **Recognition:** Detects faces in real-time and predicts names using the trained model.  
3. **Threshold:** Lower confidence values = better match. If confidence ≤ threshold, the person is recognized.

## 📌 Notes
- Ensure good lighting and multiple angles during enrollment for better accuracy.
- If the camera doesn’t open, try changing `CAM_INDEX` in the script (0, 1, or 2).

---
💡 Built as a simple demo app for learning computer vision and face recognition with OpenCV.

## 🙌 Credits
Developed by [@4bh1gn4](https://github.com/4bh1gn4)
Powered by OpenCV’s LBPH Face Recognizer

---

## 📜 License
MIT License

