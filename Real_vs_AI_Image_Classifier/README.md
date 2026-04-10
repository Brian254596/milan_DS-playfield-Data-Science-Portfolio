# 🧠 Real vs AI Image Classifier

This project is a deep learning application that classifies images as either **Real** or **AI-Generated** using a Convolutional Neural Network (CNN) with transfer learning.

It is deployed as an interactive web app using **Streamlit Cloud**.

🔗 **Live App:** https://ai-vs-real-app-pwewihktyxqvgb2ov7aclh.streamlit.app/

---

## 🚀 Project Overview

With the rapid rise of AI-generated content, distinguishing between real and synthetic images has become increasingly important across industries such as media, journalism, cybersecurity, and digital forensics.

This project builds a robust machine learning system that:
- Detects whether an image is **real** or **AI-generated**
- Provides fast and interactive predictions via a web app
- Demonstrates a complete **end-to-end machine learning pipeline**
- Showcases practical deployment using modern tools

---

## 📊 Project Workflow

### 1️⃣ Data Collection
- Collected datasets consisting of:
  - Real-world images
  - AI-generated images
- Ensured diversity in content (faces, objects, scenes)
- Maintained class balance to reduce bias

---

### 2️⃣ Data Preprocessing
- Resized all images to **224 × 224 pixels**
- Normalized pixel values to range **[0, 1]**
- Removed corrupted or unreadable images
- Applied **data augmentation** to improve generalization:
  - Horizontal flipping
  - Rotation
  - Zooming
  - Shifting

---

### 3️⃣ Model Building
- Used **MobileNetV2** (pretrained on ImageNet) as the base model
- Leveraged **transfer learning** for efficiency and performance
- Added custom layers:
  - Global Average Pooling layer
  - Dense (Fully Connected) layers
  - Dropout layers to prevent overfitting

---

### 4️⃣ Model Training
- Loss Function: **Binary Crossentropy**
- Optimizer: **Adam**
- Metrics: **Accuracy**
- Applied **class weights** to handle dataset imbalance
- Trained over multiple epochs with validation monitoring

---

### 5️⃣ Model Fine-Tuning
- Unfroze the top layers of MobileNetV2
- Reduced learning rate for stable training
- Improved feature extraction and model generalization
- Achieved better performance on validation data

---

### 6️⃣ Model Evaluation
- Achieved approximately **94% accuracy**
- Evaluated using:
  - Training vs Validation Accuracy
  - Loss curves
- Observed:
  - Strong learning performance
  - Minimal overfitting
  - Good generalization on unseen data

---

### 7️⃣ Model Saving
- Saved trained model in **HDF5 format (`.h5`)** for compatibility and stability
- Saved class labels using **pickle (`labels.pkl`)**

---

### 8️⃣ Deployment (Streamlit Cloud)
- Built an interactive UI using **Streamlit**
- Users can:
  - Upload an image
  - Instantly receive prediction:
    - **"THE IMAGE IS REAL"**
    - **"THE IMAGE IS AI GENERATED"**
- Optimized model loading using caching for faster performance

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pillow (PIL)**
- **Streamlit**
- **Git & GitHub**

---

## 📂 Project Structure

```
AI_vs_Real_Deployment/
│
├── app.py                  # Streamlit application
├── ai_vs_real_model.h5     # Trained deep learning model
├── labels.pkl              # Class labels
├── requirements.txt        # Dependencies
├── runtime.txt             # Python version
├── .gitignore              # Ignored files
└── README.md               # Project documentation
```

---

## 💡 Key Features

- End-to-end machine learning pipeline (**data collection → preprocessing → modeling → deployment**)
- Transfer learning using **MobileNetV2** for efficient and accurate predictions
- Lightweight model optimized for fast inference
- Interactive and user-friendly web interface built with **Streamlit**
- Real-world application in detecting **AI-generated vs real images**

---

## 🔮 Future Improvements

- Enhance dataset diversity to improve model robustness
- Introduce prediction confidence visualization
- Enable batch image uploads for bulk classification
- Deploy using containerization tools like **Docker**
- Integrate cloud deployment solutions (**AWS / GCP**)
- Add model explainability techniques such as **Grad-CAM**

---

## 👤 Author

**Brian Kipchumba**  
*Statistician & Data Science Enthusiast*  

- Skilled in **Machine Learning, Data Analysis, and Statistical Modeling**
- Experienced in building and deploying data-driven applications
- Passionate about leveraging data to solve real-world problems

---

## ⭐ Final Note

This project highlights the practical integration of **statistics, deep learning, and deployment technologies** to build impactful real-world solutions.

If you found this project useful or interesting, consider giving it a ⭐ on GitHub!
