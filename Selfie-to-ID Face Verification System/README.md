# 🪪 Selfie-to-ID Face Verification System

This project is a deep learning application that verifies whether a selfie and an official ID document image belong to the same person using pretrained FaceNet embeddings and machine learning classifiers.

Built for **KYC (Know Your Customer)** applications in banking, fintech, and digital onboarding platforms.

---

## 🚀 Project Overview

With the rapid growth of digital banking and online onboarding, automated identity verification has become a critical component of fraud prevention. Verifying that a live selfie matches an official identity document — such as a passport, national ID, or driver's license — is a key step in the KYC process.

This project builds a robust face verification system that:

- Determines whether a selfie and an ID document belong to the same person
- Handles real-world variations in lighting, device quality, and aging
- Evaluates bias and fairness across demographic groups
- Demonstrates a complete end-to-end machine learning pipeline

---

## 📊 Project Workflow

### 1️⃣ Data Collection
- Used the **AxonLabs Diverse Selfie & ID Photo Dataset** (public sample)
- 10 unique individuals, 154 images across 3 image types
- Each person provided 2 official ID photos, 4 current selfies, and 10 archive selfies
- Metadata included gender, ethnicity, age, device OS, lighting condition, and temporal gap

### 2️⃣ Exploratory Data Analysis
- Analyzed demographic distributions across gender, ethnicity, and age groups
- Explored image type breakdown and lighting condition coverage
- Visualized sample images per user across all image types
- Performed temporal gap analysis on archive selfies (up to 7 years old)

### 3️⃣ Data Preparation
- Applied **OpenCV Haar Cascade** face detection to all 154 images
- Cropped detected faces with 20% padding and resized to **160×160 pixels**
- Successfully cached **146/154** faces (8 failed detection)
- Generated **504 balanced pairs**:
  - Positive pairs (same person): 252
  - Negative pairs (different person): 252
- Split by `user_id` to prevent data leakage:
  - Train: 376 pairs (7 users)
  - Validation: 48 pairs (1 user)
  - Test: 80 pairs (2 users)

### 4️⃣ Feature Extraction
- Loaded pretrained **FaceNet (InceptionResnetV1, VGGFace2)** from `facenet-pytorch`
- Extracted **512-dimensional face embeddings** for each cropped face
- Same-person pairs achieved mean cosine similarity of **0.62** vs **0.05** for different-person pairs
- Engineered **1027 distance features** per pair:
  - L1 absolute difference
  - Squared difference
  - Dot product
  - Euclidean distance
  - Cosine similarity
- Normalized features using StandardScaler

### 5️⃣ Model Building
- Trained and compared 4 classifiers on engineered features:
  - Logistic Regression ✅ *(best)*
  - SVM (RBF)
  - Random Forest
  - Gradient Boosting
- Selected **Logistic Regression** based on validation accuracy and interpretability

### 6️⃣ Model Evaluation
- Achieved **93.75% test accuracy** and **99.62% AUC**
- Evaluated using:
  - Confusion matrix
  - Classification report (precision, recall, F1)
  - ROC curves for all models
  - Cosine similarity distribution (same vs different pairs)

### 7️⃣ Bias & Fairness Analysis
- Evaluated performance across gender, ethnicity, age group, selfie type, and device OS
- Equal performance on current and archive selfies — model is robust to aging
- 10% accuracy gap between some demographic groups — requires investigation at scale

### 8️⃣ Model Saving
- Saved trained classifier as `best_face_verification_model.pkl`
- Saved fitted scaler as `feature_scaler.pkl`

---

## 📈 Model Comparison

| Model | Val Accuracy | Test Accuracy | AUC |
|---|---|---|---|
| **Logistic Regression** | **100%** | **93.75%** | **99.62%** |
| SVM (RBF) | 100% | 91.25% | 100% |
| Random Forest | 100% | 93.75% | 99.00% |
| Gradient Boosting | 97.92% | 96.25% | 99.00% |

---

## ⚖️ Bias & Fairness Results

| Group | Accuracy | F1 | AUC |
|---|---|---|---|
| Male | 98% | 0.98 | 1.00 |
| Female | 88% | 0.87 | 0.99 |
| East Asian | 98% | 0.98 | 1.00 |
| Caucasian | 88% | 0.87 | 0.99 |
| Android | 98% | 0.98 | 1.00 |
| iOS | 88% | 0.87 | 0.99 |
| Current Selfie | 94% | 0.94 | 1.00 |
| Archive Selfie | 94% | 0.94 | 1.00 |

---

## 🛠️ Technologies Used

- Python 3.10
- PyTorch & facenet-pytorch
- OpenCV
- scikit-learn
- pandas & numpy
- matplotlib & seaborn
- Jupyter Notebook
- Anaconda

---

## 📂 Project Structure

```
Selfie-to-ID Face Verification System/
│
├── Data/
│   ├── Selfie & id data - public sample/
│   │   ├── 1/
│   │   │   ├── docs/                        # Official ID document photos
│   │   │   ├── selfies/                     # Current selfies (4 lighting conditions)
│   │   │   └── archive_selfies/             # Archive selfies (old & recent)
│   │   └── ...
│   ├── metadata_images.csv                  # Image metadata & demographics
│   └── meta_public.json                     # Full metadata JSON
│
├── Selfie_to_ID.ipynb                       # Main project notebook
├── best_face_verification_model.pkl         # Saved best classifier
├── feature_scaler.pkl                       # Fitted StandardScaler
├── best_siamese_model.keras                 # Siamese network experiment
└── README.md
```

---

## 💡 Key Features

- End-to-end KYC face verification pipeline (data → embeddings → classification → fairness)
- Pretrained FaceNet embeddings for robust face representation without large training data
- Balanced pair generation with positive and negative samples
- Comprehensive bias and fairness evaluation across demographic groups
- User-split train/val/test strategy to prevent data leakage
- Lightweight and interpretable Logistic Regression classifier on deep embeddings

---

## ⚠️ Limitations

- Public sample contains only 10 users — full dataset needed for production-level evaluation
- No liveness detection — system cannot distinguish a live face from a printed photo
- OpenCV Haar Cascade failed on 8/154 images in low-light or off-angle conditions
- Only 2 ethnic groups represented in the test set
- Fixed 0.5 decision threshold applied across all demographic groups

---

## 🔮 Future Improvements

- Train on the full 12,000-user dataset for production-ready performance
- Upgrade face detection to RetinaFace or MTCNN for better accuracy
- Integrate liveness detection (anti-spoofing) for real-world KYC deployment
- Tune per-group decision thresholds to reduce the observed fairness gap
- Build a REST API using FastAPI or Flask for real-time verification
- Evaluate across all 5+ ethnic groups in the full dataset
- Train an end-to-end Siamese Network with ArcFace loss on the full dataset
- Deploy as an interactive web app using Streamlit

---

## 👤 Author

**Brian Kipchumba**
Statistician & Data Science Enthusiast

- Skilled in Machine Learning, Deep Learning, and Statistical Modeling
- Experienced in building end-to-end data science pipelines
- Passionate about using AI to solve real-world identity and security problems

---

## ⭐ Final Note

This project demonstrates the practical integration of deep learning, computer vision, and fairness analysis to build an accurate and interpretable face verification system for KYC applications.

If you found this project useful or interesting, consider giving it a ⭐ on GitHub!