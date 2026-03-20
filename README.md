##🌿 Plant Disease Detection using CNN
---
A Deep Learning project built for Dexterix‑4.0 Hackathon, capable of detecting plant leaf diseases using image classification. The model is trained using Kaggle’s Plant Village dataset and deployed via a simple Python application.

## 📌 Project Structure
```project/
│
├── home_page.jpeg
├── main.py
├── model.h5
├── trained_plant_disease_model.keras
├── training_hist.json
│
├── Train_plant_disease.ipynb
├── Test_plant_disease.ipynb
│
├── train/        # Training dataset
│
├── valid/        # Validation dataset
│
└── test/         # Testing dataset
```

## 📦 Dataset
---
Download Plant Village dataset from Kaggle:
🔗 https://www.kaggle.com/datasets/emmarex/plantdisease
After downloading, split folders into:

-- train/
-- valid/
-- test/

-- You can use tools like splitfolders for automated splitting.

## 🧠 Model
-- The project uses a Convolutional Neural Network (CNN) for classification.
Trained model is available here:
## 📥 Download Trained Model:
-- https://www.mediafire.com/file/ig4am4208ckbrk1/trained_plant_disease_model.keras/file

## 🚀 How to Run the Project
-- 1️⃣ Install Dependencies
```
pip install tensorflow keras numpy pandas matplotlib pillow flask
```
-- 2️⃣ Run Model Training (Optional)
```
jupyter notebook Test_plant_disease.ipynb
```
-- 3️⃣ Test the Model
```
jupyter notebook Test_plant_disease.ipynb
```
-- 4️⃣ Run the Application
```
python main.py
```

## 🖼️ Web Demo (Reference from Dexterix Hackathon)
-- Demo inspiration:
- 🔗 https://tecresearch.github.io/plant-disease-detection/

- 📊 Outputs Generated

- trained_plant_disease_model.keras → Final trained model
- model.h5 → Backup model
- training_hist.json → Model training history
- home_page.jpeg → UI / Landing page image (optional)


## 🏗️ Tech Stack

-- Python
-- TensorFlow / Keras
-- NumPy, Pandas
-- Flask / Streamlit (as UI)
-- Matplotlib
-- OpenCV / Pillow


-- 📸 Sample Prediction (from Test Notebook)
-- The test script takes a leaf image → preprocesses → predicts disease → outputs class name.

## 🏁 Conclusion
-- This project demonstrates a robust approach to automatically classify plant leaf diseases with high accuracy using CNNs, contributing to modern agriculture solutions.

```
Developed by Mr Brijesh Nishad
```
