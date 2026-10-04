[README.md](https://github.com/user-attachments/files/33024411/README.md)
# Brain Tumor MRI Image Classification

Deep-learning pipeline that classifies brain MRI scans into **glioma, meningioma, pituitary, or no tumor**,
with a custom CNN built from scratch and transfer-learning models (MobileNetV2, ResNet50, EfficientNetB0),
deployed through a Streamlit web app.

## Project structure
```
├── brain_tumor_mri_classification.ipynb   # EDA, training, evaluation, comparison
├── app.py                                 # Streamlit application
├── best_model.h5                          # best-performing model
├── class_names.json
├── model_comparison.csv
├── saved_models/                          # all trained models (.h5)
└── requirements.txt
```

## Workflow
1. Dataset exploration (class balance, resolution, sample images)
2. Preprocessing: resize 224x224, normalize to [0, 1]
3. Augmentation: flips, rotation, zoom, shifts, brightness
4. Custom CNN with BatchNorm + Dropout
5. Transfer learning (frozen head training, then fine-tuning top layers)
6. Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
7. Evaluation: accuracy, precision, recall, F1, confusion matrix, history plots
8. Model comparison and deployment of the best model

## Results
See `model_comparison.csv` (fill in your table here after running the notebook).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
[Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Disclaimer
Educational project only. Not intended for clinical use.
