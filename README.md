# HelmNet: Safety Helmet Detection with CNNs and Transfer Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Sachin-Abishek/helmet_detection/blob/main/HelmNet.ipynb)

An image classification project that detects whether a worker is wearing a safety helmet. It compares a CNN built from scratch with three transfer-learning variants of VGG-16 on a small dataset of 631 images.

## Problem
On construction sites and in industrial plants, helmets prevent serious head injuries, but checking compliance by hand is slow and error-prone in large operations. The goal is a model that classifies each image as:

- **With Helmet** (label `1`)
- **Without Helmet** (label `0`)

## Dataset
- 631 RGB images, each 200 x 200 x 3, stored as `images_proj.npy` with labels in `Labels_proj.csv`
- Nearly balanced: 311 with helmet, 320 without helmet
- Varied settings (construction sites, factories, industrial areas) with different lighting, angles and worker postures
- The data files are **not included** in this repository

**Split (stratified, `random_state=42`):** 504 train / 63 validation / 64 test (80 / 10 / 10)

## Approach
1. **EDA:** sample images per class, class balance check
2. **Preprocessing:** pixel normalization (`/255`) and VGG-16 `preprocess_input`. A grayscale conversion is demonstrated, but the models are trained on RGB.
3. **Models:** four architectures compared on train and validation metrics (accuracy, recall, precision, F1) and confusion matrices
4. **Selection:** the best model is evaluated once on the held-out test set

| # | Model | Details |
|---|---|---|
| 1 | **Simple CNN** | 3 Conv blocks (64, 32, 32 filters) + MaxPooling, Dense(100), sigmoid output. 2.03M parameters. SGD (lr 0.01, momentum 0.9), 15 epochs. |
| 2 | **VGG-16 (Base)** | Frozen ImageNet VGG-16 + Global Average Pooling + sigmoid output. 513 trainable parameters. Adam, 20 epochs. |
| 3 | **VGG-16 (Base + FFNN)** | Frozen VGG-16 + Dense(32) + Dropout(0.5) + sigmoid output. 16,449 trainable parameters. |
| 4 | **VGG-16 (Base + Data Augmentation)** | As model 2, with rotation, shifts, shear and zoom applied to the training images only. |

## Results

| Model | Train accuracy | Validation accuracy | Test accuracy |
|---|---|---|---|
| Simple CNN | 96.8% | 96.8% | **92.2%** (F1 0.921) |
| VGG-16 (Base) | 100% | 100% | not evaluated |
| VGG-16 (Base + FFNN) | 100% | 100% | not evaluated |
| VGG-16 (Base + Data Aug) | 100% | 100% | not evaluated |

The Simple CNN was chosen as the final model and scored 92.2% accuracy on the 64 test images. The notebook describes the VGG-16 models as overfitting, since they reach 100% on both training and validation data.

## Tech stack
Python, TensorFlow / Keras, scikit-learn, OpenCV, NumPy, pandas, seaborn, matplotlib. The notebook was written for Google Colab with a GPU runtime.

## How to run
1. Click the **Open in Colab** badge above and set the runtime to **GPU**.
2. Put `images_proj.npy` and `Labels_proj.csv` in your Google Drive, and update the two paths in the notebook:
   ```python
   image_data  = '/content/drive/MyDrive/dataset/Helmnet_Data/images_proj.npy'
   image_label = '/content/drive/MyDrive/dataset/Helmnet_Data/Labels_proj.csv'
   ```
3. Run all cells. A fixed seed (`800`) is set for reproducibility.

## Limitations
- **Very small dataset.** The validation and test sets hold only 63 and 64 images, so each mistake moves accuracy by about 1.5 points. The 100% validation scores are not strong evidence.
- **VGG-16 models were never tested.** Only the Simple CNN was scored on the test set, so the models are not compared on equal terms.
- **The "overfitting" label is not proven.** Training and validation scores are identical for the VGG-16 models. This points to a small, easy validation set as much as to overfitting.
- **Not tuned.** There was no hyperparameter search, no fine-tuning of VGG-16 layers, and no cross-validation.
- **Classification only.** The model labels a whole image and cannot find or count workers, or handle several people in one frame.
- **Limited data coverage.** No analysis of performance by lighting, distance or helmet colour.

## Future improvements
- Evaluate every model on the same test set and use k-fold cross-validation
- Fine-tune the top VGG-16 layers, and try ResNet, MobileNet or EfficientNet
- Add more varied images and stronger augmentation
- Move to object detection (YOLO) so the system can find several workers per frame
- Wrap the best model in a small app (Streamlit or FastAPI) that takes an image or camera feed

## Author
**Sachin**: [LinkedIn](<your-linkedin-link>) | [GitHub](https://github.com/<your-username>)
