# Siamese Neural Network for Image Recognition

This project implements a **Siamese Neural Network** to recognize and compare images by learning feature similarity. The architecture is inspired by models used in face verification and one-shot learning.

---

## 🧠 Project Overview

- **Objective:** Learn to determine whether two images are from the same class.
- **Architecture:** Shared-weight convolutional embedding network + distance computation + dense output.
- **Framework:** TensorFlow (Keras Functional API)

---

## 📂 Project Structure

project/
  dataset/               Image dataset downloaded from Kaggle
  siamese_model.py       Model definition and training script
  model.h                Saved trained model
  README.md              Project documentation

---

## 🛠️ Dependencies

Make sure you have the following Python packages installed:

import cv2  
import os  
import random  
import numpy as np  
from matplotlib import pyplot as plt  
import time  
import uuid  
from tensorflow.keras.models import Model, load_model  
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten  
import tensorflow as tf  

*Install them using your preferred method (for example, pip). Example:*  

*Install tensorflow, opencv-python, numpy, matplotlib.*

---

## 🏗️ Model Architecture

Below is the summary of the **embedding network** used to extract features from images:

| Layer (type)           | Output Shape           | Param #     |
|------------------------|------------------------|-------------|
| Input (100×100×3)      | (None, 100, 100, 3)    | 0           |
| Conv2D (64 filters)    | (None, 91, 91, 64)     | 19,264      |
| MaxPooling2D           | (None, 46, 46, 64)     | 0           |
| Conv2D (128 filters)   | (None, 40, 40, 128)    | 401,536     |
| MaxPooling2D           | (None, 20, 20, 128)    | 0           |
| Conv2D (128 filters)   | (None, 17, 17, 128)    | 262,272     |
| MaxPooling2D           | (None, 9, 9, 128)      | 0           |
| Conv2D (256 filters)   | (None, 6, 6, 256)      | 524,544     |
| Flatten                | (None, 9216)           | 0           |
| Dense                  | (None, 4096)           | 37,752,832  |

**Total Parameters:** 38,960,448  
**Trainable Parameters:** 38,960,448  
**Non-trainable Parameters:** 0  

---

## 🏃‍♂️ Quick Start

### 1️⃣ Clone the repository

*Download or clone this repository to your local machine.*

*Example: Clone using GitHub Desktop or run git clone.*

### 2️⃣ Install dependencies

*Make sure you have installed all required packages listed above.*

*Example: Install tensorflow, opencv-python, numpy, matplotlib.*

### 3️⃣ Download the dataset

*Download your image dataset from Kaggle and place it inside the dataset directory.*

*Ensure the directory structure is clear, for example:*

dataset/
  train/
    class1/
    class2/
  test/
    class1/
    class2/

### 4️⃣ Train the model

*Run the siamese_model.py script to start training.*

*After training, the model will be saved as model.h.*

*Example: Run python siamese_model.py.*

### 5️⃣ Load and use the trained model

*In your Python code, you can load the trained model like this:*

from tensorflow.keras.models import load_model  
model = load_model('model.h')  

*You can then pass image pairs to your model to predict similarity.*

---

## ✨ Reference

This implementation is based on the excellent tutorial by Nicholas Renotte:

[Siamese Neural Network in TensorFlow – YouTube Video](https://www.youtube.com/watch?v=LKispFFQ5GU&t=15401s)

---

## 📘 Additional Notes

- The model uses **contrastive loss** to learn embeddings that minimize distance for similar pairs and maximize distance for different pairs.
- Distance metrics like **L1 norm** or **L2 norm** can be used to compare feature vectors.
- You can customize the number of convolutional layers and filter sizes in siamese_model.py.

---

## 📬 Contact

*For questions, contributions, or collaborations, please reach out:*

- **MUKUL PALIWAL**
- *Email:* mukulpaliwal2023@gmail.com
- *GitHub:* MUKULPALIWAL13


