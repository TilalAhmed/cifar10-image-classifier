[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://tilalahmed-image-classification-cnn.hf.space)
# Image Classification on CIFAR-10 (ANN vs CNN)

A deep learning project that compares a baseline **Artificial Neural Network (ANN)** against a **Convolutional Neural Network (CNN)** for image classification on the CIFAR-10 dataset.

---

## Results

| Model | Accuracy |
|---|---|
| ANN (Baseline) | ~49% |
| CNN | ~68.9% |

The CNN achieves a **~20% improvement** over the ANN baseline by learning spatial features through convolutional layers.

---

## Dataset

**CIFAR-10** — 60,000 color images (32×32 px) across 10 classes, automatically downloaded via `tensorflow.keras.datasets`.

Classes: `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

No manual download needed.

---

## Project Structure

```
cifar10-image-classification/
├── src/
│   └── train.py          # ANN and CNN training script
├── notebooks/
│   └── Image_Classification_Using_Cnn.ipynb
├── models/               # Saved model appears here after training
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

**1. Clone and install dependencies**
```bash
git clone https://github.com/YOUR_USERNAME/cifar10-image-classification.git
cd cifar10-image-classification
pip install -r requirements.txt
```

**2. Train both models (ANN + CNN)**
```bash
python src/train.py
```

**3. Train only the CNN**
```bash
python src/train.py --model cnn
```

**4. Adjust epochs**
```bash
python src/train.py --model cnn --cnn-epochs 20
```

---

## Model Architectures

**ANN (Baseline)**
- Flatten → Dense(3000, ReLU) → Dense(1000, ReLU) → Dense(10, Softmax)
- Optimizer: SGD

**CNN**
- Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(64) → Dense(10, Softmax)
- Optimizer: Adam

---

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib
- Scikit-learn (evaluation metrics)

---

## Author

**Tilal Ahmed**  
BS Computer Science — Iqra University, Karachi  
[LinkedIn](https://www.linkedin.com/in/YOUR_LINKEDIN) · tilalahmed956@gmail.com
