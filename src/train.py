"""
Image Classification on CIFAR-10
Author: Tilal Ahmed
Compares a baseline ANN (49% accuracy) vs CNN (68.9% accuracy)
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import classification_report

# ─── Classes ──────────────────────────────────────────────────────────────────

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# ─── 1. Load & Preprocess Data ────────────────────────────────────────────────

def load_data():
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test  = X_test  / 255.0

    # Flatten label arrays
    y_train = y_train.reshape(-1,)
    y_test  = y_test.reshape(-1,)

    print(f"Training samples : {X_train.shape[0]}")
    print(f"Test samples     : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ─── 2. Visualize Samples ─────────────────────────────────────────────────────

def plot_sample(X, y, index):
    plt.figure(figsize=(4, 4))
    plt.imshow(X[index])
    plt.title(f"Label: {CLASSES[y[index]]}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"sample_{index}.png")
    plt.show()


# ─── 3. Baseline ANN ──────────────────────────────────────────────────────────

def build_ann():
    model = models.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10,   activation='softmax'),
    ])
    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ─── 4. CNN Model ─────────────────────────────────────────────────────────────

def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ─── 5. Evaluate ──────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, model_name):
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation")
    print('='*50)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Test Loss     : {loss:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = [np.argmax(e) for e in y_pred]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=CLASSES))
    return y_pred_classes


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CIFAR-10 Image Classifier")
    parser.add_argument("--model",       choices=["ann", "cnn", "both"], default="both",
                        help="Which model to train (default: both)")
    parser.add_argument("--ann-epochs",  type=int, default=5,  help="Epochs for ANN")
    parser.add_argument("--cnn-epochs",  type=int, default=10, help="Epochs for CNN")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data()

    if args.model in ("ann", "both"):
        print("\nTraining ANN (baseline)...")
        ann = build_ann()
        ann.summary()
        ann.fit(X_train, y_train, epochs=args.ann_epochs)
        evaluate(ann, X_test, y_test, "ANN (Baseline)")

    if args.model in ("cnn", "both"):
        print("\nTraining CNN...")
        cnn = build_cnn()
        cnn.summary()
        cnn.fit(X_train, y_train, epochs=args.cnn_epochs)
        evaluate(cnn, X_test, y_test, "CNN")
        cnn.save("models/cnn_cifar10.h5")
        print("\nSaved: models/cnn_cifar10.h5")
