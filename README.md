
# DeepLearning Techniques Exercises in Python

This repository contains four foundational exercises on neural networks and logic functions, implemented in Python.

## 📂 Files

###  EX1.py – Logic Functions using McCulloch-Pitts Neuron
Implements the following logic gates using the McCulloch-Pitts neuron model:
- AND
- OR
- NOT
- NOR
- XOR

###  EX2.py – Logistic Regression Classifier
Builds a logistic regression classifier using a neural network approach:
- Uses the Breast Cancer dataset
- Preprocessing with NumPy
- Model building from scratch
- Evaluation and performance analysis

###  EX3.py – Neural Network with One Hidden Layer
Implements a 2-class classification problem with:
- One hidden layer using `tanh` activation
- Cross-entropy loss
- Forward and backward propagation
- Performance evaluation and result analysis

###  EX4.py – CNN on MNIST Dataset
Trains a Convolutional Neural Network on the MNIST handwritten digits dataset using TensorFlow:
- Layers: `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`
- Activation functions: `relu`, `softmax`
- Trained for 5 epochs with validation accuracy

## 💡 Notes
- All scripts are written in Python and designed for educational purposes.
- Virtual environments are recommended to manage dependencies.
- Additional enhancements and datasets can be added later.

---

## 📌 How to Run

Make sure Python and required libraries (e.g., `numpy`, `tensorflow`, `scikit-learn`) are installed. Then run:

```bash
python EX1.py
python EX2.py
python EX3.py
python EX4.py
