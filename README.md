# MNIST CNN with TensorFlow + TensorBoard

This project implements a convolutional neural network (CNN) trained on the MNIST dataset using TensorFlow and visualized with TensorBoard. It includes a classic evaluation and a novel interpretation of the confusion matrix as a geometric-statistical object.

**The model and methodology are directly inspired by the pedagogical framework of FIDLE (Formation à l’Intelligence Artificielle pour les Données de Laboratoire et d’Expériences) developed by CNRS.**

---

## 🗂️ Methodology

### 📥 Dataset

The MNIST dataset is loaded directly from Hugging Face (`ylecun/mnist`). This dataset was historically curated by **Yann LeCun**, one of the inventors of convolutional neural networks. Originally used for recognizing handwritten digits on U.S. bank checks, it formed the basis for LeNet-5 — one of the first CNN architectures, implemented in **C and Fortran** in the late 1990s.

### ⚙️ Configuration

- CPU: AMD Ryzen 5  
- RAM: 32 GB  
- OS: Windows 11  
- No GPU acceleration (CPU-only training, 5 epochs under 20 seconds)  
- JupyterLab + TensorBoard

### 🧠 Tools

- TensorFlow 2.15  
- Matplotlib (visualizations)  
- scikit-learn (confusion matrix)  
- TensorBoard (histograms, learning curves, training profile)  

---

## 🔍 Original contribution

### 📌 Beyond the confusion matrix

In addition to classic accuracy evaluation (`model.evaluate()`), this project proposes an **original geometric and statistical analysis** of the confusion matrix:

- The **trace** is interpreted as the axis of perfect predictions (i = j)
- The matrix is seen as a **weighted cloud of prediction points**
- We compute:
  - **Mean Absolute Error (MAE)** → average deviation from the diagonal
  - **Mean Squared Error (MSE)** → emphasis on distant/confused predictions
  - **Off-diagonal mass** → total error rate as dispersion
- This view is especially useful when classes are ordered or continuous (e.g., facial similarity, stages of development, ecological gradients)

About the histogram of biases (Tensorflow):  
- At epoch 0, the bias distribution is like **Mont Blanc**: sharp and centered  
- As training proceeds, it evolves into **Puy de Dôme**: wider and flatter, reflecting learned diversity and specialization

---

## 📊 Results

- Final accuracy (test set): **98.36 %**
- Training time (CPU, 5 epochs): **< 20 seconds**
- TensorBoard used to inspect:
  - Learning curves
  - Bias distributions
  - Weight evolution

---

## 📁 Files included

- `Training_MNIST.html` – Exported notebook as HTML  
- `Training_MNIST.md` – Clean Markdown version of the Python code  
- `confusion_matrix.jpg` – Static export of the confusion matrix  
- `histogram_biases.jpg` – Screenshot from TensorBoard  
- `Accuracy_and_loss` – Footage from the Learning curves 
- `Numbers_to_identify_example` – Footage from dataset itself
- `README.md` – This file

---

## ✍️ Author

**Jerôme-X1**  
This repository is part of a broader AI-education and cognitive analysis effort in preparation for the AGI horizon (**mid to late 2026**).
