# 🧠 Naive Bayes Interactive Learning System

An interactive, step-by-step web application built using **Streamlit** to understand and apply the **Naive Bayes algorithm** for classification problems.

This system is designed especially for students and beginners to learn machine learning concepts visually and practically.

---

## 🚀 Features

### 📂 1. Dataset Input

* Upload your own CSV dataset (max 5 MB)
* Built-in demo datasets:

  * Iris Dataset
  * Titanic-lite Dataset
* Dataset preview, shape, and missing values detection

---

### 🔧 2. Preprocessing

* Handle missing values:

  * Mean / Mode / Constant / Drop rows
* Encode categorical data:

  * Label Encoding (single click or selective)
  * One-Hot Encoding
* Feature scaling:

  * StandardScaler (Z-score)
  * MinMaxScaler (0–1)

---

### 📊 3. Exploratory Data Analysis (EDA)

* Class distribution (bar + pie chart)
* Feature distribution (histograms per class)
* Correlation heatmap (auto-encoded)

---

### 📐 4. Naive Bayes Theory

* Bayes’ Theorem explanation
* Prior, Likelihood, Posterior
* Gaussian & Multinomial formulas
* Step-by-step toy example (Play Tennis dataset)

---

### ⚙️ 5. Training Configuration

* Train/Test split (custom ratio)
* Model selection:

  * Gaussian Naive Bayes
  * Multinomial Naive Bayes
* k-Fold Cross Validation

---

### 🚀 6. Train & Visualize

* Model training with one click
* View learned parameters:

  * Mean & Variance (Gaussian)
  * Feature probabilities (Multinomial)
* Gaussian distribution visualization

---

### 🔮 7. Prediction

* Manual input for new sample
* Step-by-step probability calculation
* Posterior probability visualization

---

### 📈 8. Evaluation

* Accuracy, Precision, Recall, F1-score
* Confusion Matrix
* Full Classification Report
* Cross-validation results

---

### 🗑️ Extra Feature

* **Clear All / Reset Button**

  * Resets dataset, model, and all states
  * Restarts the app instantly

---

## 🛠️ Technologies Used

* Python 🐍
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📦 Installation

```bash
git clone https://github.com/your-username/naive-bayes-learning-system.git
cd naive-bayes-learning-system
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                # Main Streamlit application
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
```

---

## 🎯 Learning Objectives

This project helps you understand:

* How Naive Bayes works mathematically
* Data preprocessing techniques
* Feature encoding & scaling
* Model training & evaluation
* Probability-based prediction

---

## ⚠️ Limitations

* Assumes feature independence (Naive assumption)
* Multinomial NB requires non-negative inputs
* Performance depends on dataset quality

---

## 🤝 Contribution

Feel free to fork this repository and improve it:

* Add new models (SVM, Decision Tree)
* Improve UI/UX
* Add model saving/loading

---

## 📄 License

This project is open-source and available under the MIT License.
## 🚀 Live Demo
[![Open App](https://img.shields.io/badge/Launch-App-blue?style=for-the-badge&logo=streamlit)]
---

## 👨‍💻 Author

Developed for educational purposes to simplify Machine Learning concepts.

---
