# 🏎️ F1 Championship Prediction Using Machine Learning

A machine learning project that explores Formula 1 driver championship outcomes using historical race and driver performance data. The project uses a neural network built with Keras to learn patterns from historical championship statistics and generate predictions.

## 🎯 Project Objective

The goal is to explore how historical Formula 1 performance data can be used to build a predictive machine learning model.

The project covers:

* Data preprocessing and cleaning
* Feature selection
* Feature scaling
* Neural network development
* Model training
* Performance evaluation
* Prediction on unseen data

## 📊 Dataset

The project uses Formula 1 World Championship data spanning **1950–2020**.

The dataset includes information such as:

* Driver performance
* Championship points
* Driver rankings
* Race statistics

Dataset file used:

```text
driver_standings.csv
```

## 🔄 Project Workflow

```text
F1 Historical Data
       ↓
Data Loading
       ↓
Data Cleaning
       ↓
Feature Selection
       ↓
Train / Test Split
       ↓
Feature Scaling
       ↓
Neural Network
       ↓
Model Evaluation
       ↓
Predictions
```

## 🧠 Model Architecture

The project uses a Sequential Neural Network built with Keras:

```text
Input Layer
     ↓
Dense Layer — 16 neurons, ReLU
     ↓
Dense Layer — 8 neurons, ReLU
     ↓
Output Layer — Sigmoid
```

### Training Configuration

| Parameter        | Value               |
| ---------------- | ------------------- |
| Train/Test Split | 80/20               |
| Validation Split | 10%                 |
| Epochs           | 100                 |
| Batch Size       | 16                  |
| Optimizer        | Adam                |
| Loss             | Binary Crossentropy |
| Metric           | Accuracy            |

## 🛠️ Tech Stack

* **Python**
* **Pandas** — Data processing
* **NumPy** — Numerical computation
* **Scikit-learn** — Preprocessing and dataset splitting
* **TensorFlow / Keras** — Neural network
* **KaggleHub** — Dataset access

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/VforViolet10/f1_championship_prediction.git
cd f1_championship_prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn tensorflow keras kagglehub
```

### 3. Run the project

Open the project files inside the `code/` directory and run the notebook/script in your Python or Jupyter environment.

If using KaggleHub, configure your Kaggle credentials as required.

## 📈 Results

The trained model generates predictions on unseen test data and compares the predicted outcomes with the actual target values.

Model performance is evaluated using classification accuracy.

## 🔮 Future Improvements

* Add race-specific and constructor-level features
* Incorporate qualifying performance
* Include weather and circuit characteristics
* Experiment with XGBoost and Random Forest
* Explore LSTM and Transformer architectures for sequential race data
* Add interactive F1 performance visualizations
* Develop a dashboard for championship predictions

## 👩‍💻 Author

**Bushra Farhad**

## 📄 License

This project is licensed under the **MIT License**.
