# f1_championship_prediction


## Project Overview

This project focuses on predicting Formula 1 driver standings using machine learning techniques. A neural network model is built using **Keras** to classify outcomes based on historical race data.

The dataset is sourced from Kaggle and contains Formula 1 championship data spanning multiple years.



## Features

* Data preprocessing and cleaning
* Feature selection (numerical attributes)
* Train-test split for model evaluation
* Feature scaling using standardization
* Neural network model using Keras
* Model evaluation with accuracy metrics
* Predictions on test data



## Dataset

* Source: Kaggle
* Dataset: *Formula 1 World Championship (1950–2020)*
* File used: `driver_standings.csv`

The dataset includes:

* Driver performance metrics
* Points and rankings
* Race-wise statistics



## Tech Stack

* **Python**
* **Pandas** – Data handling
* **Scikit-learn** – Data splitting & preprocessing
* **Keras / TensorFlow** – Deep learning model
* **KaggleHub** – Dataset access



## Project Workflow

### Data Loading

* Dataset downloaded using `kagglehub`
* Loaded into a Pandas DataFrame

### Data Preprocessing

* Selected relevant numerical features
* Removed non-numeric columns
* Split into:

  * Features (`X`)
  * Target (`y`)

### Train-Test Split

* 80% training data
* 20% testing data

### Feature Scaling

* Standardized using `StandardScaler`

### Model Building

A Sequential Neural Network:

* Input Layer
* Hidden Layer (16 neurons, ReLU)
* Hidden Layer (8 neurons, ReLU)
* Output Layer (Sigmoid activation)

### Model Compilation

* Loss Function: Binary Crossentropy
* Optimizer: Adam
* Metric: Accuracy

### Model Training

* Epochs: 100
* Batch Size: 16
* Validation Split: 10%

### Evaluation

* Model evaluated on test data
* Accuracy printed

### Prediction

* Binary predictions generated
* Compared with actual values



## Results

* The model outputs prediction accuracy on unseen test data
* Predictions are compared against actual driver standings



## Notes

* Ensure Kaggle API or `kagglehub` is properly configured
* Model assumes a binary classification setup
* Further feature engineering can improve performance



## Future Improvements

* Add more relevant features (race conditions, teams, etc.)
* Try advanced models (LSTM, XGBoost, Transformers)
* Hyperparameter tuning
* Visualization dashboards (Tableau / Matplotlib)



## Contributing

Feel free to fork this repo and improve the model or add new features!



## License

This project is open-source and available under the MIT License.
