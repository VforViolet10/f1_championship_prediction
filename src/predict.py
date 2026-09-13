import numpy as np


def predict(model, x):
    """
    Generate binary predictions using the trained model.
    """

    probabilities = model.predict(
        x,
        verbose=0
    )

    predictions = (
        probabilities > 0.5
    ).astype(int).flatten()

    return predictions


def display_predictions(predictions, actual):
    """
    Display predicted and actual values.
    """

    print("Prediction:    ", predictions[:30])
    print("Actual values: ", actual[:30])
