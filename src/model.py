from keras.models import Sequential
from keras.layers import Dense


def build_model(input_features):
    """
    Build and compile the neural network.
    """

    model = Sequential()

    model.add(
        Dense(
            16,
            activation="relu",
            input_shape=(input_features,)
        )
    )

    model.add(
        Dense(
            8,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model
