from data_loader import download_dataset, load_driver_standings
from preprocessing import prepare_data
from model import build_model


def main():

    # Download dataset
    path = download_dataset()

    # Load data
    data = load_driver_standings(path)

    print("Dataset shape:", data.shape)
    print(data.head())

    # Prepare data
    x_train, x_test, y_train, y_test, scaler = prepare_data(data)

    # Build model
    model = build_model(x_train.shape[1])

    # Train model
    history = model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate model
    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Predictions
    y_pred = (
        model.predict(x_test, verbose=0) > 0.5
    ).astype(int).flatten()

    y_test_array = y_test.reset_index(
        drop=True
    ).values

    print("\nPrediction:    ", y_pred[:30])
    print("Actual values: ", y_test_array[:30])


if __name__ == "__main__":
    main()
