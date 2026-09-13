from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(data):
    """
    Prepare input and output variables,
    perform train-test split and feature scaling.
    """

    # Select input and output variables
    x = data.iloc[:, 0:6]
    y = data.iloc[:, 6]

    # Keep numerical features only
    x = x.select_dtypes(
        include=["int64", "float64"]
    )

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, scaler
