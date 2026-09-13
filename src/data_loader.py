import os
import pandas as pd
import kagglehub


def download_dataset():
    """
    Download the Formula 1 World Championship dataset
    using KaggleHub.
    """
    path = kagglehub.dataset_download(
        "rohanrao/formula-1-world-championship-1950-2020"
    )
    return path


def load_driver_standings(path):
    """
    Load the driver standings CSV file.
    """
    csv_path = os.path.join(path, "driver_standings.csv")
    data = pd.read_csv(csv_path)

    return data
