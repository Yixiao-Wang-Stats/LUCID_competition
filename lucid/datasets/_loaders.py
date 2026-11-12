from importlib.resources import files

import pandas as pd


def _load_dataset(filename, target):
    """Load a bundled dataset as a feature DataFrame and target Series."""
    resource = files("lucid.datasets").joinpath("data", filename)

    with resource.open("rb") as file:
        data = pd.read_csv(file)

    y = data.pop(target)
    return data, y


def auto_mpg():
    """Load the Auto MPG dataset."""
    return _load_dataset("auto_mpg.csv", target="mpg")


def bikesharing():
    """Load the Bike Sharing dataset."""
    return _load_dataset("bike_sharing.csv", target="y")