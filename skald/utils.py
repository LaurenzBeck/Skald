"""🛠️ utility functions for `skald`."""

import pandas as pd


def flatten_dict(d: dict, sep: str = ".") -> dict:
    """📖 flattens a nested dictionary by concatenating keys with <sep>.

    Code from https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/

    Args:
        d (dict): dictionary
        sep (str, optional): keys are concatenated using this separator. Defaults to '.'

    Returns:
        dict: flattened dictionary
    """
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    return flat_dict
