import numpy as np
import pandas as pd


def log2_transform(counts: pd.DataFrame) -> pd.DataFrame:
    return (counts + 1).apply(np.log2)
