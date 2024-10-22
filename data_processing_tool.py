import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

def read_csv(file_path: str) -> pd.DataFrame:
    """Reads data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the data by handling missing values and outliers."""
    df = df.dropna()  # Drop missing values
    df = df[(df >= df.quantile(0.01)) & (df <= df.quantile(0.99))]  # Remove outliers
    return df

def transform_data(df: pd.DataFrame, transformation: str) -> pd.DataFrame:
    """Applies a specified transformation to the data."""
    if transformation == 'normalize':
        return (df - df.min()) / (df.max() - df.min())
    elif transformation == 'standardize':
        return (df - df.mean()) / df.std()
    else:
        raise ValueError("Unknown transformation type")

def compute_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """Computes summary statistics for the data."""
    return {
        'mean': df.mean().to_dict(),
        'median': df.median().to_dict(),
        'mode': df.mode().iloc[0].to_dict(),
        'variance': df.var().to_dict(),
        'std_dev': df.std().to_dict()
    }

def plot_data(df: pd.DataFrame, plot_type: str) -> None:
    """Generates a plot of the data."""
    if plot_type == 'histogram':
        df.hist()
    elif plot_type == 'scatter':
        pd.plotting.scatter_matrix(df)
    else:
        raise ValueError("Unknown plot type")
    plt.show()

# Example usage
if __name__ == "__main__":
    df = read_csv('data.csv')
    df = clean_data(df)
    df = transform_data(df, 'normalize')
    stats = compute_statistics(df)
    print(stats)
    plot_data(df, 'histogram')

