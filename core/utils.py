"""
Common utility functions for data I/O and processing.
"""
import os
import zipfile
from typing import List
import numpy as np
import pandas as pd


def print_shape_dtype(arrays: List[np.ndarray], names: List[str]):
    """Print shape and dtype for list of arrays."""
    for arr, name in zip(arrays, names):
        print(f'{name} shape: {arr.shape}, dtype: {arr.dtype}')


def save_compressed(data: np.ndarray, filename: str):
    """Save numpy array as compressed zip."""
    npy_name = filename.replace('.zip', '.npy')
    with zipfile.ZipFile(filename, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        np.save(npy_name, data)
        zf.write(npy_name, arcname=os.path.basename(npy_name))
        os.remove(npy_name)


def load_compressed(filename: str) -> np.ndarray:
    """Load numpy array from compressed zip."""
    with zipfile.ZipFile(filename, 'r') as zf:
        npy_name = os.path.basename(filename.replace('.zip', '.npy'))
        zf.extract(npy_name)
    data = np.load(npy_name)
    os.remove(npy_name)
    return data


def load_parquet_landmarks(path: str, n_rows: int = 543) -> np.ndarray:
    """Load landmark data from parquet file."""
    df = pd.read_parquet(path, columns=['x', 'y', 'z'])
    n_frames = len(df) // n_rows
    return df.values.reshape(n_frames, n_rows, 3).astype(np.float32)


def load_metadata(csv_path: str = 'data/train.csv'):
    """Load and prepare metadata from CSV."""
    train_df = pd.read_csv(csv_path)
    train_df['file_path'] = train_df['path'].apply(lambda x: f'./{x}')
    train_df['sign_ord'] = train_df['sign'].astype('category').cat.codes
    
    # Create translation dictionaries
    sign2ord = train_df[['sign', 'sign_ord']].set_index('sign')['sign_ord'].to_dict()
    ord2sign = {v: k for k, v in sign2ord.items()}
    
    return train_df, sign2ord, ord2sign