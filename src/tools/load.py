import os
from pathlib import Path

import numpy as np

ASSETSPATH = Path(__file__).resolve().parents[2] / "assets"

def get_asset_path(*relative_path : str):
    return ASSETSPATH / os.path.join(*relative_path)

def get_asset_m_time(*relative_path : str):
    return os.path.getmtime(get_asset_path(*relative_path))

def load_asset_numpy(*relative_path : str):
    return np.load(get_asset_path(*relative_path))
