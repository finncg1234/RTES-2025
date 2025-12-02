from dataclasses import dataclass
from enum import Enum

url_no_label = "https://bitbucket.org/brooke-lampe/can-dataset/src/master/"
url_label    = "https://bitbucket.org/brooke-lampe/can-ml/raw/7268a6f564b7a1d4ad2bec089b60abaf68e18bdf/"

class feature_extraction(Enum):
    NO_CAN_DATA = 1
    NAIVE = 2
    STATISTICAL = 3

@dataclass
class Config:
    N: int = 10
    B: int = 256
    input_size: int = 16
    epochs: int = 1500
    lr: float = 1e-4
    wd: float = 0

    bfw: bool = True

    fe: feature_extraction = feature_extraction.NAIVE

    vehicle: str = "2016-chevrolet-silverado"
    dtype: str = "extra-attack-free"
    labeled: bool = False
    model_path: str = ""
    file_path: str = ""