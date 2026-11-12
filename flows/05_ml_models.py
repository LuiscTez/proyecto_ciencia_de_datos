import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from flows.common.logging import get_logger
from flows.common.paths import SILVER, SILVER_NAME

log = get_logger(__name__)

TARGET = "Survived"

# Columnas candidatas (el "esqueleto" de la guía)
NUM = ["Age", "Fare", "SibSp", "Parch", "Familia"]
CAT = ["Pclass", "Sex", "Embarked", "Alone", "TCabin"]

# Columnas que NO usaremos (El profesor las lista como DROPPED en la guía)
DROPPED = ["Name", "Ticket", "Cabin", "AgeOriginal", "AgeImputada"]

def load_dataset(in_dir=SILVER) -> tuple[pd.DataFrame, pd.Series]:
    """Lee Silver; usa la MISMA (X); el modelo imputa por su cuenta."""
    df = pd.read_parquet(in_dir / SILVER_NAME)
    
    # Eliminamos las columnas que no sirven o causan fuga de datos
    df = df.drop(columns=DROPPED, errors="ignore")
    
    # X: Características (solo las numéricas y categóricas definidas)
    X = df[NUM + CAT].copy()
    
    # y: Variable objetivo
    y = df[TARGET].copy()
    
    return X, y