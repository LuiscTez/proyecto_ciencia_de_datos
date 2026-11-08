"""Ingeniería de datos: Bronze -> Silver.

Cada transformación es una función pura: recibe un DataFrame y devuelve uno
nuevo. El orden importa y está fijado en build_silver().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE, BRONZE_LABELED_NAME, SILVER, SILVER_FINAL_NAME, SILVER_NAME,
)

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]
DECKS = list("ABCDEFGTX")            # orden FIJO: X = desconocida (reproducible)
TITLE_MAP = {                        # consolidación de títulos (Paso 26)
    "Master": "Mr", "the Countess": "Mrs", "Dona": "Mrs", "Mme": "Mrs",
    "Lady": "Miss", "Mlle": "Miss", "Ms": "Miss",
}
MAIN_TITLES = ["Mr", "Mrs", "Miss"]


def validate_bronze(df: pd.DataFrame) -> pd.DataFrame: ...      # contrato de entrada
def cast_types(df: pd.DataFrame) -> pd.DataFrame: ...           # Survived -> int, etc.
def add_person(df: pd.DataFrame) -> pd.DataFrame: ...           # Paso 8
def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame: ...   # Pasos 11 y 24 (TCabin, HasCabin)
def add_family_features(df: pd.DataFrame) -> pd.DataFrame: ...  # Paso 14 (Familia, Alone)
def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame: ...   # Paso 16 (Survivor)
def add_title(df: pd.DataFrame) -> pd.DataFrame: ...            # Paso 25 (TTitulo)
def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame: ...   # Paso 26
def impute_embarked(df: pd.DataFrame) -> pd.DataFrame: ...      # Paso 21
def impute_age(df: pd.DataFrame) -> pd.DataFrame: ...           # Paso 22
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame: ...  # Pasos 21, 23, 24, 26
def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame: ...  # Paso 32
def validate_silver(df: pd.DataFrame) -> pd.DataFrame: ...      # Paso 33


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Orden deliberado: Person usa la edad ORIGINAL, así que va antes de imputar."""
    return (
        bronze
        .pipe(validate_bronze)
        .pipe(cast_types)
        .pipe(add_person)
        .pipe(add_cabin_features)
        .pipe(add_family_features)
        .pipe(add_survivor_label)
        .pipe(add_title)
        .pipe(consolidate_titles)
        .pipe(impute_embarked)
        .pipe(impute_age)
        .pipe(encode_categoricals)
    )


def run(in_dir: Path = BRONZE, out_dir: Path = SILVER) -> Path:
    bronze = pd.read_csv(Path(in_dir) / BRONZE_LABELED_NAME)
    silver = validate_silver(build_silver(bronze))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_dir / SILVER_NAME, index=False)
    drop_original_columns(silver).to_parquet(out_dir / SILVER_FINAL_NAME, index=False)
    log.info("Silver listo: %d filas, %d columnas", *silver.shape)
    return out_dir / SILVER_NAME

def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Person': 'child' si Age < 16; si no, el valor de Sex.

    Regla para edad desconocida (NaN): se usa Sex. Esto subestima a los niños
    sin edad registrada; el sesgo se cuantifica en el Reto 8.1.
    """

    result = df.copy()

    result["Person"] = np.where(
        result["Age"] < 16,
        "child",
        result["Sex"]
    )

    return result

def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'TCabin' (primera letra de Cabin, 'X' si es nula) y 'HasCabin' (0/1)."""
    # TODO: df['Cabin'].str[0].fillna('X'); df['Cabin'].notna().astype(int)
    raise NotImplementedError

def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Familia' = SibSp + Parch y 'Alone' ('with Family' si Familia > 0, 'Alone' si no)."""
    # TODO
    raise NotImplementedError

def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Survivor' mapeando 0 -> 'No' y 1 -> 'Yes' (solo para etiquetar gráficos)."""
    
    result = df.copy()
    
    # Mapeamos los números a texto
    result['Survivor'] = result['Survived'].map({0: 'No', 1: 'Yes'})
    
    return result

def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena los nulos de 'Embarked' con la moda (el puerto más frecuente)."""
    result = df.copy()
    
    # Calculamos la moda (el valor más repetido). 
    # .mode()[0] toma el primer valor en caso de que haya empates.
    moda_embarked = result['Embarked'].mode()[0]
    
    # Rellenamos los nulos (asignando a la columna, sin usar inplace=True)
    result['Embarked'] = result['Embarked'].fillna(moda_embarked)
    
    return result

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte variables de texto a números para los modelos."""
    result = df.copy()
    
    # Codificación de Embarked (S -> 0, C -> 1, Q -> 2)
    result['NEmbarked'] = result['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # (Aquí irán las demás codificaciones de los siguientes pasos)
    return result

def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa 'Age' con el promedio por 'Sex'.

    Conserva la edad original en 'AgeOriginal' (con NaN) y marca las filas
    imputadas en 'AgeImputada' (bool). Usada para EXPLORAR; el modelo (Parte 11)
    vuelve a imputar solo con datos de entrenamiento.
    """
    # 1. df['AgeOriginal'] = df['Age']; df['AgeImputada'] = df['Age'].isna()
    # 2. promedios = df.groupby('Sex')['Age'].mean()
    # 3. df['Age'] = df['Age'].fillna(df['Sex'].map(promedios))
    raise NotImplementedError

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte variables de texto a números para los modelos."""
    result = df.copy()
    
    # Paso 21: Codificación de Embarked (S -> 0, C -> 1, Q -> 2)
    result['NEmbarked'] = result['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Paso 23: Codificación de Sex (female -> 0, male -> 1)
    result['NSex'] = result['Sex'].map({'female': 0, 'male': 1})
    
    # Paso 24: Codificación de Cabin (usando el diccionario FIJO)
    # DECKS ya está definida al principio de tu archivo como: list("ABCDEFGTX")
    mapa_cabin = {letra: i for i, letra in enumerate(DECKS)}
    result['NCabin'] = result['TCabin'].map(mapa_cabin)
    
    return result

def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el título del nombre (ej. 'Mr', 'Mrs', 'Miss') usando regex."""
    
    result = df.copy()
    
    # Usamos str.extract con la expresión regular
    # El patrón busca una coma, espacios opcionales, captura el texto hasta el punto.
    result['TTitulo'] = result['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
    
    return result

def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Consolida los títulos raros en categorías principales o 'Otro'."""
    result = df.copy()
    
    # 1. Reemplazamos usando el diccionario (ej. 'the Countess' -> 'Mrs')
    result['TTitulo'] = result['TTitulo'].replace(TITLE_MAP)
    
    # 2. Todo lo que NO esté en MAIN_TITLES (Mr, Mrs, Miss) se convierte en 'Otro'
    result['TTitulo'] = result['TTitulo'].where(result['TTitulo'].isin(MAIN_TITLES), 'Otro')
    
    return result

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # ... (código anterior de Embarked, Sex y Cabin) ...
    
    # Paso 26: Codificación de TTitulo (Mr -> 0, Mrs -> 1, Miss -> 2, Otro -> 3)
    result['NTitulo'] = result['TTitulo'].map({'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Otro': 3})
    
    return result

def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una COPIA sin Name, Sex, Embarked y Cabin."""
    
    # Hacemos una copia para no modificar el DataFrame original
    result = df.copy()
    
    # Definimos las columnas de texto que ya fueron codificadas
    columnas_a_eliminar = ['Name', 'Sex', 'Embarked', 'Cabin']
    
    # Eliminamos las columnas
    result = result.drop(columns=columnas_a_eliminar)
    
    return result

def validate_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """Lanza ValueError si faltan columnas o hay IDs duplicados."""
    # Verificamos que estén todas las columnas esperadas
    faltantes = set(EXPECTED_COLUMNS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en Bronze: {faltantes}")
    
    # Verificamos que no haya IDs duplicados
    if df['PassengerId'].duplicated().any():
        raise ValueError("Hay PassengerId duplicados")
        
    return df

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte las columnas a sus tipos numéricos correctos."""
    result = df.copy()
    
    # Convertimos a entero
    for col in ['Survived', 'Pclass', 'SibSp', 'Parch']:
        result[col] = result[col].astype('Int64') # Usamos Int64 (mayúscula) para permitir nulos si los hay
        
    # Convertimos a flotante (decimal)
    for col in ['Age', 'Fare']:
        result[col] = result[col].astype(float)
        
    return result

def validate_silver(df: pd.DataFrame) -> pd.DataFrame:
    """Lanza ValueError si hay nulos en las columnas obligatorias."""
    columnas_sin_nulos = [
        'Survived', 'Pclass', 'Age', 'Fare', 'Embarked', 'Person', 
        'Alone', 'TCabin', 'TTitulo', 'NSex', 'NEmbarked', 'NCabin', 'NTitulo'
    ]
    
    # Verificamos si alguna de esas columnas tiene nulos
    if df[columnas_sin_nulos].isnull().any().any():
        raise ValueError("Hay valores nulos en columnas que deberían estar llenas")
        
    return df