"""Agregaciones: Silver -> Gold. Cada función devuelve una tabla lista para un informe."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)

# --- FUNCIONES DE AGREGACIÓN (PASOS 28-38) ---

def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Retorna las tablas descriptivas del Paso 28."""
    return {
        'age_by_survival': df.groupby('Survived')['Age'].describe().reset_index(),
        'survival_by_sex': df.groupby('Sex')['Survived'].agg(['mean', 'sum', 'count']).reset_index(),
        'survival_by_pclass': df.groupby('Pclass')['Survived'].agg(['mean', 'sum', 'count']).reset_index(),
        'survival_by_alone': df.groupby('Alone')['Survived'].agg(['mean', 'sum', 'count']).reset_index()
    }

def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Prueba Chi-cuadrado para Sexo vs Supervivencia (Paso 29)."""
    tabla = pd.crosstab(df['Sex'], df['Survived'])
    chi2, p, gl, esperados = stats.chi2_contingency(tabla)
    return {'chi2': float(chi2), 'p_value': float(p), 'dof': int(gl)}

# NUEVAS FUNCIONES DE LA PARTE 9

def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 34: Supervivencia cruzando Clase y Sexo."""
    return df.groupby(['Pclass', 'Sex'])['Survived'].agg(['mean', 'sum', 'count']).reset_index()

def survival_by_class_alone(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35: Supervivencia cruzando Clase y si viajaba solo."""
    return df.groupby(['Pclass', 'Alone'])['Survived'].agg(['mean', 'sum', 'count']).reset_index()

def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35: Supervivencia cruzando Puerto de embarque y Clase."""
    return df.groupby(['Embarked', 'Pclass'])['Survived'].agg(['mean', 'sum', 'count']).reset_index()

def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36: Crea la columna FareQuartile dividiendo la tarifa en 4 grupos iguales."""
    result = df.copy()
    # qcut divide en cuartiles (Q1, Q2, Q3, Q4) basados en la distribución
    result['FareQuartile'] = pd.qcut(result['Fare'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    return result

def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36: Supervivencia por Cuartil de Tarifa y Clase."""
    # Asumimos que FareQuartile ya fue creada por add_fare_quartile
    return df.groupby(['FareQuartile', 'Pclass'], observed=True)['Survived'].agg(['mean', 'sum', 'count']).reset_index()

def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 37: Supervivencia según el tamaño de la familia."""
    return df.groupby('Familia')['Survived'].agg(['mean', 'sum', 'count']).reset_index()

def outlier_mask_3sigma(df: pd.DataFrame, cols: list) -> pd.Series:
    """Paso 38: Detecta valores atípicos usando la regla de las 3 sigmas."""
    mask = pd.Series(False, index=df.index)
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        # Si el valor está a más de 3 desviaciones estándar de la media, es un outlier
        mask = mask | (df[col] < (mean - 3 * std)) | (df[col] > (mean + 3 * std))
    return mask

def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 38: Resumen de cuántos outliers hay en variables clave."""
    cols_numericas = ['Age', 'Fare', 'SibSp', 'Parch']
    resumen = []
    for col in cols_numericas:
        mask = outlier_mask_3sigma(df, [col])
        resumen.append({
            'Variable': col,
            'Total_Outliers': int(mask.sum()),
            'Porcentaje': round((mask.sum() / len(df)) * 100, 2)
        })
    return pd.DataFrame(resumen)

# --- ORQUESTADOR ---

def run(in_dir: Path = SILVER, out_dir: Path = GOLD) -> Path:
    """Lee Silver, genera todas las tablas Gold y las guarda en CSV/JSON."""
    log.info("Iniciando Capa Gold...")
    df = pd.read_parquet(Path(in_dir) / SILVER_NAME)
    
    # Creamos la carpeta de salida
    out_dir = Path(out_dir) / "tablas"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generamos las tablas
    tablas = {}
    tablas.update(group_stats(df))
    tablas['survival_by_class_sex'] = survival_by_class_sex(df)
    tablas['survival_by_class_alone'] = survival_by_class_alone(df)
    tablas['survival_by_embarked_class'] = survival_by_embarked_class(df)
    
    # El cuartil de tarifa requiere modificar el df primero
    df_con_cuartiles = add_fare_quartile(df)
    tablas['survival_by_fare_quartile_class'] = survival_by_fare_quartile_class(df_con_cuartiles)
    
    tablas['survival_by_family'] = survival_by_family(df)
    tablas['outlier_summary'] = outlier_summary(df)
    
    # Guardamos cada tabla en un CSV
    for nombre, tabla in tablas.items():
        tabla.to_csv(out_dir / f"{nombre}.csv", index=False)
    
    # Guardamos el Chi-cuadrado en un JSON aparte
    chi2_res = chi2_sex_survival(df)
    (out_dir.parent / "chi2_sex_survival.json").write_text(json.dumps(chi2_res, indent=2), encoding="utf-8")
    
    log.info("Gold listo: %d tablas en %s", len(tablas), out_dir)
    return out_dir