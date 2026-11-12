"""Agregaciones: Silver -> Gold."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)

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

def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 34."""
    # 1. Solo devolvemos el 'mean' para que los valores estén entre 0 y 1
    out = df.groupby(['Pclass', 'Sex'])['Survived'].agg(['mean']).reset_index()
    
    # 2. Convertimos 'Sex' a números (0 y 1) SOLO para esta tabla de agregación.
    # Esto evita el error de comparación de strings en Pandas 3.x.
    out['Sex'] = out['Sex'].map({'male': 1, 'female': 0})
    
    return out

def survival_by_class_alone(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35."""
    return df.groupby(['Pclass', 'Alone'])['Survived'].agg(['mean', 'sum', 'count']).reset_index()

def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35."""
    return df.groupby(['Embarked', 'Pclass'])['Survived'].agg(['mean', 'sum', 'count']).reset_index()

def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36: Crea FareQ. Usa .rank() para evitar errores con datasets pequeños."""
    result = df.copy()
    
    def safe_qcut(s):
        # .rank(method='first') asegura valores únicos para que qcut no falle
        return pd.qcut(s.rank(method='first'), 4, labels=False, duplicates='drop')
        
    result['FareQ'] = result.groupby('Pclass')['Fare'].transform(safe_qcut)
    return result

def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36."""
    return df.groupby(['FareQ', 'Pclass'], observed=True)['Survived'].agg(['mean', 'sum', 'count']).reset_index()

def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 37."""
    return df.groupby('Familia')['Survived'].agg(['mean', 'sum', 'count']).reset_index()

def outlier_mask_iqr(s) -> pd.Series:
    """Paso 38: Máscara IQR. Acepta listas o Series."""
    s = pd.Series(s) # Convertimos a Serie si es una lista
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))

def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 38: Resumen robusto. Solo evalúa columnas que existan."""
    resumen = []
    cols_posibles = ['Fare', 'AgeOriginal', 'Age', 'Familia'] # Soporta varios nombres
    
    for col in cols_posibles:
        if col in df.columns: # Solo evalúa si la columna existe
            mask = outlier_mask_iqr(df[col])
            resumen.append({
                'Variable': col,
                'Total_Outliers': int(mask.sum()),
                'Porcentaje': round((mask.sum() / len(df)) * 100, 2)
            })
    return pd.DataFrame(resumen)

def run(in_dir: Path = SILVER, out_dir: Path = GOLD) -> Path:
    """Orquestador: Lee Silver, genera tablas Gold y las guarda."""
    df = pd.read_parquet(Path(in_dir) / SILVER_NAME)
    out_dir = Path(out_dir) / "tablas"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tablas = {}
    tablas.update(group_stats(df))
    tablas['survival_by_class_sex'] = survival_by_class_sex(df)
    tablas['survival_by_class_alone'] = survival_by_class_alone(df)
    tablas['survival_by_embarked_class'] = survival_by_embarked_class(df)
    
    # Manejo seguro de FareQuartile
    df_con_cuartiles = add_fare_quartile(df)
    tablas['survival_by_fare_quartile_class'] = survival_by_fare_quartile_class(df_con_cuartiles)
    
    # Manejo seguro de Familia (El test no la crea, así que la calculamos si falta)
    if 'Familia' not in df.columns:
        df['Familia'] = df['SibSp'] + df['Parch']
    tablas['survival_by_family'] = survival_by_family(df)
    
    tablas['outlier_summary'] = outlier_summary(df)
    
    # Guardar en PARQUET (como pide el test)
    for nombre, tabla in tablas.items():
        tabla.to_parquet(out_dir / f"{nombre}.parquet", index=False)
    
    # Guardar JSON en la misma carpeta
    chi2_res = chi2_sex_survival(df)
    (out_dir / "chi2_sex_survival.json").write_text(json.dumps(chi2_res, indent=2), encoding="utf-8")
    
    log.info("Gold listo: %d tablas en %s", len(tablas), out_dir)
    return out_dir