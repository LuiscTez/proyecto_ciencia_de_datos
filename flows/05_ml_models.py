import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import datetime
import sklearn
import sys
import hashlib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from flows.common.logging import get_logger
from flows.common.paths import SILVER, SILVER_NAME

log = get_logger(__name__)

TARGET = "Survived"
NUM = ["Age", "Fare", "SibSp", "Parch", "Familia"]
CAT = ["Pclass", "Sex", "Embarked", "Alone", "TCabin"]

# --- Decisiones congeladas (resultado del cuaderno 04; justificadas en reports/problema_ml.md) ---
MODEL_NAME = "titanic_model_v1"          # Nombre del modelo
FINAL_ESTIMATOR = "logreg"               # <--- CAMBIA ESTO por el modelo que te ganó (logreg, forest, hgb, etc.)

# Si tu modelo ganador fue la Regresión Logística, tus params serían algo como:
FINAL_PARAMS = {
    "model__C": 1.0,
}

# Si tu modelo ganador fue un Random Forest, sería algo como:
# FINAL_PARAMS = {
#     "model__n_estimators": 200,
#     "model__max_depth": 6,
#     "model__min_samples_leaf": 1
# }

USE_CABIN_FEATURES = False               # decisión del Paso 61 (excluir HasCabin)
THRESHOLD = 0.5                          # decisión del Paso 53 (umbral de decisión)

def load_dataset(in_dir: Path = SILVER) -> tuple[pd.DataFrame, pd.Series]:
    """Lee Silver y devuelve X e y."""
    df = pd.read_parquet(Path(in_dir) / SILVER_NAME)
    df["Age"] = df["AgeOriginal"]
    
    columnas_a_eliminar = ["Name", "Ticket", "Cabin", "AgeOriginal", "AgeImputada"]
    df = df.drop(columns=columnas_a_eliminar, errors="ignore")
    
    X = df[NUM + CAT].copy()
    y = df[TARGET].copy()
    return X, y

def split(X: pd.DataFrame, y: pd.Series, test_size: float | None = None,
          random_state: int | None = None) -> tuple:
    """Devuelve X_train, X_test, y_train, y_test, estratificado por y."""
    ts = test_size if test_size is not None else 0.2
    rs = random_state if random_state is not None else 42
    return train_test_split(X, y, test_size=ts, random_state=rs, stratify=y)

class GroupAgeImputer(BaseEstimator, TransformerMixin):
    """Rellena 'Age' con el promedio de su grupo, aprendido SOLO en fit()."""
    def __init__(self, group_cols=("Sex",), target="Age"):
        self.group_cols = group_cols
        self.target = target

    def fit(self, X, y=None):
        cols = list(self.group_cols)
        self.group_means_ = X.groupby(cols)[self.target].mean()
        self.global_mean_ = X[self.target].mean()
        return self

    def transform(self, X):
        X = X.copy()
        cols = list(self.group_cols)
        means = self.group_means_.rename("_fill").reset_index()
        fill = X[cols].merge(means, on=cols, how="left")["_fill"].to_numpy(dtype=float)
        fill = np.where(np.isnan(fill), self.global_mean_, fill)
        X[self.target] = X[self.target].fillna(pd.Series(fill, index=X.index))
        return X

def build_pipeline(model, num_features=None, cat_features=None) -> Pipeline:
    """GroupAgeImputer -> ColumnTransformer -> modelo."""
    num_features = NUM if num_features is None else num_features
    cat_features = CAT if cat_features is None else cat_features

    num = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10))
    ])

    prep = ColumnTransformer(
        transformers=[
            ('num', num, num_features),
            ('cat', cat, cat_features)
        ],
        sparse_threshold=0
    )

    return Pipeline([
        ('age_imputer', GroupAgeImputer(group_cols=('Sex', 'Pclass'))), 
        ('prep', prep),
        ('model', model)
    ])

def get_candidates(random_state: int = 42) -> dict:
    """Devuelve un diccionario con los modelos candidatos para evaluar."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000, random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "dt": DecisionTreeClassifier(random_state=random_state),
        "gb": GradientBoostingClassifier(random_state=random_state)
    }

def get_candidates(random_state: int = 42) -> dict:
    """Diccionario: nombre -> estimador sin entrenar."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000, random_state=random_state),
        "tree": DecisionTreeClassifier(random_state=random_state),
        "forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(random_state=random_state),
    }

def cross_validate_candidates(X, y, num_features=None, cat_features=None, 
                              models=None, cv=5, random_state=42) -> pd.DataFrame:
    """Devuelve un DataFrame con media y desviación estándar del ROC-AUC por modelo."""
    models = models or get_candidates(random_state)
    
    # 1. Configuramos la validación cruzada estratificada
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    results = []
    
    # 2. Evaluamos cada modelo
    for name, model in models.items():
        # Construimos el pipeline con el modelo actual
        pipe = build_pipeline(model, num_features, cat_features)
        
        # 3. Cross validation (scoring='roc_auc')
        scores = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        
        # 4. Guardamos los resultados
        results.append({
            "model": name,
            "mean": scores.mean(),
            "std": scores.std(),
            "folds": scores.tolist()
        })
    
    # 5. Retornamos el DataFrame ordenado (del mejor al peor)
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="mean", ascending=False).reset_index(drop=True)
    
    log.info("Resultados CV:\n%s", df_results[["model", "mean", "std"]].to_string())
    return df_results

def get_param_grids() -> dict[str, dict]:
    """Rejillas por modelo, con los nombres que espera el Pipeline (prefijo 'model__')."""
    return {
        'logreg': {
            'model__C': [0.01, 0.1, 1, 10], 
            'model__penalty': ['l2']
        },
        'forest': {
            'model__n_estimators': [200, 400], 
            'model__max_depth': [4, 6, 8, None],
            'model__min_samples_leaf': [1, 3, 5]
        },
        'hgb': {
            'model__learning_rate': [0.03, 0.1], 
            'model__max_depth': [2, 3, 4],
            'model__max_iter': [100, 200], 
            'model__l2_regularization': [0.0, 1.0]
        }
    }



def evaluate_on_test(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """Devuelve exactitud, F1, ROC-AUC y el tamaño de la prueba (dict serializable a JSON)."""
    # 1. Obtenemos las probabilidades (columna 1 = probabilidad de sobrevivir)
    proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 2. Aplicamos el umbral elegido
    pred = (proba >= threshold).astype(int)
    
    # 3. Calculamos las métricas
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_test": int(len(y_test))
    }

def evaluate_on_test(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """Devuelve exactitud, F1, ROC-AUC y el tamaño de la prueba."""
    # 1. Predecimos probabilidades (columna 1 = probabilidad de sobrevivir)
    proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 2. Aplicamos el umbral (por defecto 0.5)
    pred = (proba >= threshold).astype(int)
    
    # 3. Retornamos el diccionario
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_test": int(len(y_test))
    }

def train_final(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Ajusta el pipeline con las decisiones congeladas y lo entrena con TODO (X, y)."""
    # 1. Definimos las features. Si usamos cabina, la agregamos a las categóricas.
    cat_features = CAT.copy()
    if USE_CABIN_FEATURES and "HasCabin" not in cat_features:
        cat_features.append("HasCabin")
    
    num_features = NUM.copy()
    
    # 2. Obtenemos el estimador base (ej. logreg o forest)
    modelo = get_candidates(42)[FINAL_ESTIMATOR]
    
    # 3. Construimos el pipeline
    pipe = build_pipeline(modelo, num_features=num_features, cat_features=cat_features)
    
    # 4. Aplicamos los hiperparámetros congelados (los del Paso 64)
    pipe.set_params(**FINAL_PARAMS)
    
    # 5. Entrenamos con el 100% de los datos
    pipe.fit(X, y)
    log.info("Modelo final entrenado con %d filas.", len(X))
    
    return pipe

def save_model(pipeline: Pipeline, metadata: dict, name: str = MODEL_NAME, 
               model_dir: Path = Path("models")) -> Path:
    """Guarda el pipeline (.joblib) y un archivo de metadatos (.metadata.json)."""
    # Crear carpeta si no existe
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{name}.joblib"
    
    # 1. Guardar el modelo
    joblib.dump(pipeline, model_path)
    
    # 2. Añadir metadatos automáticos
    metadata.update({
        "name": name,
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    })
    
    # 3. Guardar metadatos en JSON
    (model_dir / f"{name}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), 
        encoding="utf-8"
    )
    
    log.info("Modelo guardado en %s", model_path)
    return model_path

def load_model(model_dir: Path = Path("models"), name: str = MODEL_NAME) -> tuple[Pipeline, dict]:
    """Carga el pipeline y sus metadatos; avisa si la versión de scikit-learn cambió."""
    model_path = model_dir / f"{name}.joblib"
    meta_path = model_dir / f"{name}_metadata.json"
    
    pipeline = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    
    if meta.get("sklearn_version") != sklearn.__version__:
        log.warning(
            "¡Atención! Modelo entrenado con sklearn %s; versión actual %s.",
            meta.get("sklearn_version"), sklearn.__version__
        )
    return pipeline, meta