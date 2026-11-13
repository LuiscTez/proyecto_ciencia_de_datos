import importlib
import numpy as np
import pandas as pd
import pytest

ml = importlib.import_module("flows.05_ml_models")

@pytest.fixture
def tiny():
    """Dataset sintético con las columnas candidatas, para probar sin depender de Silver."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "Age": np.where(rng.random(6) < 0.2, np.nan, rng.uniform(1.0, 70.0, 6)),
        "Fare": rng.uniform(5.0, 100.0, 6),
        "SibSp": rng.integers(0, 3, 6),
        "Parch": rng.integers(0, 3, 6),
        "Familia": rng.integers(0, 5, 6),
        "Pclass": rng.integers(1, 4, 6),
        "Sex": rng.choice(["male", "female"], 6),
        "Embarked": rng.choice(["S", "C", "Q", None], 6),
        "Alone": rng.choice(["Alone", "with Family"], 6),
        "HasCabin": rng.integers(0, 2, 6),
        "TCabin": rng.choice(["C", "X", "B"], 6),
    })
    y = pd.Series((X["Sex"] == "female") & (rng.random(6) < 0.9)).astype(int)
    return X, y

def test_group_age_imputer_uses_only_training_statistics(tiny):
    X, _ = tiny
    train = X.iloc[:4].copy()
    
    # 1. FORZAMOS los géneros para que el test no dependa del azar
    train.loc[0, "Sex"] = "male"
    train.loc[1, "Sex"] = "female"
    train.loc[2, "Sex"] = "female"
    train.loc[3, "Sex"] = "female"
    
    # 2. Asignamos los nulos y edades
    train.loc[0, "Age"] = np.nan
    train.loc[1, "Age"] = np.nan
    train.loc[2, "Age"] = 40.0
    train.loc[3, "Age"] = 0.0
    
    # 3. Ejecutamos el imputador
    out = ml.GroupAgeImputer(group_cols=["Sex"]).fit_transform(train)
    
    # 4. La media global es (40 + 0) / 2 = 20. 
    # Como la fila 0 es hombre, la media de hombres es NaN (solo hay un hombre y es nulo).
    # Por lo tanto, debe usar la media global (20.0).
    assert out.loc[0, "Age"] == 20.0

def test_pipeline_fits_and_predicts_probabilities(tiny):
    X, y = tiny
    pipe = ml.build_pipeline(ml.get_candidates()["logreg"]).fit(X, y)
    assert len(pipe.predict(X)) == 6
    probs = pipe.predict_proba(X)
    assert probs.shape == (len(X), 2) and (0 <= probs).all() and (probs <= 1).all()

def test_split_is_stratified_and_reproducible(tiny):
    X, y = tiny
    a = ml.split(X, y, test_size=0.25, random_state=0)
    b = ml.split(X, y, test_size=0.25, random_state=0)
    pd.testing.assert_index_equal(a[1].index, b[1].index)
    assert abs(a[3].mean() - b[3].mean()) < 0.03

def test_candidates_include_baseline_and_are_seeded():
    cands = ml.get_candidates(42)
    assert "dummy" in cands and "logreg" in cands and len(cands) >= 5

def test_evaluate_on_test_returns_metrics(tiny): # <--- 'tiny', no 'toy'
    X, y = tiny
    X_tr, X_te, y_tr, y_te = ml.split(X, y, test_size=0.25, random_state=1)
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X_tr, y_tr)
    res = ml.evaluate_on_test(pipe, X_te, y_te, threshold=0.5)
    
    assert {"accuracy", "f1", "roc_auc", "n_test"} <= set(res)
    assert 0 <= res["accuracy"] <= 1 
    assert res["n_test"] == len(y_te)

def test_save_and_load_model(tmp_path, tiny):
    X, y = tiny
    pipe = ml.train_final(X, y)
    
    meta = {"test": "ok"}
    path = ml.save_model(pipe, meta, name="test_model", model_dir=tmp_path)
    assert path.exists()
    
    # AQUÍ ESTÁ EL CAMBIO: agregamos name="test_model"
    loaded_pipe, loaded_meta = ml.load_model(model_dir=tmp_path, name="test_model") 
    
    assert loaded_meta["test"] == "ok"
    assert hasattr(loaded_pipe, "predict")

def test_save_and_load_roundtrip(tmp_path, tiny):
    X, y = tiny
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    meta = {"version": "1.0", "stage": "train"}
    
    path = ml.save_model(pipe, meta, name="my_test_model", model_dir=tmp_path)
    loaded, meta_loaded = ml.load_model(model_dir=tmp_path, name="my_test_model")
    
    # Mismas predicciones
    np.testing.assert_array_equal(pipe.predict(X), loaded.predict(X))
    # Metadatos correctos
    assert {"sklearn_version", "python_version", "model_sha256", "saved_at"} <= set(meta_loaded)
    assert meta_loaded["version"] == "1.0"

def test_predict_returns_probabilities_and_labels(tiny):
    X, y = tiny
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    out = ml.predict_pipeline(pipe, X, threshold=0.5)
    
    # Verificamos que el dataframe tenga las columnas correctas
    assert {"proba", "pred"} <= set(out.columns)
    # Verificamos que las predicciones sean 0 o 1
    assert set(out["pred"].unique()) <= {0, 1}        