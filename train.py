"""Entrena y evalúa el modelo base de proyección de puntos fantasy half-PPR.

Uso:
    python train.py [--alpha 0.0] [--no-fit-intercept] [--positive] [--scale-numeric]
                    [--seed 42] [--train-fraction 0.8] [--data fantasy_data.csv] [--no-mlflow]

Reproducibilidad:
    - Dependencias fijadas en requirements.txt (Python 3.12):
      python -m pip install -r requirements.txt
    - --seed fija las semillas de `random`, NumPy y el `random_state` de Ridge; se
      registra en MLflow junto con el resto de hiperparámetros.
    - La partición es cronológica (sin muestreo aleatorio) y los datos se ordenan por
      Key/Year antes de crear los rezagos, por lo que con los mismos argumentos, datos y
      versiones de requirements.txt las predicciones y métricas son idénticas.
    - Cada corrida de MLflow guarda requirements.txt y la versión de Python.
"""
import argparse
import json
import platform
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLUMNS = ["Player", "Tm", "Pos", "Key"]
KEY_COLUMNS = ["Key", "Year", "Tm"]
TARGET_COLUMNS_WITH_MISSING = ["PointsTarget_half-ppr", "PPGTarget_half-ppr"]
PREVIOUS_SEASON_STATS = [
    "Age", "Exp", "G", "GS",
    "Pass_Yds", "Pass_TD", "Pass_Int",
    "Rush_Att", "Rush_Yds", "Rush_TD",
    "Rec_Tgt", "Rec_Rec", "Rec_Yds", "Rec_TD", "FmbLost",
    "Points_half-ppr", "PPG_half-ppr",
]
NUMERIC_FEATURES = [f"previous_{column}" for column in PREVIOUS_SEASON_STATS]
FEATURES = ["previous_Pos", *NUMERIC_FEATURES]
TARGET = "target_points"
EXPERIMENT_NAME = "TC5061_Fantasy_HalfPPR_NextSeason_LinearModels"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Lee el CSV de estadísticas fantasy."""
    df = pd.read_csv(csv_path)
    print(f"Archivo: {csv_path.resolve()} | Registros: {len(df):,} | Variables: {df.shape[1]}")
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Verifica tipos y reglas de plausibilidad; falla si hay errores verificables."""
    numeric_columns = df.select_dtypes(include="number").columns
    non_numeric_stats = [
        column for column in df.columns
        if column not in CATEGORICAL_COLUMNS and not pd.api.types.is_numeric_dtype(df[column])
    ]
    assert not non_numeric_stats, f"Columnas estadísticas no numéricas: {non_numeric_stats}"
    assert all(pd.api.types.is_string_dtype(df[column]) for column in CATEGORICAL_COLUMNS)

    # Los totales 2TM pueden superar el calendario de un solo equipo.
    is_2tm = df["Tm"] == "2TM"
    violations = {
        "G fuera de rango": ((df["G"] < 0) | ((df["G"] > df["num_games"]) & ~is_2tm)).sum(),
        "GS negativo o mayor que G": ((df["GS"] < 0) | (df["GS"] > df["G"])).sum(),
        "Porcentajes inválidos": (
            (df["games_played_pct"] < 0)
            | ((df["games_played_pct"] > 1) & ~is_2tm)
            | (df["games_started_pct"] < 0)
            | (df["games_started_pct"] > 1)
        ).sum(),
        "Valores infinitos": np.isinf(df[numeric_columns]).to_numpy().sum(),
    }
    failed = {rule: int(count) for rule, count in violations.items() if count}
    assert not failed, f"Violaciones de plausibilidad: {failed}"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados exactos y añade banderas de calidad sin imputar ni recortar."""
    df_clean = df.drop_duplicates().copy()
    for column in TARGET_COLUMNS_WITH_MISSING:
        df_clean[f"{column}_missing"] = df_clean[column].isna()
    df_clean["G_over_num_games_2TM"] = (df_clean["Tm"] == "2TM") & (
        df_clean["G"] > df_clean["num_games"]
    )

    assert not df_clean.duplicated(subset=KEY_COLUMNS).any()
    print(f"Filas limpias: {len(df_clean):,} "
          f"(duplicados eliminados: {len(df) - len(df_clean):,})")
    return df_clean


def build_lagged_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Usa la temporada consecutiva previa de cada jugador para predecir los puntos actuales."""
    source = df_clean.sort_values(["Key", "Year"])
    by_player = source.groupby("Key", sort=False)

    data = pd.DataFrame(index=source.index)
    data["target_year"] = source["Year"]
    data["target_Pos"] = source["Pos"]
    data["previous_year"] = by_player["Year"].shift(1)
    data["previous_Pos"] = by_player["Pos"].shift(1)
    data[TARGET] = source["Points_half-ppr"]
    for column in PREVIOUS_SEASON_STATS:
        data[f"previous_{column}"] = by_player[column].shift(1)

    consecutive = data["target_year"] == data["previous_year"] + 1
    data = data.loc[consecutive].dropna(subset=[*FEATURES, TARGET]).copy()
    print(f"Observaciones con temporadas consecutivas: {len(data):,}")
    return data


def temporal_split(data: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide por año objetivo: los años más recientes quedan en prueba."""
    target_years = sorted(data["target_year"].unique())
    cutoff_year = target_years[int(len(target_years) * train_fraction)]
    train = data.loc[data["target_year"] < cutoff_year]
    test = data.loc[data["target_year"] >= cutoff_year]

    assert not train.empty and not test.empty
    assert train["target_year"].max() < test["target_year"].min()
    print(f"Entrenamiento: {len(train):,} filas, hasta {train['target_year'].max():.0f}")
    print(f"Prueba: {len(test):,} filas, desde {test['target_year'].min():.0f}")
    return train, test


def build_pipeline(
    alpha: float, fit_intercept: bool, positive: bool, scale_numeric: bool, seed: int
) -> Pipeline:
    """Regresión lineal (Ridge si alpha > 0) con la posición previa codificada one-hot."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("posicion", OneHotEncoder(handle_unknown="ignore"), ["previous_Pos"]),
            ("estadisticas", StandardScaler() if scale_numeric else "passthrough", NUMERIC_FEATURES),
        ]
    )
    if alpha > 0:
        regressor = Ridge(alpha=alpha, fit_intercept=fit_intercept, positive=positive,
                          random_state=seed)
    else:
        regressor = LinearRegression(fit_intercept=fit_intercept, positive=positive)
    return Pipeline([("preprocesador", preprocessor), ("regresion", regressor)])


def train_models(train: pd.DataFrame, hyperparams: dict) -> tuple[Pipeline, DummyRegressor]:
    """Ajusta el modelo lineal y el baseline que predice la media del entrenamiento."""
    model = build_pipeline(**hyperparams).fit(train[FEATURES], train[TARGET])
    baseline = DummyRegressor(strategy="mean").fit(train[FEATURES], train[TARGET])
    return model, baseline


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate(
    model: Pipeline, baseline: DummyRegressor, test: pd.DataFrame
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    """Calcula métricas globales, mejora frente al baseline y errores por posición."""
    y_test = test[TARGET]
    linear_pred = model.predict(test[FEATURES])
    baseline_pred = baseline.predict(test[FEATURES])

    linear = regression_metrics(y_test, linear_pred)
    mean_baseline = regression_metrics(y_test, baseline_pred)
    metrics = {
        **{f"linear_{name}": value for name, value in linear.items()},
        "mean_baseline_mae": mean_baseline["mae"],
        "mean_baseline_rmse": mean_baseline["rmse"],
        "mae_improvement_pct": (mean_baseline["mae"] - linear["mae"]) / mean_baseline["mae"] * 100,
        "rmse_improvement_pct": (mean_baseline["rmse"] - linear["rmse"]) / mean_baseline["rmse"] * 100,
        "linear_mean_bias": float((linear_pred - y_test.to_numpy()).mean()),
    }

    results = pd.DataFrame({
        "Posición": test["target_Pos"].to_numpy(),
        "real": y_test.to_numpy(),
        "lineal": linear_pred,
        "media": baseline_pred,
    })
    by_position = results.groupby("Posición").apply(
        lambda group: pd.Series({
            "Registros": len(group),
            "MAE lineal": mean_absolute_error(group["real"], group["lineal"]),
            "MAE media": mean_absolute_error(group["real"], group["media"]),
            "Sesgo lineal": (group["lineal"] - group["real"]).mean(),
        }),
        include_groups=False,
    )
    for position, row in by_position.iterrows():
        metrics[f"linear_mae_{position.lower()}"] = float(row["MAE lineal"])
        metrics[f"linear_bias_{position.lower()}"] = float(row["Sesgo lineal"])

    return metrics, by_position, linear_pred


def save_evaluation_artifacts(
    output_dir: Path,
    metrics: dict[str, float],
    by_position: pd.DataFrame,
    y_test: pd.Series,
    linear_pred: np.ndarray,
) -> list[Path]:
    """Guarda métricas (JSON), errores por posición (CSV) y gráficas de diagnóstico (PNG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    by_position_path = output_dir / "errors_by_position.csv"
    by_position.round(4).to_csv(by_position_path)

    signed_error = linear_pred - y_test.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].scatter(y_test, linear_pred, s=12, alpha=0.25, color="#197278")
    plot_limit = max(float(y_test.max()), float(linear_pred.max()))
    axes[0].plot([0, plot_limit], [0, plot_limit], linestyle="--", color="#d1495b")
    axes[0].set_title("Puntos reales vs. predichos")
    axes[0].set_xlabel("Puntos reales")
    axes[0].set_ylabel("Puntos predichos")
    axes[0].grid(alpha=0.2)
    axes[1].hist(signed_error, bins=35, color="#197278", edgecolor="white")
    axes[1].axvline(0, linestyle="--", color="#d1495b")
    axes[1].set_title("Distribución del error firmado")
    axes[1].set_xlabel("Predicción - valor real (puntos)")
    axes[1].set_ylabel("Frecuencia")
    plot_path = output_dir / "predicted_vs_actual.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    return [metrics_path, by_position_path, plot_path]


def log_to_mlflow(
    model: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    metrics: dict[str, float],
    by_position: pd.DataFrame,
    linear_pred: np.ndarray,
    tracking_db: Path,
    hyperparams: dict,
    train_fraction: float,
    experiment_name: str,
    run_name: str,
) -> str:
    """Registra parámetros, métricas, artefactos de evaluación y el pipeline en MLflow."""
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    experiment = mlflow.set_experiment(experiment_name)
    params = {
        "model": type(model.named_steps["regresion"]).__name__,
        **hyperparams,
        "train_fraction": train_fraction,
        "target": "Points_half-ppr_next_season",
        "feature_source": "previous_consecutive_season",
        "features": ",".join(FEATURES),
        "position_encoding": "OneHotEncoder(handle_unknown=ignore)",
        "split_strategy": "chronological_by_target_year",
        "train_year_start": int(train["target_year"].min()),
        "train_year_end": int(train["target_year"].max()),
        "test_year_start": int(test["target_year"].min()),
        "test_year_end": int(test["target_year"].max()),
        "train_rows": len(train),
        "test_rows": len(test),
        "consecutive_seasons_only": True,
    }

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=run_name,
    ) as run:
        for name, value in params.items():
            mlflow.log_param(name, value)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.set_tags({
            "dataset": "fantasy_data.csv",
            "validation": "temporal_holdout",
            "stage": "baseline",
            "python_version": platform.python_version(),
        })
        requirements_path = Path(__file__).resolve().parent / "requirements.txt"
        if requirements_path.exists():
            mlflow.log_artifact(str(requirements_path), artifact_path="environment")
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_paths = save_evaluation_artifacts(
                Path(tmp_dir), metrics, by_position, test[TARGET], linear_pred
            )
            for path in artifact_paths:
                mlflow.log_artifact(str(path), artifact_path="evaluation")
        logged_model = mlflow.sklearn.log_model(
            sk_model=model,
            name="linear_regression_baseline",
            input_example=train[FEATURES].head(5),
        )

    reloaded = mlflow.sklearn.load_model(logged_model.model_uri)
    assert np.allclose(reloaded.predict(test[FEATURES].head(5)), linear_pred[:5])
    logged_artifacts = {item.path for item in mlflow.MlflowClient().list_artifacts(run.info.run_id, "evaluation")}
    assert logged_artifacts == {f"evaluation/{path.name}" for path in artifact_paths}
    print(f"MLflow run: {run.info.run_id} | Modelo: {logged_model.model_uri}")
    print(f"Artefactos registrados: {sorted(logged_artifacts)}")
    return run.info.run_id


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model_group = parser.add_argument_group("hiperparámetros del modelo")
    model_group.add_argument("--alpha", type=float, default=0.0,
                             help="Regularización L2; 0 usa LinearRegression, > 0 usa Ridge.")
    model_group.add_argument("--fit-intercept", action=argparse.BooleanOptionalAction, default=True,
                             help="Estimar el intercepto del modelo.")
    model_group.add_argument("--positive", action=argparse.BooleanOptionalAction, default=False,
                             help="Forzar coeficientes no negativos.")
    model_group.add_argument("--scale-numeric", action=argparse.BooleanOptionalAction, default=False,
                             help="Estandarizar las variables numéricas (recomendado con alpha > 0).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument("--train-fraction", type=float, default=0.8,
                        help="Fracción de años objetivo usada para entrenamiento.")
    parser.add_argument("--data", type=Path, default=script_dir / "fantasy_data.csv")
    parser.add_argument("--tracking-db", type=Path, default=script_dir / "mlflow.db")
    parser.add_argument("--no-mlflow", action="store_true", help="No registrar la corrida en MLflow.")
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME, help="Experimento de MLflow.")
    parser.add_argument("--run-name", default=None,
                        help="Nombre de la corrida; por defecto se genera con los hiperparámetros.")
    args = parser.parse_args()

    if args.alpha < 0:
        parser.error("--alpha debe ser >= 0.")
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction debe estar entre 0 y 1 (exclusivo).")
    return args


def set_seed(seed: int) -> None:
    """Fija las semillas globales; scikit-learn usa el estado de NumPy si no recibe random_state."""
    random.seed(seed)
    np.random.seed(seed)


def default_run_name(hyperparams: dict) -> str:
    model = f"ridge_a{hyperparams['alpha']:g}" if hyperparams["alpha"] > 0 else "ols"
    parts = [model]
    if not hyperparams["fit_intercept"]:
        parts.append("nointercept")
    if hyperparams["positive"]:
        parts.append("positive")
    if hyperparams["scale_numeric"]:
        parts.append("scaled")
    parts.append(f"seed{hyperparams['seed']}")
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    hyperparams = {
        "alpha": args.alpha,
        "fit_intercept": args.fit_intercept,
        "positive": args.positive,
        "scale_numeric": args.scale_numeric,
        "seed": args.seed,
    }
    print(f"Hiperparámetros: {hyperparams} | train_fraction: {args.train_fraction}")

    df = load_data(args.data)
    validate_data(df)
    df_clean = clean_data(df)
    data = build_lagged_dataset(df_clean)
    train, test = temporal_split(data, args.train_fraction)

    model, baseline = train_models(train, hyperparams)
    metrics, by_position, linear_pred = evaluate(model, baseline, test)

    print("\nMétricas en prueba:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")
    print("\nErrores por posición:")
    print(by_position.round(2).to_string())

    if not args.no_mlflow:
        log_to_mlflow(model, train, test, metrics, by_position, linear_pred, args.tracking_db,
                      hyperparams, args.train_fraction, args.experiment_name,
                      args.run_name or default_run_name(hyperparams))


if __name__ == "__main__":
    main()
