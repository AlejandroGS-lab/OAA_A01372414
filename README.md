# TC5061 – Proyección de puntos fantasy half-PPR

Entrena una regresión lineal (o Ridge) que estima los puntos fantasy half-PPR de una temporada a partir de la temporada consecutiva anterior del jugador (`fantasy_data.csv`). Evalúa contra un baseline de la media con una partición temporal y registra cada corrida en MLflow.

## Requisitos

- Python 3.12
- Dependencias fijadas en `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

## Ejecución

Desde la carpeta del proyecto:

```powershell
python train.py
```

Sin argumentos usa los valores por defecto y registra la corrida en `mlflow.db`, dentro del experimento `TC5061_Fantasy_HalfPPR_NextSeason_LinearModels`.

### Argumentos

| Argumento | Defecto | Descripción |
|---|---|---|
| `--alpha` | `0.0` | Regularización L2; `0` usa `LinearRegression`, `> 0` usa `Ridge`. |
| `--fit-intercept` / `--no-fit-intercept` | activado | Estimar el intercepto. |
| `--positive` / `--no-positive` | desactivado | Forzar coeficientes no negativos. |
| `--scale-numeric` / `--no-scale-numeric` | desactivado | Estandarizar variables numéricas. |
| `--seed` | `42` | Semilla aleatoria. |
| `--train-fraction` | `0.8` | Fracción de años objetivo para entrenamiento. |
| `--data` | `fantasy_data.csv` | Ruta del dataset. |
| `--tracking-db` | `mlflow.db` | Base SQLite de MLflow. |
| `--experiment-name` | `TC5061_Fantasy_HalfPPR_NextSeason_LinearModels` | Experimento de MLflow. |
| `--run-name` | generado | Nombre de la corrida (por defecto se arma con los hiperparámetros). |
| `--no-mlflow` | — | Entrenar y evaluar sin registrar en MLflow. |

Ejemplos:

```powershell
python train.py --scale-numeric --alpha 10
python train.py --positive --no-mlflow
python train.py --help
```

## Resultados en MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir <http://127.0.0.1:5000>, entrar al experimento, seleccionar las corridas y pulsar **Compare**. Cada corrida guarda:

- los hiperparámetros y las métricas (MAE, RMSE, R² y error por posición);
- los artefactos `evaluation/metrics.json`, `evaluation/errors_by_position.csv` y `evaluation/predicted_vs_actual.png`, además de `environment/requirements.txt`;
- el pipeline entrenado.

## Archivos

- `train.py`: script de entrenamiento y evaluación.
- `requirements.txt`: dependencias con versiones fijas.
- `analisis_resultados.md`: análisis de las corridas y capturas de MLflow (`capturas/`).
- `TC5061_Notebook_Baseline_GutierrezAlejandro.ipynb`: notebook original (EDA y baseline).
