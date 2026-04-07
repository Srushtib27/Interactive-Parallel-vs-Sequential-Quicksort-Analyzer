"""
ml_model.py - Lightweight Regression Model for Execution Time Prediction
Uses polynomial regression (numpy-based, no sklearn required).
"""
import numpy as np
import json
import os
from analytics import load_results, BenchmarkResult

MODEL_FILE = "ml_model_coeffs.json"


def _features(n: int, processes: int) -> np.ndarray:
    """Build feature vector: [n, log(n), n*log(n), processes, 1/processes, n/processes]"""
    log_n = np.log(n + 1)
    return np.array([
        n,
        log_n,
        n * log_n,
        processes,
        1.0 / max(processes, 1),
        n / max(processes, 1),
        1.0,  # bias
    ])


def train_model(results: list[BenchmarkResult]) -> dict | None:
    """Fit separate linear regressors for seq and par times. Returns coeffs dict."""
    if len(results) < 4:
        return None

    X_seq, y_seq, X_par, y_par = [], [], [], []

    for r in results:
        feat = _features(r.n, r.processes)
        X_seq.append(feat)
        y_seq.append(r.seq_time)
        X_par.append(feat)
        y_par.append(r.par_time)

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    X_par = np.array(X_par)
    y_par = np.array(y_par)

    # Least-squares via numpy
    try:
        coeffs_seq, _, _, _ = np.linalg.lstsq(X_seq, y_seq, rcond=None)
        coeffs_par, _, _, _ = np.linalg.lstsq(X_par, y_par, rcond=None)
    except np.linalg.LinAlgError:
        return None

    model = {
        "coeffs_seq": coeffs_seq.tolist(),
        "coeffs_par": coeffs_par.tolist(),
        "n_samples": len(results),
    }

    with open(MODEL_FILE, "w") as f:
        json.dump(model, f, indent=2)

    return model


def load_model() -> dict | None:
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE) as f:
        return json.load(f)


def predict(n: int, processes: int, model: dict = None) -> dict | None:
    """Predict seq/par times given n and processes."""
    if model is None:
        model = load_model()
    if model is None:
        return None

    feat = _features(n, processes)
    pred_seq = float(np.dot(feat, model["coeffs_seq"]))
    pred_par = float(np.dot(feat, model["coeffs_par"]))

    pred_seq = max(pred_seq, 1e-6)
    pred_par = max(pred_par, 1e-6)

    return {
        "pred_seq": round(pred_seq, 6),
        "pred_par": round(pred_par, 6),
        "pred_speedup": round(pred_seq / pred_par, 4),
    }


def generate_prediction_curve(processes: int, model: dict) -> tuple:
    """Return arrays of (sizes, pred_seq_times, pred_par_times) for plotting."""
    sizes = np.linspace(1000, 50000, 30, dtype=int)
    pred_seq = []
    pred_par = []
    for n in sizes:
        p = predict(int(n), processes, model)
        if p:
            pred_seq.append(p["pred_seq"])
            pred_par.append(p["pred_par"])
        else:
            pred_seq.append(0)
            pred_par.append(0)
    return sizes.tolist(), pred_seq, pred_par


def compute_r2(results: list[BenchmarkResult], model: dict) -> dict:
    """Compute R² for seq and par predictions."""
    y_seq_true = np.array([r.seq_time for r in results])
    y_par_true = np.array([r.par_time for r in results])
    y_seq_pred = np.array([predict(r.n, r.processes, model)["pred_seq"] for r in results])
    y_par_pred = np.array([predict(r.n, r.processes, model)["pred_par"] for r in results])

    def r2(true, pred):
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "r2_seq": round(r2(y_seq_true, y_seq_pred), 4),
        "r2_par": round(r2(y_par_true, y_par_pred), 4),
    }
