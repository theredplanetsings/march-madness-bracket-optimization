"""Train win-probability models on the matchup feature matrix.

We compare three models, all evaluated by leave-one-season-out (LOSO) CV
which mimics how we'd really apply the model — train on past tournaments,
predict the next.

  1. Seed-only logistic regression (baseline)
  2. Full-feature logistic regression (KenPom deltas + seed)
  3. XGBoost on the same feature set

Metrics: log-loss (proper scoring rule, also the optimization target),
accuracy, and Brier score. Calibration matters for downstream EV
optimization, so we also report a reliability bin.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    RandomForestClassifier = None
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import xgboost as xgb
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = f"{ROOT}/data/processed"
MODELS = f"{ROOT}/models"
os.makedirs(MODELS, exist_ok=True)

KENPOM_FEATS = ["AdjEM", "AdjO", "AdjD", "AdjT", "Luck", "SOS-AdjEM", "NCSOS-AdjEM",
                "Momentum", "OffEFG", "DefEFG"]
DELTA_COLS = [f"d_{f}" for f in KENPOM_FEATS] + ["d_Seed"]
SEED_ONLY = ["d_Seed"]


def loso_eval(df: pd.DataFrame, feature_cols: list[str], make_model):
    """Leave-one-season-out evaluation. Returns per-fold metrics + concat OOF preds."""
    seasons = sorted(df["Season"].unique())
    fold_metrics = []
    oof = np.full(len(df), np.nan)
    for s in seasons:
        train = df[df.Season != s]
        test = df[df.Season == s]
        Xtr, ytr = train[feature_cols].values, train["label"].values
        Xte, yte = test[feature_cols].values, test["label"].values

        model = make_model()
        if isinstance(model, tuple):  # (scaler, model) for logistic
            scaler, m = model
            scaler.fit(Xtr); Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)
            m.fit(Xtr_s, ytr)
            p = m.predict_proba(Xte_s)[:, 1]
        else:
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
        oof[test.index] = p
        fold_metrics.append({
            "season": s, "n": len(test),
            "log_loss": log_loss(yte, p, labels=[0, 1]),
            "brier": brier_score_loss(yte, p),
            "acc": accuracy_score(yte, p > 0.5),
        })
    fm = pd.DataFrame(fold_metrics)
    summary = {
        "log_loss_mean": fm.log_loss.mean(),
        "log_loss_std": fm.log_loss.std(),
        "brier_mean": fm.brier.mean(),
        "acc_mean": fm.acc.mean(),
        "n_total": int(fm.n.sum()),
    }
    return summary, fm, oof


def reliability_bins(y: np.ndarray, p: np.ndarray, nbins: int = 10):
    """Return DataFrame of (bin_center, predicted_mean, observed_mean, count)."""
    bins = np.linspace(0, 1, nbins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, nbins - 1)
    rows = []
    for b in range(nbins):
        m = idx == b
        if m.sum() == 0: continue
        rows.append({"bin": b, "center": (bins[b] + bins[b+1]) / 2,
                     "pred_mean": p[m].mean(), "obs_mean": y[m].mean(), "n": m.sum()})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(f"{PROC}/matchups.csv").reset_index(drop=True)
    print(f"Matchups loaded: {len(df)} rows, seasons {df.Season.min()}..{df.Season.max()}")

    configs = [
        ("seed_only_LR", SEED_ONLY, lambda: (StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))),
        ("full_LR", DELTA_COLS, lambda: (StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))),
        ("xgb", DELTA_COLS, lambda: xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=4)),
    ]
    if RandomForestClassifier is not None:
        configs.append(("rf", DELTA_COLS, lambda: RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            random_state=42, n_jobs=4)))
    if lgb is not None:
        configs.append(("lgbm", DELTA_COLS, lambda: lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42)))
    if CatBoostClassifier is not None:
        configs.append(("catboost", DELTA_COLS, lambda: CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05,
            loss_function="Logloss", verbose=False, random_seed=42)))

    all_summaries = {}
    oof_store = {}
    for name, feats, mk in configs:
        print(f"\n=== {name} ===")
        s, fm, oof = loso_eval(df, feats, mk)
        print(json.dumps(s, indent=2, default=float))
        all_summaries[name] = s
        oof_store[name] = oof

    # Save OOF preds and summaries
    out = df[["Season", "Round", "TeamA", "TeamB", "SeedA", "SeedB", "label"]].copy()
    for name, oof in oof_store.items():
        out[f"p_{name}"] = oof
    out.to_csv(f"{PROC}/oof_predictions.csv", index=False)
    with open(f"{MODELS}/cv_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=float)

    # Reliability bins for the best model
    best = min(all_summaries.items(), key=lambda kv: kv[1]["log_loss_mean"])[0]
    print(f"\nBest model by log-loss: {best}")
    rel = reliability_bins(out["label"].values, out[f"p_{best}"].values, nbins=10)
    print("Reliability bins:")
    print(rel.to_string(index=False))
    rel.to_csv(f"{PROC}/reliability_{best}.csv", index=False)

    # Train final models on all data and persist
    print("\nFitting final models on all data...")
    for name, feats, mk in configs:
        m = mk()
        if isinstance(m, tuple):
            scaler, model = m
            X = df[feats].values
            scaler.fit(X)
            model.fit(scaler.transform(X), df["label"].values)
            joblib.dump({"scaler": scaler, "model": model, "features": feats},
                        f"{MODELS}/{name}.joblib")
        else:
            X = df[feats].values
            m.fit(X, df["label"].values)
            joblib.dump({"model": m, "features": feats}, f"{MODELS}/{name}.joblib")
    print(f"Models saved to {MODELS}/")


if __name__ == "__main__":
    main()
