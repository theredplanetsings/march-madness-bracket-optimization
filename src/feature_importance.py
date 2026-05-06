"""Round-specific feature importance.

The proposal asked whether some features matter more in late rounds than
early rounds (e.g. "does 3-point shooting matter more in the Sweet 16
than in the Round of 64?"). We answer this two ways:

1. Per-round logistic regression. For each round R we fit an LR on the
   matchup rows for that round and report standardized coefficients.
   Larger |coef| means the feature has more pull on the predicted
   probability conditional on being in that round.

2. Per-round permutation importance, using the LOSO out-of-fold
   predictions of each model (full LR, XGBoost, Random Forest, LightGBM,
   CatBoost). For each (model, round, feature) we shuffle that single
   feature column among the round's rows, refit nothing, recompute the
   probability with the persisted final model, and measure the increase
   in log-loss vs. the unpermuted prediction. We average over 30 random
   shuffles for stability.

Outputs:
  data/processed/round_importance_lr.csv      (per-round LR coefs)
  data/processed/round_importance_perm.csv    (per-round permutation imp.)
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Callable

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names, but")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = f"{ROOT}/data/processed"
MODELS = f"{ROOT}/models"

KENPOM_FEATS = ["AdjEM", "AdjO", "AdjD", "AdjT", "Luck", "SOS-AdjEM", "NCSOS-AdjEM",
                "Momentum", "OffEFG", "DefEFG"]
DELTA_COLS = [f"d_{f}" for f in KENPOM_FEATS] + ["d_Seed"]
ROUND_ORDER = ["First Round", "Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship"]
ROUND_SHORT = {"First Round": "R64", "Second Round": "R32", "Sweet 16": "S16",
               "Elite 8": "E8", "Final Four": "F4", "Championship": "NCG"}


# --------------------------------------------------------------------- LR

def per_round_lr_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    """Per-round logistic regression with standardized features. Returns a
    long-form DataFrame with columns: Round, Feature, Coef, AbsCoef, n."""
    rows = []
    for rname in ROUND_ORDER:
        sub = df[df["Round"] == rname]
        if len(sub) < 20:
            continue
        X = sub[DELTA_COLS].values
        y = sub["label"].values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(Xs, y)
        coefs = m.coef_[0]
        for fname, c in zip(DELTA_COLS, coefs):
            rows.append({
                "Round": ROUND_SHORT[rname],
                "Feature": fname,
                "Coef": float(c),
                "AbsCoef": float(abs(c)),
                "n": int(len(sub)),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------- permutation imp.

def _predict_proba_from_bundle(bundle: dict, X: np.ndarray) -> np.ndarray:
    """Score X using a saved (scaler+model) or (model) bundle."""
    if "scaler" in bundle:
        Xs = bundle["scaler"].transform(X)
        return bundle["model"].predict_proba(Xs)[:, 1]
    return bundle["model"].predict_proba(X)[:, 1]


def per_round_permutation_importance(df: pd.DataFrame, model_name: str,
                                     n_repeats: int = 30, seed: int = 0) -> pd.DataFrame:
    """Permutation importance per round using the persisted final model.

    The persisted model is trained on all data; we only use it as a fixed
    scoring function. For each round we permute one feature column at a
    time among that round's rows, score with the model, and measure the
    increase in log-loss vs. the unpermuted prediction.
    """
    bundle = joblib.load(f"{MODELS}/{model_name}.joblib")
    feats = bundle.get("features", DELTA_COLS)
    rng = np.random.default_rng(seed)

    rows = []
    for rname in ROUND_ORDER:
        sub = df[df["Round"] == rname].reset_index(drop=True)
        if len(sub) < 20:
            continue
        X = sub[feats].values.copy()
        y = sub["label"].values
        base_p = _predict_proba_from_bundle(bundle, X)
        base_ll = log_loss(y, base_p, labels=[0, 1])
        for fi, fname in enumerate(feats):
            deltas = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[:, fi] = rng.permutation(Xp[:, fi])
                p = _predict_proba_from_bundle(bundle, Xp)
                deltas.append(log_loss(y, p, labels=[0, 1]) - base_ll)
            rows.append({
                "Model": model_name,
                "Round": ROUND_SHORT[rname],
                "Feature": fname,
                "MeanDeltaLogLoss": float(np.mean(deltas)),
                "StdDeltaLogLoss": float(np.std(deltas)),
                "BaseLogLoss": float(base_ll),
                "n": int(len(sub)),
            })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(f"{PROC}/matchups.csv")
    print(f"Matchups: {len(df)} rows, rounds: {sorted(df['Round'].unique())}")

    print("\n=== Per-round LR coefficients (standardized) ===")
    lr_imp = per_round_lr_coefficients(df)
    pivot = lr_imp.pivot(index="Feature", columns="Round", values="Coef")
    pivot = pivot[[c for c in ["R64", "R32", "S16", "E8", "F4", "NCG"] if c in pivot.columns]]
    print(pivot.round(3).to_string())
    out_lr = f"{PROC}/round_importance_lr.csv"
    lr_imp.to_csv(out_lr, index=False)
    print(f"-> {out_lr}")

    print("\n=== Per-round permutation importance ===")
    all_perm = []
    candidates = ["full_LR", "xgb", "rf", "lgbm", "catboost"]
    for name in candidates:
        path = f"{MODELS}/{name}.joblib"
        if not os.path.exists(path):
            print(f"  [skip] no model file at {path}")
            continue
        print(f"  {name} ...", flush=True)
        perm = per_round_permutation_importance(df, name, n_repeats=30, seed=42)
        all_perm.append(perm)
    if all_perm:
        perm_df = pd.concat(all_perm, ignore_index=True)
        out_perm = f"{PROC}/round_importance_perm.csv"
        perm_df.to_csv(out_perm, index=False)
        print(f"-> {out_perm}")

        print("\n=== Top features per round (full_LR permutation) ===")
        sub = perm_df[perm_df.Model == "full_LR"]
        for r in ["R64", "R32", "S16", "E8", "F4", "NCG"]:
            r_sub = sub[sub.Round == r].sort_values("MeanDeltaLogLoss", ascending=False)
            if len(r_sub) == 0:
                continue
            top3 = r_sub.head(3)
            tops = ", ".join(f"{row.Feature}({row.MeanDeltaLogLoss:+.3f})" for row in top3.itertuples())
            print(f"  {r}: {tops}")


if __name__ == "__main__":
    main()
