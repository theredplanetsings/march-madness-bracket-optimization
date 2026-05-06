"""Prototype EV-optimized training via random search.

We avoid overwriting existing models/outputs by writing to:
  - data/processed/ev_opt_results.csv
  - models/ev_opt_theta.json

The objective is average expected ESPN score (EV) for an EV-max bracket
under the candidate win-probability model across historical seasons.
"""
from __future__ import annotations
import json
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backtest import build_bracket, expected_score
from bracket import simulate_many
from strategies import ev_max_strategy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = f"{ROOT}/data/processed"
MODELS = f"{ROOT}/models"

KENPOM_FEATS = ["AdjEM", "AdjO", "AdjD", "AdjT", "Luck", "SOS-AdjEM", "NCSOS-AdjEM",
                "Momentum", "OffEFG", "DefEFG"]
DELTA_COLS = [f"d_{f}" for f in KENPOM_FEATS] + ["d_Seed"]


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def build_win_prob_from_theta(kp_year: pd.DataFrame, seeds_year: pd.DataFrame,
                              scaler: StandardScaler, theta: np.ndarray,
                              intercept: float):
    kp = kp_year.set_index("Team")
    seed_lookup = seeds_year.set_index("Team")["Seed"].to_dict()
    mean = scaler.mean_
    scale = scaler.scale_

    def win_prob(a: str, b: str) -> float:
        ra, rb = kp.loc[a], kp.loc[b]
        deltas = [ra[f] - rb[f] for f in KENPOM_FEATS]
        sa, sb = seed_lookup.get(a, np.nan), seed_lookup.get(b, np.nan)
        deltas.append((sa - sb) if (sa == sa and sb == sb) else 0.0)
        x = (np.array(deltas) - mean) / scale
        return float(sigmoid(intercept + float(np.dot(theta, x))))

    return win_prob


def evaluate_candidate(theta: np.ndarray, intercept: float,
                        scaler: StandardScaler,
                        seasons: list[int], seeds: pd.DataFrame, kp: pd.DataFrame,
                        n_sims: int, rng: np.random.Generator) -> dict:
    ev_scores = []
    for season in seasons:
        seeds_year = seeds[seeds.Season == season]
        kp_year = kp[kp.Season == season]
        bracket = build_bracket(season, seeds_year)
        win_prob = build_win_prob_from_theta(kp_year, seeds_year, scaler, theta, intercept)
        reach = simulate_many(bracket, win_prob, n_sims=n_sims, seed=int(rng.integers(1, 1_000_000)))
        picks = ev_max_strategy(bracket, reach)
        ev_scores.append(expected_score(picks, reach))
    ev_scores = np.array(ev_scores)
    return {
        "mean_ev": float(ev_scores.mean()),
        "std_ev": float(ev_scores.std()),
    }


def random_search(base_theta: np.ndarray, base_intercept: float,
                  scaler: StandardScaler,
                  seasons: list[int], seeds: pd.DataFrame, kp: pd.DataFrame,
                  n_sims: int, n_candidates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    # Include baseline
    base_stats = evaluate_candidate(base_theta, base_intercept, scaler, seasons, seeds, kp, n_sims, rng)
    rows.append({
        "candidate": 0,
        "kind": "baseline",
        "mean_ev": base_stats["mean_ev"],
        "std_ev": base_stats["std_ev"],
        "intercept": float(base_intercept),
        **{f"theta_{i}": float(v) for i, v in enumerate(base_theta)},
    })

    # Random perturbations around baseline
    for i in range(1, n_candidates + 1):
        scale = 0.25
        noise = rng.normal(0.0, scale, size=base_theta.shape)
        theta = base_theta + noise
        intercept = base_intercept + rng.normal(0.0, scale * 0.2)
        stats = evaluate_candidate(theta, intercept, scaler, seasons, seeds, kp, n_sims, rng)
        rows.append({
            "candidate": i,
            "kind": "random",
            "mean_ev": stats["mean_ev"],
            "std_ev": stats["std_ev"],
            "intercept": float(intercept),
            **{f"theta_{j}": float(v) for j, v in enumerate(theta)},
        })
    return pd.DataFrame(rows)


def parse_seasons(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=2000)
    parser.add_argument("--n-candidates", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seasons", type=str, default="2010-2025")
    args = parser.parse_args()

    matchups = pd.read_csv(f"{PROC}/matchups.csv")
    seeds = pd.read_csv(f"{PROC}/tourney_seeds.csv")
    kp = pd.read_csv(f"{PROC}/kenpom_all.csv")

    seasons = [s for s in parse_seasons(args.seasons) if s not in (2020, 2021)]

    X = matchups[DELTA_COLS].values
    y = matchups["label"].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    base_model = LogisticRegression(max_iter=1000, C=1.0)
    base_model.fit(Xs, y)
    base_theta = base_model.coef_[0]
    base_intercept = float(base_model.intercept_[0])

    df = random_search(base_theta, base_intercept, scaler, seasons, seeds, kp,
                       n_sims=args.n_sims, n_candidates=args.n_candidates,
                       seed=args.seed)
    df = df.sort_values("mean_ev", ascending=False).reset_index(drop=True)

    os.makedirs(PROC, exist_ok=True)
    out_csv = f"{PROC}/ev_opt_results.csv"
    df.to_csv(out_csv, index=False)

    best = df.iloc[0]
    out_json = f"{MODELS}/ev_opt_theta.json"
    with open(out_json, "w") as f:
        json.dump({
            "mean_ev": float(best["mean_ev"]),
            "std_ev": float(best["std_ev"]),
            "intercept": float(best["intercept"]),
            "theta": [float(best[f"theta_{i}"]) for i in range(len(base_theta))],
            "n_sims": args.n_sims,
            "n_candidates": args.n_candidates,
            "seasons": seasons,
        }, f, indent=2)

    print(f"Saved results -> {out_csv}")
    print(f"Saved best theta -> {out_json}")
    print("Top 5 candidates:")
    print(df[["candidate", "kind", "mean_ev", "std_ev"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
