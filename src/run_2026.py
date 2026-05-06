"""End-to-end run for the 2026 NCAA tournament.

For 2026 we:
  1. Load the bracket
  2. Train the win-probability model on 2010-2025 (LOSO-style — exclude 2026)
  3. Run K=10000 Monte Carlo tournaments using the model
  4. Build candidate strategies (chalk_seed, chalk_model, ev_max)
  5. Score each strategy against (a) actual 2026 outcomes (point-estimate)
     and (b) the K simulated outcomes (distribution + pool percentile)
  6. Save tables of results and a per-team reach-probability summary
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from bracket import simulate_many, simulate_outcomes
from backtest import (
    build_bracket, train_model_excluding, build_win_prob,
    actual_advancement, score_picks
)
from strategies import (
    chalk_seed_strategy, chalk_model_strategy, ev_max_strategy,
    evaluate_strategies
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = f"{ROOT}/data/processed"

SEASON = 2026
N_SIMS = 10000


def main():
    matchups = pd.read_csv(f"{PROC}/matchups.csv")
    seeds = pd.read_csv(f"{PROC}/tourney_seeds.csv")
    kp = pd.read_csv(f"{PROC}/kenpom_all.csv")
    results = pd.read_csv(f"{PROC}/tourney_results.csv")

    seeds_y = seeds[seeds.Season == SEASON]
    kp_y = kp[kp.Season == SEASON]

    bracket = build_bracket(SEASON, seeds_y)
    seed_lookup = seeds_y.set_index("Team")["Seed"].to_dict()

    print(f"=== Training model (full LR, excluding {SEASON}) ===")
    scaler, model = train_model_excluding(matchups, SEASON)
    win_prob = build_win_prob(scaler, model, kp_y, seeds_y)

    # Inspect a few key matchups
    print("\nKey matchup probabilities (model predictions):")
    for a, b in [("Duke", "Connecticut"), ("Houston", "Florida"), ("Auburn", "Tennessee")]:
        if a in seed_lookup and b in seed_lookup:
            print(f"  P({a} beats {b}) = {win_prob(a, b):.3f}")

    print(f"\n=== Monte Carlo simulation: {N_SIMS} tournaments ===")
    reach_probs_model = simulate_many(bracket, win_prob, n_sims=N_SIMS, seed=42)
    print("Top 10 by P(champion):")
    top_champ = reach_probs_model.sort_values("p_Champ", ascending=False).head(10)
    print(top_champ.round(3).to_string())

    print("\n=== Building strategies ===")
    strategies = {
        "chalk_seed":  chalk_seed_strategy(bracket, seed_lookup),
        "chalk_model": chalk_model_strategy(bracket, win_prob),
        "ev_max":      ev_max_strategy(bracket, reach_probs_model),
    }
    for name, picks in strategies.items():
        print(f"  {name:12s}: champion = {picks['NCG'][0]}, F4 = {picks['F4']}")

    print("\n=== Scoring strategies vs ACTUAL 2026 outcomes ===")
    actual = actual_advancement(SEASON, results)
    actual_rows = []
    for name, picks in strategies.items():
        total, per_round = score_picks(picks, actual)
        actual_rows.append({"strategy": name, "actual_score": total, **per_round})
    actual_df = pd.DataFrame(actual_rows)
    print(actual_df.to_string(index=False))
    actual_df.to_csv(f"{PROC}/2026_actual_scores.csv", index=False)

    print(f"\n=== Simulating outcomes for distribution analysis ({N_SIMS}) ===")
    outcomes, teams_order, _ = simulate_outcomes(bracket, win_prob, n_sims=N_SIMS, seed=43)
    # Pass a dummy equal-sized public_outcomes array (not used in scoring here)
    eval_df = evaluate_strategies(strategies, outcomes, teams_order)
    print("\n=== Strategy score distribution (over simulated outcomes) ===")
    print(eval_df.round(3).to_string(index=False))
    eval_df.to_csv(f"{PROC}/2026_strategy_distribution.csv", index=False)

    # Save reach probs and picks
    reach_probs_model.to_csv(f"{PROC}/2026_reach_probs_model.csv")
    for name, picks in strategies.items():
        with open(f"{PROC}/2026_picks_{name}.json", "w") as f:
            json.dump(picks, f, indent=2)


if __name__ == "__main__":
    main()
