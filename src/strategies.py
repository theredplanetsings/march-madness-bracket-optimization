"""Bracket pick-strategies and Monte-Carlo evaluation against a public field.

We compare three strategies for how to fill out a bracket given:
  - a per-matchup win-probability model (from train_models)
  - a per-team public-pick prior (from public_picks)

Strategies (all return picks in the {round_name: list[team]} format used
by backtest.score_picks):

  1. chalk_seed     — pick lower-seeded team at every game
  2. chalk_model    — pick the model favorite at every game (greedy wrt P)
  3. ev_max         — at each game, pick the team that maximizes
                       expected raw points (uses pre-computed reach probs)
  4. leverage       — like chalk_model but for late-round games (S16+)
                       prefer the team with the highest model-vs-public
                       leverage in the corresponding round

For each candidate strategy we run K Monte Carlo tournaments and a
parallel pool of M public-sampled competitor brackets, and report:
  mean_score, std_score, P(top1%), P(top0.1%), median_pool_percentile.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from team_names import canonicalize
from bracket import Bracket, simulate_outcomes, ROUND1_PAIRS, ROUND_NAMES
from public_picks import sample_public_brackets, ROUNDS

# ---- strategies producing {round_name: [team_picked]} dicts ---------------

def _walk(bracket: Bracket, decision_fn) -> dict[str, list[str]]:
    """Walk the bracket, picking via decision_fn(a, b, round_name) at each game.
    Returns {round_name: [team]} where team is picked to advance through that round.
    """
    picks = {rn: [] for rn in ROUND_NAMES}
    f4_winners = []
    for region, teams in bracket.regions.items():
        r1 = []
        for s1, s2 in ROUND1_PAIRS:
            a, b = teams[s1 - 1], teams[s2 - 1]
            w = decision_fn(a, b, "R64"); r1.append(w); picks["R64"].append(w)
        r2 = []
        for i in range(0, 8, 2):
            a, b = r1[i], r1[i + 1]
            w = decision_fn(a, b, "R32"); r2.append(w); picks["R32"].append(w)
        r3 = []
        for i in range(0, 4, 2):
            a, b = r2[i], r2[i + 1]
            w = decision_fn(a, b, "S16"); r3.append(w); picks["S16"].append(w)
        a, b = r3[0], r3[1]
        rw = decision_fn(a, b, "E8"); picks["E8"].append(rw); f4_winners.append(rw)
    region_w = dict(zip(list(bracket.regions.keys()), f4_winners))
    ncg = []
    for r1, r2 in bracket.final_four_pairs:
        a, b = region_w[r1], region_w[r2]
        w = decision_fn(a, b, "F4"); picks["F4"].append(w); ncg.append(w)
    a, b = ncg
    champ = decision_fn(a, b, "NCG"); picks["NCG"].append(champ)
    return picks


def chalk_seed_strategy(bracket: Bracket, seed_lookup: dict[str, int]) -> dict:
    return _walk(bracket, lambda a, b, rn: a if seed_lookup.get(a, 16) < seed_lookup.get(b, 16) else b)


def chalk_model_strategy(bracket: Bracket, win_prob) -> dict:
    return _walk(bracket, lambda a, b, rn: a if win_prob(a, b) >= 0.5 else b)


def ev_max_strategy(bracket: Bracket, reach_probs: pd.DataFrame) -> dict:
    """Pick the team maximizing model P(reach this round)."""
    col_for_round = {"R64": "p_R32", "R32": "p_S16", "S16": "p_E8",
                     "E8": "p_F4", "F4": "p_NCG", "NCG": "p_Champ"}
    def pick(a, b, rn):
        c = col_for_round[rn]
        pa = reach_probs.loc[a, c] if a in reach_probs.index else 0
        pb = reach_probs.loc[b, c] if b in reach_probs.index else 0
        return a if pa >= pb else b
    return _walk(bracket, pick)


def leverage_strategy(bracket: Bracket, reach_probs_model: pd.DataFrame,
                      reach_probs_public: pd.DataFrame, late_rounds_only: bool = True) -> dict:
    """Pick team maximizing (model_reach - public_reach). For early rounds
    (R64, R32) we still prefer chalk-model since flipping early picks
    sacrifices too many low-risk points; leverage kicks in S16+."""
    col_model = {"R64": "p_R32", "R32": "p_S16", "S16": "p_E8",
                 "E8": "p_F4", "F4": "p_NCG", "NCG": "p_Champ"}
    col_public = {"R64": "R32", "R32": "S16", "S16": "E8",
                  "E8": "F4", "F4": "NCG", "NCG": "NCG"}
    chalk_only_rounds = {"R64", "R32"} if late_rounds_only else set()

    def pick(a, b, rn):
        cm, cp = col_model[rn], col_public[rn]
        pa_m = reach_probs_model.loc[a, cm] if a in reach_probs_model.index else 0
        pb_m = reach_probs_model.loc[b, cm] if b in reach_probs_model.index else 0
        if rn in chalk_only_rounds:
            return a if pa_m >= pb_m else b
        pa_p = reach_probs_public.loc[a, cp] if a in reach_probs_public.index else 0
        pb_p = reach_probs_public.loc[b, cp] if b in reach_probs_public.index else 0
        # Score = model_reach * (1 - public_reach), favoring teams the model
        # likes that the public underrates. (Multiplicative form so we still
        # require the team to be a plausible winner.)
        sa = pa_m * (1 - pa_p)
        sb = pb_m * (1 - pb_p)
        return a if sa >= sb else b
    return _walk(bracket, pick)


# ---- scoring against simulated outcomes -----------------------------------

POINTS = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "NCG": 320}
MIN_ADVANCE = {"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "NCG": 6}


def picks_to_indicator(picks: dict[str, list[str]], teams: list[str]) -> np.ndarray:
    """Convert picks to a (6, 64) matrix where row r, col t = 1 if team t was
    picked to advance through round r (i.e. win round r)."""
    team_to_idx = {t: i for i, t in enumerate(teams)}
    M = np.zeros((6, len(teams)), dtype=np.int8)
    for ri, rn in enumerate(["R64", "R32", "S16", "E8", "F4", "NCG"]):
        for t in picks[rn]:
            if t in team_to_idx:
                M[ri, team_to_idx[t]] = 1
    return M


def score_picks_vs_outcomes(picks_M: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Vectorized scoring. picks_M is (6, 64), outcomes is (n_sims, 64) of
    max-round-reached. Returns (n_sims,) array of total points."""
    n_sims = outcomes.shape[0]
    points_arr = np.array([10, 20, 40, 80, 160, 320])
    min_advance_arr = np.array([1, 2, 3, 4, 5, 6])
    # For each round r, correct picks = picks_M[r] AND outcomes >= min_advance[r]
    scores = np.zeros(n_sims, dtype=np.int64)
    for r in range(6):
        correct = (picks_M[r:r+1, :] == 1) & (outcomes >= min_advance_arr[r])
        scores += correct.sum(axis=1) * points_arr[r]
    return scores


def evaluate_strategies(strategies: dict[str, dict[str, list[str]]],
                        outcomes: np.ndarray, teams: list[str],
                        public_outcomes: np.ndarray | None = None) -> pd.DataFrame:
    """Score each strategy against simulated outcomes and report stats.

    outcomes: (n_sims, 64) — your "what could happen" tournament samples
    public_outcomes: optional (n_brackets, 64) — competitor brackets for
                     pool-percentile stats. When None, pool columns are omitted.
    """
    n_sims = outcomes.shape[0]

    competitor_scores = None
    if public_outcomes is not None:
        n_pub = public_outcomes.shape[0]
        print(f"  Scoring {n_pub} competitors x {n_sims} outcomes ...")
        rounds = np.arange(6).reshape(-1, 1, 1) + 1
        pub_picks = (public_outcomes[None, :, :] >= rounds).astype(np.int8)
        competitor_scores = np.zeros((n_sims, n_pub), dtype=np.int64)
        for r in range(6):
            oc = (outcomes >= (r + 1))
            contrib = oc.astype(np.int64) @ pub_picks[r].T.astype(np.int64) * [10, 20, 40, 80, 160, 320][r]
            competitor_scores += contrib

    rows = []
    for name, picks in strategies.items():
        pm = picks_to_indicator(picks, teams)
        my_scores = score_picks_vs_outcomes(pm, outcomes)
        row = {
            "strategy": name,
            "mean_score": float(my_scores.mean()),
            "std_score": float(my_scores.std()),
            "median_score": float(np.median(my_scores)),
            "p10_score": float(np.percentile(my_scores, 10)),
            "p90_score": float(np.percentile(my_scores, 90)),
        }
        if competitor_scores is not None:
            my_vs = (my_scores[:, None] > competitor_scores).mean(axis=1)
            row["median_percentile"] = float(np.median(my_vs))
            row["P_top1pct"] = float((my_vs >= 0.99).mean())
            row["P_top5pct"] = float((my_vs >= 0.95).mean())
            row["P_top10pct"] = float((my_vs >= 0.90).mean())
        rows.append(row)
    return pd.DataFrame(rows)
