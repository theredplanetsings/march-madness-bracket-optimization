"""Backtest and bracket-strategy comparison.

Pipeline per season:
  1. Reconstruct that season's bracket from data/processed/tourney_seeds.csv
  2. Train win-prob model on all OTHER seasons (proper LOSO)
  3. Run K Monte Carlo simulations of the bracket using the model
  4. Construct strategies (sets of picks) and compute scoring stats:
        - chalk_seed: always pick lower seed
        - chalk_model: at each game pick higher-model-prob team
        - leverage: like chalk_model but flip late-round picks toward
          high-EV-spread teams when ESPN public picks are available
  5. Score each strategy against the ACTUAL tournament outcome
  6. Aggregate.

We also report Monte-Carlo expected score (over simulations) so we can
contrast realized score (one outcome) with the model's prior belief.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from team_names import canonicalize
from bracket import Bracket, simulate_many, ROUND1_PAIRS, ESPN_POINTS, ROUND_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = f"{ROOT}/data/processed"

KENPOM_FEATS = ["AdjEM", "AdjO", "AdjD", "AdjT", "Luck", "SOS-AdjEM", "NCSOS-AdjEM",
                "Momentum", "OffEFG", "DefEFG"]
DELTA_COLS = [f"d_{f}" for f in KENPOM_FEATS] + ["d_Seed"]


# ----- bracket construction --------------------------------------------------

# Standard region-pair convention. The actual NCAA pairings vary year to year;
# we use the alphabetic convention as a reasonable default — F4 outcomes
# depend on the model anyway since both region winners enter symmetrically.
REGION_ORDER_FALLBACK = ["east", "midwest", "south", "west"]


def build_bracket(season: int, seeds_df: pd.DataFrame) -> Bracket:
    """Construct a Bracket object from the seeds dataframe for a season."""
    s = seeds_df[seeds_df.Season == season].copy()
    regions_in_data = sorted(s.Region.unique())
    regions = {}
    for r in regions_in_data:
        teams_by_seed = s[s.Region == r].sort_values("Seed")
        # If multiple teams share a seed (First Four), keep the first.
        teams = []
        for seed in range(1, 17):
            row = teams_by_seed[teams_by_seed.Seed == seed]
            if len(row) == 0:
                # Missing — should not happen with our filtered data
                teams.append(None)
            else:
                teams.append(row.iloc[0]["Team"])
        regions[r] = teams
    # Final Four pairing: pair the regions by alphabetical order — since the
    # model is symmetric in region this affects the conditional question
    # "who do we expect in the championship" but not the per-team reach probs.
    rs = list(regions.keys())
    f4_pairs = [(rs[0], rs[1]), (rs[2], rs[3])] if len(rs) == 4 else None
    return Bracket(regions=regions, final_four_pairs=f4_pairs)


# ----- model training (LOSO) -------------------------------------------------

def train_model_excluding(matchups: pd.DataFrame, hold_out_season: int):
    """Train the full-feature logistic model on all seasons except `hold_out_season`."""
    train = matchups[matchups.Season != hold_out_season]
    X = train[DELTA_COLS].values
    y = train["label"].values
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(X), y)
    return scaler, model


def build_win_prob(scaler, model, kenpom_year: pd.DataFrame, seeds_year: pd.DataFrame):
    """Return a callable win_prob(team_a, team_b) using the trained model."""
    kp = kenpom_year.set_index("Team")
    seed_lookup = seeds_year.set_index("Team")["Seed"].to_dict()

    def win_prob(a: str, b: str) -> float:
        ra, rb = kp.loc[a], kp.loc[b]
        deltas = [ra[f] - rb[f] for f in KENPOM_FEATS]
        sa, sb = seed_lookup.get(a, np.nan), seed_lookup.get(b, np.nan)
        deltas.append((sa - sb) if (sa == sa and sb == sb) else 0.0)
        x = np.array(deltas).reshape(1, -1)
        return float(model.predict_proba(scaler.transform(x))[0, 1])

    return win_prob


# ----- strategies -----------------------------------------------------------

def chalk_seed_picks(bracket: Bracket) -> dict[str, str]:
    """Pick the lower-seeded team at every game.
    Returns {slot_id: team_name}, one entry per game in the 63-game bracket.
    But for scoring we need only the set of teams advanced through each round.
    Return {round_name: list[team]} format."""
    return _picks_from_simulation(bracket, lambda a, b: 1.0 if _seed_lookup_global.get(a, 16) < _seed_lookup_global.get(b, 16) else 0.0)


def chalk_model_picks(bracket: Bracket, win_prob) -> dict[str, list[str]]:
    return _picks_from_simulation(bracket, win_prob, deterministic=True)


def _picks_from_simulation(bracket: Bracket, win_prob, deterministic: bool = True) -> dict[str, list[str]]:
    """Walk the bracket making the deterministic max-prob pick at each game."""
    picks = {rn: [] for rn in ROUND_NAMES}
    f4_winners = []
    for region, teams in bracket.regions.items():
        # Round 1
        r1 = []
        for s1, s2 in ROUND1_PAIRS:
            a, b = teams[s1 - 1], teams[s2 - 1]
            w = a if win_prob(a, b) >= 0.5 else b
            r1.append(w); picks["R64"].append(w)  # picked to advance past R64
        # Round 2 (S16 advance picks)
        r2 = []
        for i in range(0, 8, 2):
            a, b = r1[i], r1[i + 1]
            w = a if win_prob(a, b) >= 0.5 else b
            r2.append(w); picks["R32"].append(w)
        # Sweet 16 (E8 picks)
        r3 = []
        for i in range(0, 4, 2):
            a, b = r2[i], r2[i + 1]
            w = a if win_prob(a, b) >= 0.5 else b
            r3.append(w); picks["S16"].append(w)
        # Elite 8 (F4 picks)
        a, b = r3[0], r3[1]
        rw = a if win_prob(a, b) >= 0.5 else b
        picks["E8"].append(rw)
        f4_winners.append(rw)
    # Final Four (championship-game picks)
    if bracket.final_four_pairs:
        ncg = []
        # Map region->winner
        region_w = dict(zip(list(bracket.regions.keys()), f4_winners))
        for r1, r2 in bracket.final_four_pairs:
            a, b = region_w[r1], region_w[r2]
            w = a if win_prob(a, b) >= 0.5 else b
            ncg.append(w); picks["F4"].append(w)
        # Champion
        a, b = ncg
        champ = a if win_prob(a, b) >= 0.5 else b
        picks["NCG"].append(champ)
    return picks


# Will be set externally per-season for the seed-chalk strategy
_seed_lookup_global: dict[str, int] = {}


# ----- scoring --------------------------------------------------------------

def actual_advancement(season: int, results_df: pd.DataFrame) -> dict[str, int]:
    """Build a {team: max_round_index_reached} dict for actual outcomes.
    Round indices: 0=R64, 1=R32, 2=S16, 3=E8, 4=F4, 5=NCG, 6=Champion."""
    res = results_df[results_df.Season == season]
    round_to_idx = {"First Round": 0, "Second Round": 1, "Sweet 16": 2,
                    "Elite 8": 3, "Final Four": 4, "Championship": 5}
    adv: dict[str, int] = {}
    for _, g in res.iterrows():
        rn_idx = round_to_idx[g["Round"]]
        # Winner advances to (rn_idx + 1); loser ends at rn_idx (still "made" rn_idx)
        w, l = g["WTeam"], g["LTeam"]
        adv[w] = max(adv.get(w, 0), rn_idx + 1)
        adv[l] = max(adv.get(l, 0), rn_idx)
    return adv


def score_picks(picks: dict[str, list[str]], actual: dict[str, int]) -> tuple[int, dict[str, int]]:
    """Score a {round_name: list[team]} picks dict against actual advancement.
    Returns (total_points, per_round_points)."""
    # round_name = team picked to advance THROUGH this round (i.e. win game in that round).
    # So if picks['R64']=[A,B,...], for each team T in that list we award 10 points if actual[T] >= 1.
    points_for_round_pick = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "NCG": 320}
    min_round_advanced_for_correct = {"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "NCG": 6}
    total = 0
    per_round = {}
    for rn, teams in picks.items():
        pr = 0
        for t in teams:
            if actual.get(t, 0) >= min_round_advanced_for_correct[rn]:
                pr += points_for_round_pick[rn]
        per_round[rn] = pr
        total += pr
    return total, per_round


def expected_score(picks: dict[str, list[str]], reach_probs: pd.DataFrame) -> float:
    """Model-expected score = sum over rounds of points * sum_of_team_reach_probs."""
    points_for_round_pick = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "NCG": 320}
    round_to_col = {"R64": "p_R32", "R32": "p_S16", "S16": "p_E8",
                    "E8": "p_F4", "F4": "p_NCG", "NCG": "p_Champ"}
    total = 0.0
    for rn, teams in picks.items():
        col = round_to_col[rn]
        for t in teams:
            if t in reach_probs.index:
                total += points_for_round_pick[rn] * reach_probs.loc[t, col]
    return total


# ----- main backtest --------------------------------------------------------

def main():
    matchups = pd.read_csv(f"{PROC}/matchups.csv")
    seeds = pd.read_csv(f"{PROC}/tourney_seeds.csv")
    kp = pd.read_csv(f"{PROC}/kenpom_all.csv")
    results = pd.read_csv(f"{PROC}/tourney_results.csv")

    # Prediction model: full logistic regression (KenPom deltas + seed + momentum).
    # Chosen as the best LOSO model by log-loss (0.428) vs XGBoost (0.453) and RF (0.467).
    # 2020 was cancelled (COVID); 2021 has the Oregon/VCU forfeit so the
    # bracket is missing one seed slot — we drop both for clean backtesting.
    seasons = [s for s in sorted(seeds.Season.unique()) if s not in (2020, 2021)]
    all_rows = []
    for season in seasons:
        seeds_year = seeds[seeds.Season == season]
        kp_year = kp[kp.Season == season]

        bracket = build_bracket(season, seeds_year)
        # Update global seed lookup for chalk_seed
        global _seed_lookup_global
        _seed_lookup_global = seeds_year.set_index("Team")["Seed"].to_dict()

        # Train on all OTHER seasons
        scaler, model = train_model_excluding(matchups, season)
        win_prob = build_win_prob(scaler, model, kp_year, seeds_year)

        # Per-team reach probabilities under the model
        reach = simulate_many(bracket, win_prob, n_sims=5000, seed=season)

        # Strategies
        seed_picks = chalk_seed_picks(bracket)
        model_picks = chalk_model_picks(bracket, win_prob)

        # Score
        actual = actual_advancement(season, results)
        seed_score, _ = score_picks(seed_picks, actual)
        model_score, _ = score_picks(model_picks, actual)

        # Model-expected scores
        seed_exp = expected_score(seed_picks, reach)
        model_exp = expected_score(model_picks, reach)

        all_rows.append({
            "season": season,
            "actual_seed_score": seed_score,
            "actual_model_score": model_score,
            "expected_seed_score": seed_exp,
            "expected_model_score": model_exp,
            "diff_actual": model_score - seed_score,
            "diff_expected": model_exp - seed_exp,
        })
        print(f"{season}: chalk-seed={seed_score:>4}, chalk-model={model_score:>4}, "
              f"E[chalk-seed]={seed_exp:6.1f}, E[chalk-model]={model_exp:6.1f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{PROC}/backtest_results.csv", index=False)

    print("\n=== Summary across seasons ===")
    print(f"Mean chalk-seed score:  {df.actual_seed_score.mean():6.1f}")
    print(f"Mean chalk-model score: {df.actual_model_score.mean():6.1f}")
    print(f"Mean E[chalk-seed]:     {df.expected_seed_score.mean():6.1f}")
    print(f"Mean E[chalk-model]:    {df.expected_model_score.mean():6.1f}")
    print(f"Years model > seed (actual): {int((df.diff_actual > 0).sum())} / {len(df)}")


if __name__ == "__main__":
    main()
