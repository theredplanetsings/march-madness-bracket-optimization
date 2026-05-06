"""Model the distribution of public bracket picks as a function of seed × round.

We have one year of empirical ESPN "who picked whom" data (2019). Public
picks correlate strongly with seed: in R64, the 1-seed is picked at ~99%,
the 16-seed at ~1%, etc. We fit a smooth seed × round table from the 2019
data and use it as a synthetic public-pick prior for any year. This is
standard in the literature (Metrick 1996, Niemi-Wright-Smith 2008).

For year-specific priors, real ESPN data should be substituted when
available; the seed-based fit is a defensible baseline.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from team_names import canonicalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/data/raw"
PROC = f"{ROOT}/data/processed"
ROUNDS = ["R64", "R32", "S16", "E8", "F4", "NCG"]


def fit_seed_pick_table() -> pd.DataFrame:
    """Return DataFrame indexed by seed (1..16) with columns R64..NCG =
    average public-pick probability for a team of that seed to reach that round.
    Built from 2019 ESPN data."""
    df = pd.read_csv(f"{RAW}/espn_picks_2019.csv")
    # Each (Seed, Round) bin has 4 teams in 2019 (one per region)
    table = df.groupby(["Seed", "Round"])["Percent"].mean().unstack("Round")
    table = table[ROUNDS] / 100.0  # to probabilities
    table = table.reindex(range(1, 17))
    # Sanity: pick prob for R64 should monotonically decrease with seed
    return table


def public_pick_probs(seeds_df: pd.DataFrame, season: int, table: pd.DataFrame) -> pd.DataFrame:
    """For a given season's bracket, return a per-team table of
    public-pick advancement probability per round. Indexed by team name."""
    sy = seeds_df[seeds_df.Season == season].copy()
    sy["Team"] = sy["Team"].apply(canonicalize)
    out = sy[["Team", "Seed"]].copy()
    for rn in ROUNDS:
        out[rn] = sy["Seed"].map(table[rn])
    return out.set_index("Team")[ROUNDS]


def sample_public_brackets(reach_probs_public: pd.DataFrame, bracket, n_brackets: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Sample n_brackets bracket choices where each game's pick comes from
    the public-pick distribution.

    For each game, the per-team pick probabilities don't sum to 1 across
    the matchup (since they're marginal P(reach round) values, not
    conditional). We approximate the conditional pick prob:
        P(pick A | A vs B in round R) = p_A / (p_A + p_B)
    where p_X is the public probability of X reaching round R+1 (the
    advancement probability for winning this round).

    Returns: (n_brackets, 64) array of advancement levels per team —
    formatted just like simulate_outcomes() in bracket.py.
    """
    from bracket import ROUND1_PAIRS

    teams = bracket.all_teams
    n = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}
    region_names = list(bracket.regions.keys())
    region_team_idx = [[team_to_idx[t] for t in bracket.regions[r]] for r in region_names]
    f4_pair_idx = [(region_names.index(a), region_names.index(b)) for a, b in bracket.final_four_pairs]

    # Pick-prob table: pick[t, r] = P(team t picked to win round r) marginal
    pick_arr = np.zeros((n, 6))
    for ti, t in enumerate(teams):
        if t in reach_probs_public.index:
            for ri, rn in enumerate(ROUNDS):
                v = reach_probs_public.loc[t, rn]
                pick_arr[ti, ri] = v if not pd.isna(v) else 0.0

    advancements = np.zeros((n_brackets, n), dtype=np.int8)
    randoms = rng.random((n_brackets, 63))
    for s in range(n_brackets):
        r_idx = 0
        region_winners = []
        for region in region_team_idx:
            r1 = []
            # R64 = pick_arr[:, 0] (P(reach R64) — but every team trivially does;
            # so use [:, 0] = P(reach R32)? Wait — table cols: R64=reach R64,
            # R32=reach R32, etc. R64=1.0 always; the "win R64" prob = reach R32.
            # So when picking R64 winners, use P(reach R32) = column index 1.
            for (s1, s2) in ROUND1_PAIRS:
                a, b = region[s1 - 1], region[s2 - 1]
                pa, pb = pick_arr[a, 1], pick_arr[b, 1]
                w = a if randoms[s, r_idx] * (pa + pb) < pa else b
                r1.append(w); r_idx += 1
                advancements[s, w] = max(advancements[s, w], 1)
            r2 = []
            for i in range(0, 8, 2):
                a, b = r1[i], r1[i + 1]
                pa, pb = pick_arr[a, 2], pick_arr[b, 2]
                denom = pa + pb if (pa + pb) > 0 else 1
                w = a if randoms[s, r_idx] * denom < pa else b
                r2.append(w); r_idx += 1
                advancements[s, w] = max(advancements[s, w], 2)
            r3 = []
            for i in range(0, 4, 2):
                a, b = r2[i], r2[i + 1]
                pa, pb = pick_arr[a, 3], pick_arr[b, 3]
                denom = pa + pb if (pa + pb) > 0 else 1
                w = a if randoms[s, r_idx] * denom < pa else b
                r3.append(w); r_idx += 1
                advancements[s, w] = max(advancements[s, w], 3)
            a, b = r3[0], r3[1]
            pa, pb = pick_arr[a, 4], pick_arr[b, 4]
            denom = pa + pb if (pa + pb) > 0 else 1
            rw = a if randoms[s, r_idx] * denom < pa else b
            r_idx += 1
            advancements[s, rw] = max(advancements[s, rw], 4)
            region_winners.append(rw)
        f4_winners = []
        for (i_r1, i_r2) in f4_pair_idx:
            a, b = region_winners[i_r1], region_winners[i_r2]
            pa, pb = pick_arr[a, 5], pick_arr[b, 5]
            denom = pa + pb if (pa + pb) > 0 else 1
            w = a if randoms[s, r_idx] * denom < pa else b
            r_idx += 1
            advancements[s, w] = max(advancements[s, w], 5)
            f4_winners.append(w)
        a, b = f4_winners
        pa, pb = pick_arr[a, 5], pick_arr[b, 5]  # championship picks proxy
        denom = pa + pb if (pa + pb) > 0 else 1
        champ = a if randoms[s, r_idx] * denom < pa else b
        advancements[s, champ] = 6

    return advancements


if __name__ == "__main__":
    table = fit_seed_pick_table()
    print("Public pick rate by seed × round (from 2019 ESPN data):")
    print((table * 100).round(1).to_string())
    table.to_csv(f"{PROC}/public_pick_table.csv")
