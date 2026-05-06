"""Build the matchup feature matrix.

For every historical NCAA tournament game we have a winner and loser.
We construct two rows per game (winner-favorite and loser-favorite frames)
to avoid label imbalance, with features = team A KenPom stats minus team B
KenPom stats, plus seed differential. This is a standard symmetric framing
for tournament prediction.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from team_names import canonicalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/data/raw"
PROC = f"{ROOT}/data/processed"
os.makedirs(PROC, exist_ok=True)

KENPOM_FEATS = ["AdjEM", "AdjO", "AdjD", "AdjT", "Luck", "SOS-AdjEM", "NCSOS-AdjEM",
                "Momentum", "OffEFG", "DefEFG"]
MOMENTUM_FILE = f"{PROC}/momentum_by_season.csv"


def add_momentum_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback end-of-season AdjEM x Luck proxy, used only when the
    decay-weighted momentum feature has not been built yet."""
    adjem = pd.to_numeric(df["AdjEM"], errors="coerce")
    luck = pd.to_numeric(df["Luck"], errors="coerce")
    df["Momentum"] = adjem * luck
    return df


def attach_decay_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the `Momentum` column with the decay-weighted late-season
    momentum signal computed from per-game KenPom schedules in
    `data/processed/momentum_by_season.csv`. Falls back to the AdjEM x
    Luck proxy for any (Season, Team) missing from the game-log file
    (e.g. if scraping has not yet completed for some teams).
    """
    if not os.path.exists(MOMENTUM_FILE):
        print(f"  [momentum] {MOMENTUM_FILE} not found - using AdjEM x Luck proxy")
        return add_momentum_proxy(df)

    mom = pd.read_csv(MOMENTUM_FILE)
    mom["Team"] = mom["Team"].apply(canonicalize)
    # Use MomentumDelta (decay-weighted form minus season AdjEM) so the
    # feature carries trajectory information orthogonal to d_AdjEM.
    mom_lookup = mom.set_index(["Season", "Team"])["MomentumDelta"].to_dict()

    # Start with the proxy as a fallback then overwrite where we have data.
    df = add_momentum_proxy(df)
    df["_canon_team"] = df["Team"]  # already canonicalized in load_kenpom_all
    keys = list(zip(df["Season"], df["_canon_team"]))
    decay_vals = [mom_lookup.get(k) for k in keys]
    have = sum(v is not None and not pd.isna(v) for v in decay_vals)
    df["Momentum"] = [v if (v is not None and not pd.isna(v)) else proxy
                      for v, proxy in zip(decay_vals, df["Momentum"].values)]
    df = df.drop(columns=["_canon_team"])
    print(f"  [momentum] decay-weighted momentum applied to {have}/{len(df)} team-seasons")
    return df


def load_fourfactors() -> pd.DataFrame:
    """Concat all KenPom Four Factors per-season files into one frame
    keyed by (Season, Team_canonical) with columns OffEFG and DefEFG.

    The proposal called out Effective Field Goal Percentage by name; we
    add only the eFG% pair here. The remaining four-factors (TO%, OR%,
    FTRate) are persisted on disk in case we want to extend later.
    """
    frames = []
    for y in range(2010, 2027):
        path = f"{RAW}/fourfactors_{y}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["Team"] = df["Team"].astype(str).str.strip().apply(canonicalize)
        df = df[["Season", "Team", "Off-eFG%", "Def-eFG%"]].rename(
            columns={"Off-eFG%": "OffEFG", "Def-eFG%": "DefEFG"})
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Season", "Team", "OffEFG", "DefEFG"])
    return pd.concat(frames, ignore_index=True)


def attach_fourfactors(kp: pd.DataFrame) -> pd.DataFrame:
    """Left-join the eFG% columns into the season-end KenPom frame."""
    ff = load_fourfactors()
    if len(ff) == 0:
        print("  [fourfactors] no fourfactors_*.csv files - filling with NaN")
        kp["OffEFG"] = float("nan")
        kp["DefEFG"] = float("nan")
        return kp
    merged = kp.merge(ff, on=["Season", "Team"], how="left")
    n_have = merged[["OffEFG", "DefEFG"]].notna().all(axis=1).sum()
    print(f"  [fourfactors] eFG% joined for {n_have}/{len(merged)} team-seasons")
    return merged


def load_kenpom_all() -> pd.DataFrame:
    frames = []
    for y in range(2010, 2027):
        df = pd.read_csv(f"{RAW}/kenpom_{y}.csv")
        df["Team"] = df["Team"].str.strip().apply(canonicalize)
        frames.append(df)
    kp = pd.concat(frames, ignore_index=True)
    kp = attach_decay_momentum(kp)
    kp = attach_fourfactors(kp)
    return kp


def load_tournament() -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = pd.read_csv(f"{RAW}/tourney_seeds.csv")
    seeds["Team"] = seeds["Team"].apply(canonicalize)
    res = pd.read_csv(f"{RAW}/tourney_results.csv")
    res["WTeam"] = res["WTeam"].apply(canonicalize)
    res["LTeam"] = res["LTeam"].apply(canonicalize)
    return seeds, res


def build_matchups(kp: pd.DataFrame, results: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (Season, TeamA, TeamB) ordered such that A=winner.
    Then we mirror to avoid label leakage: balanced 50/50 dataset of
    (TeamA - TeamB feature deltas, label = 1 if A won)."""
    # Quick lookups
    kp_idx = kp.set_index(["Season", "Team"])

    seed_lookup = seeds.set_index(["Season", "Team"])["Seed"].to_dict()

    rows = []
    for _, g in results.iterrows():
        yr, w, l = g["Season"], g["WTeam"], g["LTeam"]
        try:
            wf = kp_idx.loc[(yr, w)]
            lf = kp_idx.loc[(yr, l)]
        except KeyError:
            continue
        ws = seed_lookup.get((yr, w), np.nan)
        ls = seed_lookup.get((yr, l), np.nan)

        base = {"Season": yr, "Round": g["Round"], "TeamA": w, "TeamB": l,
                "SeedA": ws, "SeedB": ls, "label": 1}
        for f in KENPOM_FEATS:
            base[f"{f}_A"] = wf[f]
            base[f"{f}_B"] = lf[f]
            base[f"d_{f}"] = wf[f] - lf[f]
        base["d_Seed"] = (ws - ls) if (ws == ws and ls == ls) else np.nan
        rows.append(base)

        # Mirror: flip A/B, label=0
        mir = {"Season": yr, "Round": g["Round"], "TeamA": l, "TeamB": w,
               "SeedA": ls, "SeedB": ws, "label": 0}
        for f in KENPOM_FEATS:
            mir[f"{f}_A"] = lf[f]
            mir[f"{f}_B"] = wf[f]
            mir[f"d_{f}"] = lf[f] - wf[f]
        mir["d_Seed"] = (ls - ws) if (ws == ws and ls == ls) else np.nan
        rows.append(mir)
    return pd.DataFrame(rows)


def main():
    kp = load_kenpom_all()
    seeds, results = load_tournament()
    print(f"KenPom rows: {len(kp)}, tourney games: {len(results)}, seed rows: {len(seeds)}")
    # Persist canonicalized KenPom (needed elsewhere)
    kp.to_csv(f"{PROC}/kenpom_all.csv", index=False)
    seeds.to_csv(f"{PROC}/tourney_seeds.csv", index=False)
    results.to_csv(f"{PROC}/tourney_results.csv", index=False)

    matchups = build_matchups(kp, results, seeds)
    print(f"Matchup rows (with mirrors): {len(matchups)}")
    print(f"Label balance: {matchups['label'].mean():.3f}")
    print(f"NaN by column:\n{matchups.isna().sum().loc[lambda s: s>0]}")
    matchups.to_csv(f"{PROC}/matchups.csv", index=False)

    # Quick sanity: AdjEM diff should correlate with winning
    by_round = matchups[matchups.label == 1].groupby("Round")["d_AdjEM"].agg(["mean", "count"])
    print("\nWinner's AdjEM advantage over loser, by round:")
    print(by_round)


if __name__ == "__main__":
    main()
