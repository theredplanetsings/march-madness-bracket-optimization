"""Scrape per-game KenPom schedules for every tournament team-season.

We need game-level results (date, opponent, score, possessions, location,
postseason flag) to build a decay-weighted late-season momentum feature
that goes beyond the existing AdjEM x Luck end-of-season proxy.

Output: data/raw/schedules/{season}_{team_slug}.csv

We only fetch tournament teams (those appearing in tourney_seeds.csv).
KenPom's get_schedule() takes the team's KenPom-formatted name; we look
that up by inverting canonicalize() against each season's KenPom roster.
"""
from __future__ import annotations

import os
import re
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import kenpompy.utils as kputils
import kenpompy.team as kpt

from team_names import canonicalize
from kenpom_credentials import load_kenpom_credentials

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/data/raw"
OUT = f"{RAW}/schedules"
os.makedirs(OUT, exist_ok=True)


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def build_kenpom_name_lookup() -> dict[tuple[int, str], str]:
    """For each (season, canonical_team) -> the raw KenPom team name."""
    out = {}
    for y in range(2010, 2027):
        path = f"{RAW}/kenpom_{y}.csv"
        if not os.path.exists(path):
            continue
        kp = pd.read_csv(path)
        for raw_name in kp["Team"].astype(str).str.strip():
            canon = canonicalize(raw_name)
            out[(y, canon)] = raw_name
    return out


def main():
    seeds = pd.read_csv(f"{ROOT}/data/processed/tourney_seeds.csv")
    seeds["Team"] = seeds["Team"].apply(canonicalize)
    targets = list(seeds[["Season", "Team"]].itertuples(index=False, name=None))
    print(f"Total tournament team-seasons: {len(targets)}")

    kp_lookup = build_kenpom_name_lookup()
    missing = [t for t in targets if t not in kp_lookup]
    print(f"Missing from KenPom roster: {len(missing)}")
    if missing[:5]:
        print("  examples:", missing[:5])

    email, password = load_kenpom_credentials()
    print("Logging in to kenpom.com ...", flush=True)
    browser = kputils.login(email, password)
    print("Logged in.", flush=True)

    n_ok = n_skip = n_fail = 0
    for i, (season, canon_team) in enumerate(targets, 1):
        kp_name = kp_lookup.get((season, canon_team))
        if kp_name is None:
            n_fail += 1
            continue
        out_path = f"{OUT}/{season}_{slugify(canon_team)}.csv"
        if os.path.exists(out_path):
            n_skip += 1
            continue
        try:
            sched = kpt.get_schedule(browser, team=kp_name, season=str(season))
            sched.insert(0, "Season", season)
            sched.insert(1, "TeamCanonical", canon_team)
            sched.insert(2, "TeamKP", kp_name)
            sched.to_csv(out_path, index=False)
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"  [{i}/{len(targets)}] {n_ok} fetched, {n_skip} cached, {n_fail} failed", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"  [FAIL] {season} {canon_team} ({kp_name}): {e}", flush=True)
        time.sleep(0.4)

    print(f"\nDone: {n_ok} fetched, {n_skip} cached, {n_fail} failed")


if __name__ == "__main__":
    main()
