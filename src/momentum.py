"""Decay-weighted late-season momentum from per-game KenPom schedules.

For each (season, team) we compute a momentum signal that captures how
the team has been playing heading into March, with extra weight on
conference-tournament games and exponential decay over calendar time.

For each non-NCAA-tournament game we compute:
    raw_em = (PF - PA) / Possessions * 100
    eff_em = raw_em + opp_AdjEM + loc_adj
where loc_adj is +3.5 if Away, -3.5 if Home, 0 otherwise (KenPom's
standard ~3.5 pts/100 home-court advantage). eff_em answers the
question "what AdjEM did this team play at in this game?"

We then apply two multiplicative adjustments before averaging:
  1. Recency decay: w_recency = exp(-(days_back) / tau_days), so games
     played close to selection-Sunday get full weight, games from
     November are heavily discounted.
  2. Conference-tournament boost: w_boost = boost_factor for postseason
     conference-tournament games (Postseason column matches a non-NCAA
     conference name), 1.0 otherwise.

Outputs:
    data/processed/momentum_by_season.csv
       columns: Season, Team, MomentumDecay, MomentumDelta, NGames

MomentumDecay is the weighted-mean effective AdjEM (think of it as the
team's recent form, rated on KenPom's AdjEM scale).
MomentumDelta = MomentumDecay - SeasonAdjEM (trajectory: positive means
the team is playing above their season-long level heading into March).
"""
from __future__ import annotations

import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from team_names import canonicalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/data/raw"
SCHEDULES = f"{RAW}/schedules"
PROC = f"{ROOT}/data/processed"

LOC_ADJ = {"Home": -3.5, "Away": 3.5, "Neutral": 0.0,
           "Semi-Home": -1.5, "Semi-Away": 1.5}
TAU_DAYS = 30.0
CONF_TOURNEY_BOOST = 1.5
NEUTRAL_WEEKDAYS = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}

MONTH_TO_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def parse_date(date_str: str, season: int) -> datetime | None:
    """Parse a 'Mon Nov 6' style string into a datetime.
    Games in Nov-Dec belong to year (season-1); Jan-Apr to year season."""
    if not isinstance(date_str, str):
        return None
    parts = date_str.strip().split()
    if len(parts) < 3:
        return None
    weekday, month, day = parts[0], parts[1], parts[2]
    if month not in MONTH_TO_NUM:
        return None
    m = MONTH_TO_NUM[month]
    yr = season - 1 if m >= 11 else season
    try:
        return datetime(year=yr, month=m, day=int(day))
    except (ValueError, TypeError):
        return None


def parse_result(result: str) -> tuple[int, int, int] | None:
    """Parse 'W, 84-31' or 'L, 70-72' into (pf, pa, win)."""
    if not isinstance(result, str):
        return None
    m = re.match(r"\s*([WL])\s*,\s*(\d+)\s*-\s*(\d+)", result)
    if not m:
        return None
    outcome, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    win = 1 if outcome == "W" else 0
    return a, b, win


def is_conf_tourney(postseason: str) -> bool:
    """A postseason value is a conference tournament if it's a non-empty,
    non-'NCAA', non-'NIT'/etc. string identifying a conference."""
    if not isinstance(postseason, str):
        return False
    s = postseason.strip()
    if not s:
        return False
    s_lower = s.lower()
    excluded = {"ncaa", "nit", "cbi", "cit", "vegas16", "tbc"}
    return s_lower not in excluded


def load_kenpom_lookup() -> dict[tuple[int, str], float]:
    """(season, canonical_team) -> season-end AdjEM."""
    out = {}
    for y in range(2010, 2027):
        path = f"{RAW}/kenpom_{y}.csv"
        if not os.path.exists(path):
            continue
        kp = pd.read_csv(path)
        kp["TeamCanon"] = kp["Team"].astype(str).str.strip().apply(canonicalize)
        for _, r in kp.iterrows():
            try:
                out[(y, r["TeamCanon"])] = float(r["AdjEM"])
            except (ValueError, TypeError):
                continue
    return out


def selection_sunday(season: int) -> datetime:
    """Approximate selection Sunday as the second Sunday of March in `season`.
    For decay weighting we just need a stable late-regular-season anchor."""
    d = datetime(year=season, month=3, day=1)
    sundays = []
    for i in range(31):
        try:
            cur = datetime(year=season, month=3, day=1 + i)
        except ValueError:
            break
        if cur.weekday() == 6:
            sundays.append(cur)
    return sundays[1] if len(sundays) >= 2 else sundays[0]


def build_team_momentum(sched: pd.DataFrame, kp_lookup: dict, season: int) -> dict:
    """Compute MomentumDecay and MomentumDelta for one team-season."""
    sched = sched.copy()
    sched["date"] = sched["Date"].apply(lambda s: parse_date(s, season))
    sched = sched.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Drop NCAA tournament games to avoid label leakage.
    is_ncaa = sched["Postseason"].astype(str).str.strip().str.upper().eq("NCAA")
    sched = sched[~is_ncaa].reset_index(drop=True)

    if len(sched) == 0:
        return {"MomentumDecay": np.nan, "MomentumDelta": np.nan, "NGames": 0}

    anchor = selection_sunday(season)

    rows = []
    for _, g in sched.iterrows():
        parsed = parse_result(g["Result"])
        if parsed is None:
            continue
        pf, pa, _win = parsed
        try:
            poss = float(g["Possession Number"])
        except (TypeError, ValueError):
            continue
        if poss <= 0:
            continue
        raw_em = (pf - pa) / poss * 100.0

        opp_canon = canonicalize(str(g["Opponent Name"]))
        opp_em = kp_lookup.get((season, opp_canon))
        if opp_em is None:
            continue

        loc = str(g["Location"]).strip()
        loc_adj = LOC_ADJ.get(loc, 0.0)
        eff_em = raw_em + opp_em + loc_adj

        days_back = max(0.0, (anchor - g["date"]).days)
        w_decay = float(np.exp(-days_back / TAU_DAYS))
        w_boost = CONF_TOURNEY_BOOST if is_conf_tourney(g["Postseason"]) else 1.0
        w = w_decay * w_boost
        rows.append((eff_em, w))

    if not rows:
        return {"MomentumDecay": np.nan, "MomentumDelta": np.nan, "NGames": 0}

    em_arr = np.array([r[0] for r in rows])
    w_arr = np.array([r[1] for r in rows])
    if w_arr.sum() == 0:
        return {"MomentumDecay": np.nan, "MomentumDelta": np.nan, "NGames": len(rows)}
    momentum = float(np.average(em_arr, weights=w_arr))
    return {"MomentumDecay": momentum, "NGames": len(rows)}


def main():
    files = sorted(glob.glob(f"{SCHEDULES}/*.csv"))
    print(f"Schedule files: {len(files)}")
    if not files:
        raise SystemExit("No schedule files found in data/raw/schedules/. Run scrape_schedules.py first.")

    kp_lookup = load_kenpom_lookup()
    print(f"KenPom AdjEM lookup entries: {len(kp_lookup)}")

    rows = []
    skipped = 0
    for f in files:
        df = pd.read_csv(f)
        if len(df) == 0:
            skipped += 1
            continue
        season = int(df["Season"].iloc[0])
        team = str(df["TeamCanonical"].iloc[0])
        info = build_team_momentum(df, kp_lookup, season)
        season_em = kp_lookup.get((season, team), np.nan)
        info["MomentumDelta"] = (info["MomentumDecay"] - season_em
                                 if pd.notna(info["MomentumDecay"]) and pd.notna(season_em)
                                 else np.nan)
        rows.append({"Season": season, "Team": team, "SeasonAdjEM": season_em, **info})

    out = pd.DataFrame(rows).sort_values(["Season", "Team"]).reset_index(drop=True)
    print(f"Computed momentum for {len(out)} team-seasons (skipped {skipped} empty files)")
    print(out.head(10).to_string(index=False))
    print()
    print("MomentumDecay summary:")
    print(out["MomentumDecay"].describe().round(2).to_string())
    print()
    print("MomentumDelta summary (eff_em - season_AdjEM):")
    print(out["MomentumDelta"].describe().round(2).to_string())

    out_path = f"{PROC}/momentum_by_season.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
