"""
Task 1: Scrape KenPom historical ratings 2010-2025.

Uses the kenpompy package which handles login (the site uses CloudFlare,
so a plain requests.Session() POST cannot bypass JS challenges - kenpompy
uses cloudscraper under the hood).

Output: data/raw/kenpom_YYYY.csv with columns:
    Rk, Team, Conf, W-L, AdjEM, AdjO, AdjO.Rank, AdjD, AdjD.Rank,
    AdjT, AdjT.Rank, Luck, Luck.Rank, SOS-AdjEM, SOS-AdjEM.Rank,
    SOS-OppO, SOS-OppO.Rank, SOS-OppD, SOS-OppD.Rank,
    NCSOS-AdjEM, NCSOS-AdjEM.Rank, Seed
"""
import os
import sys
import time
import pandas as pd

import kenpompy.utils as kputils
from kenpompy.misc import get_pomeroy_ratings

from kenpom_credentials import load_kenpom_credentials

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

# 2020 had no NCAA tournament (COVID) but KenPom still published season ratings.
YEARS = list(range(2010, 2027))  # 2010..2026 inclusive (2026 needed for prediction target)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    email, password = load_kenpom_credentials()
    print("Logging into kenpom.com ...", flush=True)
    browser = kputils.login(email, password)
    print("Login successful.", flush=True)

    summary = []
    for year in YEARS:
        out_path = os.path.join(OUT_DIR, f"kenpom_{year}.csv")
        if os.path.exists(out_path):
            df = pd.read_csv(out_path)
            print(f"[skip] {year}: already have {len(df)} rows at {out_path}")
            summary.append((year, len(df), "cached"))
            continue
        try:
            df = get_pomeroy_ratings(browser, season=str(year))
            df.insert(0, "Season", year)
            df.to_csv(out_path, index=False)
            print(f"[ok]   {year}: {len(df)} rows -> {out_path}", flush=True)
            summary.append((year, len(df), "ok"))
        except Exception as e:
            print(f"[FAIL] {year}: {e}", flush=True)
            summary.append((year, 0, f"fail: {e}"))
        time.sleep(1.0)  # be polite

    print("\n=== SUMMARY ===")
    for y, n, s in summary:
        print(f"  {y}: {n:>4} rows  ({s})")

    # Verify by printing top 5 rows for 2024
    p2024 = os.path.join(OUT_DIR, "kenpom_2024.csv")
    if os.path.exists(p2024):
        d = pd.read_csv(p2024)
        cols = [c for c in ["Rk", "Team", "Conf", "W-L", "AdjEM", "AdjO", "AdjD", "AdjT", "Luck", "SOS-AdjEM", "Seed"] if c in d.columns]
        print("\n=== 2024 top 5 ===")
        print(d[cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
