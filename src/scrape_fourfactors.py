"""Scrape KenPom Four Factors page for every season we model.

The four factors are eFG%, TO%, OR%, FTRate, computed for both offense
and defense. The proposal calls out Effective Field Goal Percentage
(eFG%) by name. We persist all four factors in case they are useful
later, but the only columns wired into the model are Off-eFG% and
Def-eFG%.

Output: data/raw/fourfactors_YYYY.csv
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import kenpompy.utils as kputils
import kenpompy.summary as kps

from kenpom_credentials import load_kenpom_credentials

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/data/raw"
YEARS = list(range(2010, 2027))


def main():
    os.makedirs(RAW, exist_ok=True)
    email, password = load_kenpom_credentials()
    print("Logging in...", flush=True)
    browser = kputils.login(email, password)
    print("Logged in.", flush=True)

    for year in YEARS:
        out = f"{RAW}/fourfactors_{year}.csv"
        if os.path.exists(out):
            print(f"  [skip] {year}: cached")
            continue
        try:
            df = kps.get_fourfactors(browser, season=str(year))
            df.insert(0, "Season", year)
            df.to_csv(out, index=False)
            print(f"  [ok]   {year}: {len(df)} rows", flush=True)
        except Exception as e:
            print(f"  [FAIL] {year}: {e}", flush=True)
        time.sleep(0.4)


if __name__ == "__main__":
    main()
