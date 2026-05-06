"""
Task 2b: Scrape team schedules from sports-reference to build momentum.

We only scrape NCAA tournament teams (from tourney_seeds.csv) to reduce load.
Output:
  data/processed/momentum_game_log.csv
with columns:
  Season, Team, Momentum

Momentum = exponentially weighted win rate over the last 10 games prior to
NCAA tournament start (uses the earliest tournament game date as cutoff).
"""
from __future__ import annotations
import os
import time
from datetime import datetime, date

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from team_names import canonicalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/data/raw"
PROC = f"{ROOT}/data/processed"
CACHE_DIR = f"{ROOT}/data/raw/team_schedules"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

YEARS = [y for y in range(2010, 2027) if y != 2020]
N_GAMES = 10
DECAY = 0.85


def _find_table_including_comments(soup: BeautifulSoup, table_id: str):
    table = soup.find("table", id=table_id)
    if table is not None:
        return table
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id in c:
            table = BeautifulSoup(c, "html.parser").find("table", id=table_id)
            if table is not None:
                return table
    return None


def fetch_school_slugs() -> dict[str, str]:
    """Return {canonical_school_name: slug} from the sports-reference schools index."""
    url = "https://www.sports-reference.com/cbb/schools/"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = _find_table_including_comments(soup, "schools")
    if table is None:
        raise RuntimeError("Could not find schools table on sports-reference.")
    slug_map: dict[str, str] = {}
    for row in table.select("tbody tr"):
        a = row.find("a")
        if not a or not a.get("href"):
            continue
        name = a.get_text(strip=True)
        href = a.get("href")  # /cbb/schools/duke/
        parts = href.strip("/").split("/")
        if len(parts) >= 3:
            slug = parts[2]
            slug_map[canonicalize(name)] = slug
    return slug_map


def fetch_schedule(slug: str, year: int) -> pd.DataFrame:
    """Fetch a single team's schedule for a season.
    Returns DataFrame with Date and Result columns.
    Uses cache when available.
    """
    os.makedirs(os.path.join(CACHE_DIR, str(year)), exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, str(year), f"{slug}.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    url = f"https://www.sports-reference.com/cbb/schools/{slug}/{year}-schedule.html"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = _find_table_including_comments(soup, "schedule")
    if table is None:
        raise RuntimeError(f"No schedule table for {slug} {year}")

    rows = []
    for tr in table.select("tbody tr"):
        if "thead" in (tr.get("class") or []):
            continue
        date_cell = tr.find("th", {"data-stat": "date_game"})
        result_cell = tr.find("td", {"data-stat": "game_result"})
        if date_cell is None or result_cell is None:
            continue
        date_str = date_cell.get_text(strip=True)
        res = result_cell.get_text(strip=True)
        if res not in {"W", "L"}:
            continue
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dt):
            continue
        rows.append({"Date": dt.date().isoformat(), "Result": res})

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


def compute_momentum(games: pd.DataFrame, cutoff: date) -> float:
    """Exponentially weighted win rate over last N_GAMES before cutoff."""
    if games.empty:
        return float("nan")
    games = games.copy()
    games["Date"] = pd.to_datetime(games["Date"], errors="coerce")
    games = games[games["Date"].dt.date <= cutoff]
    if games.empty:
        return float("nan")
    games = games.sort_values("Date")
    tail = games.tail(N_GAMES)
    n = len(tail)
    weights = [DECAY ** (n - 1 - i) for i in range(n)]
    wins = (tail["Result"] == "W").astype(float).values
    wsum = sum(weights)
    return float((wins * weights).sum() / wsum) if wsum > 0 else float("nan")


def main():
    os.makedirs(PROC, exist_ok=True)
    seeds = pd.read_csv(f"{RAW}/tourney_seeds.csv")
    results = pd.read_csv(f"{RAW}/tourney_results.csv")
    seeds["Team"] = seeds["Team"].apply(canonicalize)
    results["WTeam"] = results["WTeam"].apply(canonicalize)
    results["LTeam"] = results["LTeam"].apply(canonicalize)

    slug_map = fetch_school_slugs()
    rows = []

    for year in YEARS:
        seeds_y = seeds[seeds.Season == year]
        teams = sorted(seeds_y["Team"].unique())
        # Use earliest tournament game date as cutoff
        res_y = results[results.Season == year]
        if not res_y.empty and res_y["Date"].notna().any():
            cutoff = pd.to_datetime(res_y["Date"], errors="coerce").min().date()
        else:
            cutoff = date(year, 3, 15)

        print(f"[season {year}] teams={len(teams)} cutoff={cutoff}")
        for team in teams:
            slug = slug_map.get(team)
            if slug is None:
                print(f"  [miss] no slug for {team}")
                continue
            try:
                sched = fetch_schedule(slug, year)
                momentum = compute_momentum(sched, cutoff)
                rows.append({"Season": year, "Team": team, "Momentum": momentum})
            except Exception as e:
                print(f"  [fail] {team} ({slug}) {year}: {e}")
            time.sleep(1.0)  # be polite

    out = pd.DataFrame(rows)
    out.to_csv(f"{PROC}/momentum_game_log.csv", index=False)
    print(f"Wrote {PROC}/momentum_game_log.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
