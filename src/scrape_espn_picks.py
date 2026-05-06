"""
Task 3: ESPN "Who Picked Whom" public-pick percentages.

The live URL (fantasy.espn.com/.../whopickedwhom) returns a JS shell only -
the data is loaded client-side from internal ESPN APIs that require auth /
session cookies. The clean alternative is the Internet Archive's Wayback
Machine, which has full HTML snapshots for many years where the table is
already rendered server-side.

Approach:
  1. Use the Wayback CDX API to find the *latest* snapshot of the
     whopickedwhom page taken during the tournament window for each year
     (closest to Apr 1 of that year - this is when Final Four picks are
     finalized and percentages reflect actual completed brackets).
  2. Fetch that snapshot, parse the <table class="wpw-table"> which has
     one row per advancing team across columns R64..NCG.
  3. Save one CSV per year.

Output: data/raw/espn_picks_YYYY.csv with columns:
    Year, Round, Seed, Team, Percent
where Round in {R64, R32, S16, E8, F4, NCG}.
"""
import os
import re
import time
import urllib.parse

import pandas as pd
import requests
from bs4 import BeautifulSoup

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

YEARS = list(range(2010, 2026))

ROUND_ALIASES = {
    "r64": "R64",
    "round of 64": "R64",
    "first round": "R64",
    "r32": "R32",
    "round of 32": "R32",
    "second round": "R32",
    "sweet 16": "S16",
    "s16": "S16",
    "elite 8": "E8",
    "e8": "E8",
    "final four": "F4",
    "f4": "F4",
    "championship": "NCG",
    "title": "NCG",
    "ncg": "NCG",
}

# Multiple URL patterns ESPN has used over the years
URL_PATTERNS = [
    "https://fantasy.espn.com/tournament-challenge-bracket/{yr}/en/whopickedwhom",
    "https://games.espn.com/tournament-challenge-bracket/{yr}/en/whopickedwhom",
    "https://games.espn.go.com/tournament-challenge-bracket/{yr}/en/whopickedwhom",
    "https://espn.go.com/tournament-challenge-bracket/{yr}/en/whopickedwhom",
]


def _normalize_round(label: str):
    if not label:
        return None
    key = label.strip().lower()
    key = re.sub(r"\s+", " ", key)
    return ROUND_ALIASES.get(key)


def list_snapshots(year: int):
    """Return a list of Wayback snapshot URLs near tournament time.
    We search all URL patterns and filter to March/April captures.
    """
    snapshots = []
    for pat in URL_PATTERNS:
        target = pat.format(yr=year)
        cdx = (
            "https://web.archive.org/cdx/search/cdx?"
            f"url={urllib.parse.quote(target)}&from={year}&to={year}"
            "&filter=statuscode:200&output=json&fl=timestamp,original"
        )
        try:
            r = requests.get(cdx, timeout=30, headers=HEADERS)
            if r.status_code != 200:
                continue
            rows = r.json()[1:]
            for ts, orig in rows:
                # Keep March/April captures when brackets are stable
                if ts[4:8] in {"0301", "0315", "0320", "0325", "0331", "0401", "0405", "0410", "0415"}:
                    snapshots.append((ts, f"https://web.archive.org/web/{ts}/{orig}"))
                else:
                    # Coarser filter: March/April only
                    if ts[4:6] in {"03", "04"}:
                        snapshots.append((ts, f"https://web.archive.org/web/{ts}/{orig}"))
        except Exception:
            continue
        time.sleep(0.5)
    # Deduplicate and prefer latest snapshots first
    snapshots = list({url: (ts, url) for ts, url in snapshots}.values())
    snapshots.sort(key=lambda x: x[0], reverse=True)
    return snapshots


def parse_wpw_table(html: str, year: int):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_=re.compile(r"wpw-table"))
    if table is None:
        # Try generic - some snapshots may have different markup
        return []
    headers = [th.get_text(strip=True) for th in table.select("thead th")]
    if not headers:
        headers = ["R64", "R32", "S16", "E8", "F4", "NCG"]
    rows = []
    for tr in table.select("tbody tr"):
        cells = tr.find_all("td", recursive=False)
        for col_idx, cell in enumerate(cells):
            seed_span = cell.find("span", class_="seed")
            team_span = cell.find("span", class_="teamName")
            pct_span = cell.find("span", class_="percentage")
            if not (team_span and pct_span):
                continue
            try:
                seed = int(seed_span.get_text(strip=True)) if seed_span else None
            except ValueError:
                seed = None
            pct_text = pct_span.get_text(strip=True).replace("%", "")
            try:
                pct = float(pct_text)
            except ValueError:
                continue
            round_label = headers[col_idx] if col_idx < len(headers) else f"col{col_idx}"
            round_label = _normalize_round(round_label) or round_label
            rows.append({
                "Year": year, "Round": round_label, "Seed": seed,
                "Team": team_span.get_text(strip=True), "Percent": pct,
            })
    return rows


def _valid_rows(rows: list[dict]) -> bool:
    if not rows:
        return False
    df = pd.DataFrame(rows)
    if df.empty:
        return False
    # Expect at least 200 cells in a full tournament table
    if len(df) < 200:
        return False
    # Require known rounds
    expected = {"R64", "R32", "S16", "E8", "F4", "NCG"}
    have = set(df["Round"].unique())
    return len(expected.intersection(have)) >= 4


def fetch_year(year: int):
    snapshots = list_snapshots(year)
    if not snapshots:
        print(f"[skip] {year}: no Wayback snapshots found")
        return None, None
    for ts, url in snapshots:
        try:
            r = requests.get(url, headers=HEADERS, timeout=45)
            if r.status_code != 200:
                continue
            rows = parse_wpw_table(r.text, year)
            if _valid_rows(rows):
                print(f"[snap] {year}: {ts} -> {url}")
                return ts, rows
        except Exception:
            continue
        time.sleep(0.5)
    print(f"[skip] {year}: no valid snapshot table found")
    return None, None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    all_rows = []
    for yr in YEARS:
        ts, rows = fetch_year(yr)
        if rows:
            df = pd.DataFrame(rows)
            out_path = os.path.join(OUT_DIR, f"espn_picks_{yr}.csv")
            df.to_csv(out_path, index=False)
            print(f"[ok]   {yr}: wrote {len(df)} rows -> {out_path}")
            summary.append((yr, len(df), ts, "ok"))
            all_rows.append(df)
        else:
            summary.append((yr, 0, ts, "missing"))
        time.sleep(2)

    print("\n=== SUMMARY ===")
    for yr, n, ts, st in summary:
        print(f"  {yr}: {n:>4} rows  ({st}, snapshot={ts})")

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_path = os.path.join(OUT_DIR, "espn_picks_all.csv")
        all_df.to_csv(all_path, index=False)
        print(f"\nWrote combined file: {all_path} ({len(all_df)} rows)")


if __name__ == "__main__":
    main()
