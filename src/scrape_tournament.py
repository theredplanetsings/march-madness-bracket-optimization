"""
Task 2: Historical NCAA tournament results 2010-2025.

Scrapes sports-reference.com bracket pages, which expose every game in a
predictable nested-div structure under #brackets > #<region> > .round.

Each game div has two team divs; the one with class "winner" is the winner.
Each team div contains: <span>seed</span>, <a>team name</a>, <a>score</a>.
The boxscore href yields the date.

Output:
  data/raw/tourney_seeds.csv   -> Season, Region, Seed, Team
  data/raw/tourney_results.csv -> Season, Round, Region, DayNum (date), WTeam, WSeed, WScore, LTeam, LSeed, LScore
"""
import os
import re
import time
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

YEARS = [y for y in range(2010, 2026) if y != 2020]  # 2020 cancelled (COVID)

REGION_ROUND_NAMES = ["First Round", "Second Round", "Sweet 16", "Elite 8"]
NATIONAL_ROUND_NAMES = ["Final Four", "Championship"]


def parse_date_from_boxscore_href(href: str):
    # /cbb/boxscores/2024-03-22-14-connecticut.html
    m = re.search(r"/(\d{4})-(\d{2})-(\d{2})-", href or "")
    if not m:
        return None
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()


def parse_team_div(td):
    """Return dict(seed, team, score, date_iso) for a team div, or None."""
    seed_span = td.find("span")
    name_a = td.find_all("a")
    if not seed_span or len(name_a) < 1:
        return None
    seed = seed_span.get_text(strip=True)
    team = name_a[0].get_text(strip=True)
    score = None
    date_iso = None
    if len(name_a) >= 2:
        try:
            score = int(name_a[1].get_text(strip=True))
        except ValueError:
            score = None
        date_iso = parse_date_from_boxscore_href(name_a[1].get("href", ""))
    try:
        seed_int = int(seed)
    except ValueError:
        seed_int = None
    return {"seed": seed_int, "team": team, "score": score, "date": date_iso}


def scrape_year(year: int):
    url = f"https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # Sports-reference sometimes hides bracket inside HTML comments; check both
    brackets_div = soup.find(id="brackets")
    if brackets_div is None:
        # Look in comments
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if "brackets" in c:
                brackets_div = BeautifulSoup(c, "html.parser").find(id="brackets")
                if brackets_div is not None:
                    break
    if brackets_div is None:
        raise RuntimeError(f"No #brackets div found on {url}")

    seeds_rows = []
    games_rows = []
    seen_seeds = set()  # (region, seed, team)

    for region_div in brackets_div.find_all("div", recursive=False):
        region_id = region_div.get("id", "?")  # east/west/south/midwest/national
        rounds = region_div.find_all("div", class_="round")
        round_names = NATIONAL_ROUND_NAMES if region_id == "national" else REGION_ROUND_NAMES
        for ridx, round_div in enumerate(rounds):
            if ridx >= len(round_names):
                # final round in regional bracket is championship slot mirror; skip
                continue
            round_name = round_names[ridx]
            # Each round div contains alternating game divs
            game_divs = [d for d in round_div.find_all("div", recursive=False) if d.find("span")]
            for game in game_divs:
                team_divs = [d for d in game.find_all("div", recursive=False)]
                team_divs = [d for d in team_divs if d.find("span")]
                if len(team_divs) < 2:
                    continue
                t1 = parse_team_div(team_divs[0])
                t2 = parse_team_div(team_divs[1])
                if t1 is None or t2 is None:
                    continue
                w_is_first = "winner" in (team_divs[0].get("class") or [])
                w_is_second = "winner" in (team_divs[1].get("class") or [])
                if w_is_first and not w_is_second:
                    w, l = t1, t2
                elif w_is_second and not w_is_first:
                    w, l = t2, t1
                else:
                    # ambiguous (championship game's winner div may be elsewhere); pick higher score
                    if t1["score"] is not None and t2["score"] is not None:
                        w, l = (t1, t2) if t1["score"] >= t2["score"] else (t2, t1)
                    else:
                        continue

                # Record seeds (region only - national rounds re-list teams already seen)
                if region_id != "national":
                    for t in (t1, t2):
                        key = (region_id, t["seed"], t["team"])
                        if key not in seen_seeds and t["seed"] is not None:
                            seen_seeds.add(key)
                            seeds_rows.append({
                                "Season": year, "Region": region_id,
                                "Seed": t["seed"], "Team": t["team"],
                            })
                games_rows.append({
                    "Season": year, "Region": region_id, "Round": round_name,
                    "Date": w["date"] or l["date"],
                    "WTeam": w["team"], "WSeed": w["seed"], "WScore": w["score"],
                    "LTeam": l["team"], "LSeed": l["seed"], "LScore": l["score"],
                })

    return pd.DataFrame(seeds_rows), pd.DataFrame(games_rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_seeds, all_games = [], []
    for year in YEARS:
        try:
            s, g = scrape_year(year)
            print(f"[ok] {year}: {len(s)} teams, {len(g)} games")
            all_seeds.append(s)
            all_games.append(g)
        except Exception as e:
            print(f"[FAIL] {year}: {e}")
        time.sleep(3.5)  # sports-reference rate-limits aggressively (~20/min)

    seeds = pd.concat(all_seeds, ignore_index=True)
    games = pd.concat(all_games, ignore_index=True)
    seeds_path = os.path.join(OUT_DIR, "tourney_seeds.csv")
    games_path = os.path.join(OUT_DIR, "tourney_results.csv")
    seeds.to_csv(seeds_path, index=False)
    games.to_csv(games_path, index=False)
    print(f"\nWrote {seeds_path}: {len(seeds)} rows")
    print(f"Wrote {games_path}: {len(games)} rows")
    print("\nSample games:")
    print(games.head(10).to_string(index=False))
    print("\nGames per season:")
    print(games.groupby("Season").size().to_string())


if __name__ == "__main__":
    main()
