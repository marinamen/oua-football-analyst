"""
Scrapes per-game team stats from oua.ca team game log pages.
Run with: python3 -m scraper.stats_scraper
Outputs to data/raw/team_stats_all.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time
import re

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

TEAMS = [
    "carleton", "guelph", "laurier", "mcmaster", "ottawa",
    "queens", "toronto", "waterloo", "western", "windsor", "york",
]

SEASONS = ["2024-25", "2023-24", "2022-23"]

TEAM_NAME_MAP = {
    "carleton": "Carleton", "guelph": "Guelph", "laurier": "Laurier",
    "mcmaster": "McMaster", "ottawa": "Ottawa", "queens": "Queen's",
    "toronto": "Toronto", "waterloo": "Waterloo", "western": "Western",
    "windsor": "Windsor", "york": "York",
}


def parse_top(top_str: str) -> int:
    """Convert 'MM:SS' time of possession to seconds."""
    try:
        parts = top_str.strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0


def parse_ca(ca_str: str) -> tuple[int, int]:
    """Parse 'completions-attempts' string."""
    try:
        c, a = ca_str.split("-")
        return int(c), int(a)
    except Exception:
        return 0, 0


def safe_int(val: str) -> int:
    val = val.strip().replace("-", "0")
    try:
        return int(val)
    except Exception:
        return 0


def safe_float(val: str) -> float:
    val = val.strip().replace("-", "0")
    try:
        return float(val)
    except Exception:
        return 0.0


def scrape_team_gamelog(team_slug: str, season: str) -> pd.DataFrame:
    url = f"https://oua.ca/sports/fball/{season}/teams/{team_slug}?view=gamelog"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Table 1 = per-game offense/defense stats
    tables = soup.find_all("table")
    if len(tables) < 2:
        return pd.DataFrame()

    stat_table = tables[1]
    rows = []

    for tr in stat_table.find_all("tr")[1:]:  # skip header
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) < 14:
            continue

        # cols: Date, Opponent, Score, yds, pass, c-a, comp%, rush, r, y/r, int, fum, tack, sac, pen yds, top
        date_raw = cols[0].replace("#", "").strip()
        opponent_raw = cols[1].strip()
        score_raw = cols[2].strip()      # e.g. "W, 42-23" or "L, 10-37"
        total_yds = safe_int(cols[3])
        pass_yds = safe_int(cols[4])
        ca = cols[5]
        comp_pct = cols[6]
        rush_yds = safe_int(cols[7])
        rush_att = safe_int(cols[8])
        turnovers_int = safe_int(cols[10])
        turnovers_fum = safe_int(cols[11])
        sacks_taken = safe_int(cols[13])
        pen_yds = safe_int(cols[14])
        top_sec = parse_top(cols[15]) if len(cols) > 15 else 0

        completions, attempts = parse_ca(ca)

        # parse result and scores
        result = ""
        team_score, opp_score = 0, 0
        m = re.match(r"([WL]),\s*(\d+)-(\d+)", score_raw)
        if m:
            result = m.group(1)
            team_score = int(m.group(2))
            opp_score = int(m.group(3))

        # clean opponent name (strip "at " prefix)
        opponent = re.sub(r"^at\s+", "", opponent_raw).strip()

        rows.append({
            "season": season,
            "team": TEAM_NAME_MAP.get(team_slug, team_slug),
            "date": date_raw,
            "opponent": opponent,
            "home_away": "away" if "at " in opponent_raw else "home",
            "result": result,
            "team_score": team_score,
            "opp_score": opp_score,
            "total_yards": total_yds,
            "passing_yards": pass_yds,
            "rushing_yards": rush_yds,
            "rush_attempts": rush_att,
            "completions": completions,
            "attempts": attempts,
            "turnovers_int": turnovers_int,
            "turnovers_fum": turnovers_fum,
            "turnovers": turnovers_int + turnovers_fum,
            "sacks_taken": sacks_taken,
            "penalty_yards": pen_yds,
            "time_of_possession_sec": top_sec,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_rows = []
    for season in SEASONS:
        print(f"\n=== {season} ===")
        for team in TEAMS:
            try:
                df = scrape_team_gamelog(team, season)
                print(f"  {TEAM_NAME_MAP[team]:12s}: {len(df)} games")
                all_rows.append(df)
                time.sleep(0.5)
            except Exception as e:
                print(f"  {team}: ERROR — {e}")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out = RAW_DIR / "team_stats_all.csv"
        combined.to_csv(out, index=False)
        print(f"\nSaved {len(combined)} team-game rows → {out}")
        print(combined.head(5).to_string(index=False))
