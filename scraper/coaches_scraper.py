"""
Scrapes season-level team stats from the OUA coaches view (print template).
Captures offense totals, opponent (defense allowed) totals, and defensive stats.
Run with: python3 -m scraper.coaches_scraper
Outputs to data/raw/coaches_stats_all.csv
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

SEASONS = ["2025-26", "2024-25", "2023-24"]

TEAM_NAME_MAP = {
    "carleton": "Carleton", "guelph": "Guelph", "laurier": "Laurier",
    "mcmaster": "McMaster", "ottawa": "Ottawa", "queens": "Queen's",
    "toronto": "Toronto", "waterloo": "Waterloo", "western": "Western",
    "windsor": "Windsor", "york": "York",
}


def safe_int(val: str) -> int:
    val = re.sub(r"[^\d\-]", "", val.strip())
    try:
        return int(val.split("-")[0]) if val else 0
    except Exception:
        return 0


def safe_float(val: str) -> float:
    try:
        return float(re.sub(r"[^\d\.]", "", val.strip()))
    except Exception:
        return 0.0


def get_totals_and_opponent(table) -> tuple[list, list]:
    """Return the Totals row and Opponent row from a stat table."""
    totals, opponent = [], []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        if cells[0] == "Totals":
            totals = cells
        elif cells[0] == "Opponent":
            opponent = cells
    return totals, opponent


def parse_passing(table) -> dict:
    totals, opponent = get_totals_and_opponent(table)
    # Totals cols: Totals, GP, CMP-INT, PCT, YDS, TD, Long, AVG/G
    result = {}
    if len(totals) >= 6:
        result["off_pass_yds"] = safe_int(totals[4])
        result["off_pass_tds"] = safe_int(totals[5])
        result["off_pass_int"] = safe_int(totals[2].split("-")[-1]) if "-" in totals[2] else 0
    if len(opponent) >= 6:
        result["def_pass_yds_allowed"] = safe_int(opponent[4])
        result["def_pass_tds_allowed"] = safe_int(opponent[5])
        result["def_int_made"] = safe_int(opponent[2].split("-")[-1]) if "-" in opponent[2] else 0
    return result


def parse_rushing(table) -> dict:
    totals, opponent = get_totals_and_opponent(table)
    # Totals cols: Totals, GP, Att, Gain, Loss, Net, AVG, Td, Long, AVG/G
    result = {}
    if len(totals) >= 8:
        result["off_rush_yds"] = safe_int(totals[5])
        result["off_rush_tds"] = safe_int(totals[7])
        result["off_rush_att"] = safe_int(totals[2])
    if len(opponent) >= 8:
        result["def_rush_yds_allowed"] = safe_int(opponent[5])
        result["def_rush_tds_allowed"] = safe_int(opponent[7])
    return result


def parse_defense(table) -> dict:
    totals, _ = get_totals_and_opponent(table)
    # Totals cols: Totals, GP, -, total_tackles, TFL-Yards, SCK-YDS, INT-YDS, BU, FF, FR-YDS, BLK
    result = {}
    if len(totals) >= 10:
        result["def_total_tackles"] = safe_float(totals[3])
        result["def_tfl"] = safe_float(totals[4].split("-")[0]) if "-" in totals[4] else 0
        result["def_sacks"] = safe_float(totals[5].split("-")[0]) if "-" in totals[5] else 0
        result["def_int_yds"] = totals[6]
        result["def_pass_breakups"] = safe_int(totals[7])
        result["def_forced_fumbles"] = safe_int(totals[8])
    return result


def scrape_team_coaches(team_slug: str, season: str) -> dict | None:
    url = f"https://oua.ca/sports/fball/{season}/teams/{team_slug}?tmpl=teaminfo-network-monospace-template"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"    request failed: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table")
    if len(tables) < 9:
        print(f"    only {len(tables)} tables found, skipping")
        return None

    row = {
        "season": season,
        "team": TEAM_NAME_MAP.get(team_slug, team_slug),
    }

    row.update(parse_passing(tables[1]))
    row.update(parse_rushing(tables[2]))
    row.update(parse_defense(tables[8]))

    # derived
    row["off_total_yds"] = row.get("off_pass_yds", 0) + row.get("off_rush_yds", 0)
    row["def_total_yds_allowed"] = row.get("def_pass_yds_allowed", 0) + row.get("def_rush_yds_allowed", 0)
    row["off_total_tds"] = row.get("off_pass_tds", 0) + row.get("off_rush_tds", 0)
    row["def_total_tds_allowed"] = row.get("def_pass_tds_allowed", 0) + row.get("def_rush_tds_allowed", 0)
    row["turnover_margin"] = row.get("def_int_made", 0) + row.get("def_forced_fumbles", 0) - row.get("off_pass_int", 0)

    return row


if __name__ == "__main__":
    all_rows = []
    for season in SEASONS:
        print(f"\n=== {season} ===")
        for team in TEAMS:
            print(f"  {TEAM_NAME_MAP[team]:12s}", end=" ", flush=True)
            row = scrape_team_coaches(team, season)
            if row:
                all_rows.append(row)
                print(f"✓  pass_yds={row.get('off_pass_yds')}  rush_yds={row.get('off_rush_yds')}  "
                      f"pass_allowed={row.get('def_pass_yds_allowed')}  sacks={row.get('def_sacks')}")
            time.sleep(0.5)

    if all_rows:
        df = pd.DataFrame(all_rows)
        out = RAW_DIR / "coaches_stats_all.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows → {out}")
        print(df.to_string(index=False))
