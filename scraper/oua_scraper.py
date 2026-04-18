"""
Scrapes OUA football schedules and results from oua.ca.
Run with: python3 -m scraper.oua_scraper
Outputs to data/raw/schedule_YYYY.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

SEASONS = {
    2024: "2024-25",
    2023: "2023-24",
    2022: "2022-23",
    2021: "2021-22",
}


def scrape_season(year: int) -> pd.DataFrame:
    slug = SEASONS[year]
    url = f"https://oua.ca/sports/fball/{slug}/schedule"
    print(f"Fetching {url} ...")

    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    rows = []
    game_id = 1

    for table in soup.select("table.table"):
        # each table = one week; caption or preceding heading has the date
        caption = table.find_previous(["h2", "h3", "caption", "div"], class_=lambda c: c and "date" in (c or "").lower())
        week_label = caption.get_text(strip=True) if caption else ""

        for row in table.select("tr.event-row"):
            away_td = row.select_one("td.awayteam")
            home_td = row.select_one("td.hometeam")
            status_td = row.select_one("td.status")

            if not away_td or not home_td:
                continue

            away_team = away_td.select_one(".team-name")
            home_team = home_td.select_one(".team-name")

            away_score_el = away_td.select_one(".result")
            home_score_el = home_td.select_one(".result")

            away_score = away_score_el.get_text(strip=True) if away_score_el else ""
            home_score = home_score_el.get_text(strip=True) if home_score_el else ""
            # strip the caret arrow from winner score
            home_score = home_score.replace("", "").strip()
            away_score = away_score.replace("", "").strip()

            status = status_td.get_text(strip=True) if status_td else ""

            # only keep completed games
            if "Final" not in status:
                continue

            rows.append({
                "game_id": f"{year}_{game_id:03d}",
                "season": year,
                "week_label": week_label,
                "away_team": away_team.get_text(strip=True) if away_team else "",
                "home_team": home_team.get_text(strip=True) if home_team else "",
                "away_score": away_score,
                "home_score": home_score,
            })
            game_id += 1

    df = pd.DataFrame(rows)
    out = RAW_DIR / f"schedule_{year}.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} games → {out}")
    return df


if __name__ == "__main__":
    all_games = []
    for year in [2024, 2023, 2022]:
        try:
            df = scrape_season(year)
            all_games.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"  ERROR for {year}: {e}")

    if all_games:
        combined = pd.concat(all_games, ignore_index=True)
        out = RAW_DIR / "all_games.csv"
        combined.to_csv(out, index=False)
        print(f"\nAll seasons combined → {out} ({len(combined)} total games)")
        print(combined.head(10).to_string(index=False))
