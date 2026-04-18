"""
Computes per-team aggregate stats, weakness scores, and trend data
from scraped OUA game log CSVs.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_games = DATA_DIR / "raw" / "all_games.csv"
    raw_gamelog = DATA_DIR / "raw" / "team_stats_all.csv"
    raw_coaches = DATA_DIR / "raw" / "coaches_stats_all.csv"

    if raw_games.exists():
        games = pd.read_csv(raw_games)
        games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
        games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")
        games = games.dropna(subset=["home_score", "away_score"])
        games["date"] = pd.NaT
        games["week"] = games.groupby("season").cumcount() // 5 + 1
    else:
        games = pd.read_csv(DATA_DIR / "manual" / "games_template.csv", parse_dates=["date"])

    gamelog = pd.read_csv(raw_gamelog) if raw_gamelog.exists() else pd.DataFrame()
    coaches = pd.read_csv(raw_coaches) if raw_coaches.exists() else pd.DataFrame()
    return games, gamelog, coaches


def compute_team_aggregates(coaches: pd.DataFrame) -> pd.DataFrame:
    """Average coaches-view season stats per team across all available seasons."""
    num_cols = coaches.select_dtypes(include="number").columns.tolist()
    agg = coaches.groupby("team")[num_cols].mean().reset_index()

    agg["yards_per_game"] = (agg["off_pass_yds"] + agg["off_rush_yds"]) / 8
    agg["yards_allowed_per_game"] = (agg["def_pass_yds_allowed"] + agg["def_rush_yds_allowed"]) / 8
    agg["passing_yards"] = agg["off_pass_yds"]
    agg["rushing_yards"] = agg["off_rush_yds"]
    agg["passing_yards_allowed"] = agg["def_pass_yds_allowed"]
    agg["rushing_yards_allowed"] = agg["def_rush_yds_allowed"]
    agg["turnovers"] = agg["off_pass_int"]
    agg["sacks_taken"] = agg["def_sacks"]  # sacks our defense makes
    agg["penalty_yards"] = agg.get("penalty_yards", 0)
    return agg


def weakness_scores(agg: pd.DataFrame) -> pd.DataFrame:
    df = agg.copy()

    def rank_desc(col):
        return df[col].rank(ascending=False) if col in df.columns else pd.Series(np.nan, index=df.index)

    def rank_asc(col):
        return df[col].rank(ascending=True) if col in df.columns else pd.Series(np.nan, index=df.index)

    df["weakness_pass_defense"] = rank_desc("passing_yards_allowed")
    df["weakness_run_defense"] = rank_desc("rushing_yards_allowed")
    df["weakness_ball_security"] = rank_desc("turnovers")
    df["weakness_pass_offense"] = rank_asc("passing_yards")
    df["weakness_run_offense"] = rank_asc("rushing_yards")
    df["weakness_turnover_margin"] = rank_asc("turnover_margin")

    weakness_cols = [c for c in df.columns if c.startswith("weakness_")]
    df["overall_weakness_score"] = df[weakness_cols].mean(axis=1)
    return df


def season_trend(gamelog: pd.DataFrame, team: str) -> pd.DataFrame:
    team_games = gamelog[gamelog["team"] == team].copy().reset_index(drop=True)
    team_games["game_num"] = range(1, len(team_games) + 1)
    team_games["total_offense"] = team_games["passing_yards"] + team_games["rushing_yards"]
    return team_games[["game_num", "opponent", "total_offense", "turnovers",
                        "sacks_taken", "penalty_yards", "result"]]
