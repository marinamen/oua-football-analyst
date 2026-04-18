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


def compute_sos_adjusted_aggregates(gamelog: pd.DataFrame, coaches: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust each team's per-game offensive stats by the defensive quality of
    their opponents. Facing a top defense and still gaining 300 yards is worth
    more than 300 yards against a weak defense.

    adjustment factor = league_avg_yards_allowed / opponent_yards_allowed
      > 1.0 means opponent had a good defense → your yards are worth more
      < 1.0 means opponent had a weak defense → your yards are worth less
    """
    if gamelog.empty or coaches.empty:
        return pd.DataFrame()

    # defensive quality: avg total yards allowed per team across seasons
    def_quality = (
        coaches.groupby("team")[["def_pass_yds_allowed", "def_rush_yds_allowed"]]
        .mean()
        .assign(def_total=lambda d: d["def_pass_yds_allowed"] + d["def_rush_yds_allowed"])
    )
    league_avg_pass = def_quality["def_pass_yds_allowed"].mean()
    league_avg_rush = def_quality["def_rush_yds_allowed"].mean()

    adj = gamelog.copy()

    def adjustment(opponent: str, col: str, league_avg: float) -> float:
        if opponent not in def_quality.index:
            return 1.0
        opp_val = def_quality.loc[opponent, col]
        if opp_val == 0:
            return 1.0
        return league_avg / opp_val

    adj["pass_factor"] = adj["opponent"].apply(lambda o: adjustment(o, "def_pass_yds_allowed", league_avg_pass))
    adj["rush_factor"] = adj["opponent"].apply(lambda o: adjustment(o, "def_rush_yds_allowed", league_avg_rush))
    adj["passing_yards"] = adj["passing_yards"] * adj["pass_factor"]
    adj["rushing_yards"] = adj["rushing_yards"] * adj["rush_factor"]

    # re-aggregate: sum per team (season totals) to match compute_team_aggregates scale
    num_cols = ["passing_yards", "rushing_yards", "turnovers", "sacks_taken", "penalty_yards"]
    existing = [c for c in num_cols if c in adj.columns]
    agg = adj.groupby("team")[existing].sum().reset_index()
    agg["yards_per_game"] = (agg["passing_yards"] + agg["rushing_yards"]) / 8

    # merge in defensive stats from coaches (defense doesn't get SOS-adjusted here)
    def_cols = ["team", "def_pass_yds_allowed", "def_rush_yds_allowed",
                "def_sacks", "def_forced_fumbles", "turnover_margin",
                "off_pass_yds", "off_rush_yds"]
    coach_avg = coaches.groupby("team")[[c for c in def_cols[1:] if c in coaches.columns]].mean().reset_index()
    agg = agg.merge(coach_avg, on="team", how="left")

    agg["yards_allowed_per_game"] = (
        agg["def_pass_yds_allowed"].fillna(0) + agg["def_rush_yds_allowed"].fillna(0)
    ) / 8
    agg["passing_yards_allowed"] = agg["def_pass_yds_allowed"].fillna(0)
    agg["rushing_yards_allowed"] = agg["def_rush_yds_allowed"].fillna(0)
    return agg


def season_trend(gamelog: pd.DataFrame, team: str) -> pd.DataFrame:
    team_games = gamelog[gamelog["team"] == team].copy().reset_index(drop=True)
    team_games["game_num"] = range(1, len(team_games) + 1)
    team_games["total_offense"] = team_games["passing_yards"] + team_games["rushing_yards"]
    return team_games[["game_num", "opponent", "total_offense", "turnovers",
                        "sacks_taken", "penalty_yards", "result"]]
