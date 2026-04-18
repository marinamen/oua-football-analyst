"""
Logistic regression model to predict win probability for OUA matchups.
Trained on historical game + team stat data.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path

from analysis.team_stats import load_data, compute_team_aggregates

MODEL_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "yards_per_game", "yards_allowed_per_game",
    "turnovers", "turnover_margin",
    "def_sacks", "def_forced_fumbles",
    "passing_yards", "rushing_yards",
    "passing_yards_allowed", "rushing_yards_allowed",
]


def build_training_set(games: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, compute feature deltas (home - away) and label = 1 if home won.
    Uses season-averaged stats per team as features.
    """
    home_stats = agg.rename(columns={c: f"home_{c}" for c in agg.columns if c != "team"})
    away_stats = agg.rename(columns={c: f"away_{c}" for c in agg.columns if c != "team"})

    df = games.copy()
    df = df.merge(home_stats, left_on="home_team", right_on="team", how="left")
    df = df.merge(away_stats, left_on="away_team", right_on="team", how="left")
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    for feat in FEATURES:
        h, a = f"home_{feat}", f"away_{feat}"
        if h in df.columns and a in df.columns:
            df[f"delta_{feat}"] = df[h] - df[a]

    delta_cols = [f"delta_{f}" for f in FEATURES if f"delta_{f}" in df.columns]
    return df.dropna(subset=delta_cols)


def train(games: pd.DataFrame, agg: pd.DataFrame) -> tuple:
    df = build_training_set(games, agg)
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    X = df[delta_cols].values
    y = df["home_win"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=500, C=0.5)
    scores = cross_val_score(model, X_scaled, y, cv=min(5, len(y)), scoring="accuracy")
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(delta_cols, MODEL_DIR / "features.pkl")

    return model, scaler, delta_cols, scores.mean()


def predict_matchup(home_team: str, away_team: str, agg: pd.DataFrame) -> dict:
    """Return win probability and key factor breakdown for a matchup."""
    try:
        model = joblib.load(MODEL_DIR / "model.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        delta_cols = joblib.load(MODEL_DIR / "features.pkl")
    except FileNotFoundError:
        return {"error": "Model not trained yet. Add data and click Train Model."}

    home = agg[agg["team"] == home_team].iloc[0]
    away = agg[agg["team"] == away_team].iloc[0]

    feat_map = {f"delta_{f}": f for f in FEATURES}
    deltas = []
    breakdown = {}
    for col in delta_cols:
        feat = col.replace("delta_", "")
        if feat in home.index and feat in away.index:
            delta = home[feat] - away[feat]
        else:
            delta = 0.0
        deltas.append(delta)
        breakdown[feat] = {"home": float(home.get(feat, 0)), "away": float(away.get(feat, 0))}

    X = scaler.transform([deltas])
    prob_home = model.predict_proba(X)[0][1]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": round(prob_home, 3),
        "away_win_prob": round(1 - prob_home, 3),
        "breakdown": breakdown,
    }
