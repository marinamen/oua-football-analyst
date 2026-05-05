"""
Play-level situational analysis engine.
Ingests a tagged play-by-play Excel/CSV and answers:
  - What does a team do on a given down & distance?
  - What formations/personnel do they favour in each situation?
  - What are their run tendencies by direction and field zone?
  - What are their pass tendencies by depth and coverage situation?
"""

import pandas as pd
import numpy as np


# ── Column normalization ───────────────────────────────────────────────────────

REQUIRED_COLS = {"down", "distance", "play_type"}

COL_ALIASES = {
    # down
    "dn": "down", "dwn": "down",
    # distance
    "dist": "distance", "yards to go": "distance", "ytg": "distance",
    # play type
    "type": "play_type", "play": "play_type", "play type": "play_type",
    # formation
    "form": "formation", "off formation": "formation",
    # personnel
    "pers": "personnel", "personnel group": "personnel",
    # direction
    "dir": "direction", "run dir": "direction", "run direction": "direction",
    # gain
    "yards": "gain", "yds": "gain", "yards gained": "gain",
    # result
    "outcome": "result",
    # quarter
    "qtr": "quarter", "q": "quarter",
    # field zone / yard line
    "yard line": "yard_line", "yl": "yard_line", "field position": "yard_line",
    # hash
    "hash mark": "hash", "hashmark": "hash",
    # motion
    "pre-snap motion": "motion", "pre snap motion": "motion",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=COL_ALIASES)
    return df


def derive_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # distance bucket
    if "distance" in df.columns:
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
        df["dist_bucket"] = pd.cut(
            df["distance"],
            bins=[-1, 2, 6, 100],
            labels=["Short (1-2)", "Medium (3-6)", "Long (7+)"],
        )

    # field zone from yard_line (opponent's yard line, 0=goal line, 100=own goal)
    if "yard_line" in df.columns:
        df["yard_line"] = pd.to_numeric(df["yard_line"], errors="coerce")
        df["field_zone"] = pd.cut(
            df["yard_line"],
            bins=[-1, 10, 20, 50, 100],
            labels=["Red Zone (1-10)", "Scoring Zone (11-20)", "Midfield (21-50)", "Own Territory (51+)"],
        )

    # normalise play_type to Run / Pass / Other
    if "play_type" in df.columns:
        df["play_type"] = df["play_type"].astype(str).str.strip().str.title()
        run_keywords = ["run", "rush", "draw", "sweep", "option", "qb sneak", "kneel"]
        pass_keywords = ["pass", "screen", "play action", "pa", "scramble", "throw"]
        def classify(p):
            pl = p.lower()
            if any(k in pl for k in run_keywords):
                return "Run"
            if any(k in pl for k in pass_keywords):
                return "Pass"
            return p  # keep original if unrecognised
        df["play_category"] = df["play_type"].apply(classify)

    # gain
    if "gain" in df.columns:
        df["gain"] = pd.to_numeric(df["gain"], errors="coerce")

    return df


def load_play_file(uploaded_file) -> pd.DataFrame:
    """Load an uploaded Excel or CSV file and normalise it."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df = normalize_columns(df)
    df = derive_buckets(df)
    return df


def load_play_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df = derive_buckets(df)
    return df


# ── Situational filters ────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filter selections to the play dataframe."""
    out = df.copy()
    for col, val in filters.items():
        if val and val != "All" and col in out.columns:
            if isinstance(val, list):
                out = out[out[col].isin(val)]
            else:
                out = out[out[col].astype(str) == str(val)]
    return out


# ── Analysis functions ─────────────────────────────────────────────────────────

def run_pass_split(df: pd.DataFrame) -> pd.Series:
    if "play_category" not in df.columns or df.empty:
        return pd.Series(dtype=float)
    return df["play_category"].value_counts(normalize=True).mul(100).round(1)


def tendency_by_down_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Run% and Pass% broken down by down × distance bucket."""
    if "down" not in df.columns or "dist_bucket" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby(["down", "dist_bucket"])["play_category"].value_counts(normalize=True)
    grp = grp.mul(100).round(1).reset_index(name="pct")
    return grp


def formation_tendency(df: pd.DataFrame) -> pd.DataFrame:
    if "formation" not in df.columns or df.empty:
        return pd.DataFrame()
    t = df.groupby(["formation", "play_category"]).size().reset_index(name="count")
    total = t.groupby("formation")["count"].transform("sum")
    t["pct"] = (t["count"] / total * 100).round(1)
    return t.sort_values("count", ascending=False)


def personnel_tendency(df: pd.DataFrame) -> pd.DataFrame:
    if "personnel" not in df.columns or df.empty:
        return pd.DataFrame()
    t = df.groupby(["personnel", "play_category"]).size().reset_index(name="count")
    total = t.groupby("personnel")["count"].transform("sum")
    t["pct"] = (t["count"] / total * 100).round(1)
    return t.sort_values("count", ascending=False)


def direction_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """Where do they run? Only looks at run plays."""
    if "direction" not in df.columns or df.empty:
        return pd.DataFrame()
    runs = df[df["play_category"] == "Run"]
    if runs.empty:
        return pd.DataFrame()
    return runs["direction"].value_counts(normalize=True).mul(100).round(1).reset_index(
        names=["direction"]
    ).rename(columns={"proportion": "pct", "count": "pct"})


def avg_gain_by_situation(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns or "gain" not in df.columns or df.empty:
        return pd.DataFrame()
    return (
        df.groupby(group_col)["gain"]
        .agg(avg_gain="mean", plays="count")
        .round(2)
        .reset_index()
        .sort_values("avg_gain", ascending=False)
    )


def redzone_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    if "field_zone" not in df.columns:
        return pd.DataFrame()
    rz = df[df["field_zone"].astype(str).str.contains("Red Zone", na=False)]
    if rz.empty:
        return pd.DataFrame()
    return rz["play_category"].value_counts(normalize=True).mul(100).round(1).reset_index(
        names=["play_type"]
    ).rename(columns={"proportion": "pct", "count": "pct"})


def motion_tendency(df: pd.DataFrame) -> pd.DataFrame:
    if "motion" not in df.columns or df.empty:
        return pd.DataFrame()
    df2 = df.copy()
    df2["motion"] = df2["motion"].astype(str).str.strip().str.lower().isin(
        ["yes", "y", "true", "1", "motion"]
    )
    t = df2.groupby(["motion", "play_category"]).size().reset_index(name="count")
    total = t.groupby("motion")["count"].transform("sum")
    t["pct"] = (t["count"] / total * 100).round(1)
    t["motion"] = t["motion"].map({True: "With Motion", False: "No Motion"})
    return t
