"""
Advanced scouting analytics — beyond what coaches see in the box score.
Four engines:
  1. win_condition_fingerprint  — stats that separate a team's wins from losses
  2. how_to_beat                — common patterns across a team's losses
  3. momentum_score             — recent form vs season baseline (weighted)
  4. matchup_exploiter          — biggest statistical mismatches in a specific game
"""

import pandas as pd
import numpy as np


# ── 1. Win Condition Fingerprint ───────────────────────────────────────────────

STAT_LABELS = {
    "total_offense": "Total Offensive Yards",
    "passing_yards": "Passing Yards",
    "rushing_yards": "Rushing Yards",
    "turnovers": "Turnovers Committed",
    "sacks_taken": "Sacks Taken",
    "penalty_yards": "Penalty Yards",
}


def win_condition_fingerprint(gamelog: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    For each tracked stat, compute the mean in wins vs losses and the
    % swing. Returns rows sorted by absolute impact.
    """
    df = gamelog[gamelog["team"] == team].copy()
    df["total_offense"] = df["passing_yards"] + df["rushing_yards"]

    wins = df[df["result"] == "W"]
    losses = df[df["result"] == "L"]

    if wins.empty or losses.empty:
        return pd.DataFrame()

    rows = []
    for col, label in STAT_LABELS.items():
        if col not in df.columns:
            continue
        w_mean = wins[col].mean()
        l_mean = losses[col].mean()
        if l_mean == 0:
            continue
        pct_diff = ((w_mean - l_mean) / abs(l_mean)) * 100
        # for turnovers/sacks/penalties, fewer = better when winning
        better_when = "higher" if pct_diff > 0 else "lower"
        if col in ("turnovers", "sacks_taken", "penalty_yards"):
            better_when = "lower" if pct_diff > 0 else "higher"
        rows.append({
            "Stat": label,
            "Avg in Wins": round(w_mean, 1),
            "Avg in Losses": round(l_mean, 1),
            "Swing": f"{pct_diff:+.0f}%",
            "swing_abs": abs(pct_diff),
            "better_when": better_when,
            "_raw_diff": pct_diff,
        })

    result = pd.DataFrame(rows).sort_values("swing_abs", ascending=False)
    return result.drop(columns=["swing_abs"])


# ── 2. How to Beat [Team] ──────────────────────────────────────────────────────

def how_to_beat(gamelog: pd.DataFrame, target_team: str) -> dict:
    """
    Analyse what a team looks like when they lose — surface the patterns
    opponents exploited. Returns a dict of findings with narrative text.
    """
    df = gamelog[gamelog["team"] == target_team].copy()
    df["total_offense"] = df["passing_yards"] + df["rushing_yards"]

    wins = df[df["result"] == "W"]
    losses = df[df["result"] == "L"]

    if losses.empty:
        return {"summary": f"{target_team} has no losses in the dataset."}

    findings = []

    # Offensive collapse
    off_drop = losses["total_offense"].mean() - wins["total_offense"].mean() if not wins.empty else 0
    if abs(off_drop) > 30:
        findings.append({
            "category": "Offensive Output",
            "insight": (
                f"In losses, {target_team}'s offense drops to "
                f"{losses['total_offense'].mean():.0f} yards/game "
                f"({'↓' if off_drop < 0 else '↑'}{abs(off_drop):.0f} vs wins). "
                f"Force a low-yardage game."
            ),
            "priority": abs(off_drop),
        })

    # Turnover vulnerability
    to_loss = losses["turnovers"].mean()
    to_win = wins["turnovers"].mean() if not wins.empty else 0
    if to_loss - to_win > 0.4:
        findings.append({
            "category": "Turnovers",
            "insight": (
                f"{target_team} turns it over {to_loss:.1f}x/game in losses vs "
                f"{to_win:.1f}x in wins. Pressure the QB and strip the ball — "
                f"turnover margin is their biggest loss predictor."
            ),
            "priority": (to_loss - to_win) * 40,
        })

    # Rush vs pass balance in losses
    rush_loss = losses["rushing_yards"].mean()
    pass_loss = losses["passing_yards"].mean()
    rush_win = wins["rushing_yards"].mean() if not wins.empty else rush_loss
    if rush_win - rush_loss > 25:
        findings.append({
            "category": "Run Game",
            "insight": (
                f"Their run game collapses in losses: {rush_loss:.0f} rush yards/game "
                f"vs {rush_win:.0f} in wins. Commit to stopping the run — "
                f"force them into obvious passing situations."
            ),
            "priority": rush_win - rush_loss,
        })
    elif pass_loss < losses["passing_yards"].quantile(0.4):
        findings.append({
            "category": "Pass Defense",
            "insight": (
                f"When held to {pass_loss:.0f} passing yards or fewer, "
                f"{target_team} struggles to win. Apply press coverage early."
            ),
            "priority": 30,
        })

    # Sack/pressure vulnerability
    sack_loss = losses["sacks_taken"].mean()
    sack_win = wins["sacks_taken"].mean() if not wins.empty else 0
    if sack_loss - sack_win > 0.5:
        findings.append({
            "category": "Pass Rush",
            "insight": (
                f"They absorb {sack_loss:.1f} sacks/game in losses vs {sack_win:.1f} in wins. "
                f"Their O-line breaks down under consistent pressure — bring heat early."
            ),
            "priority": (sack_loss - sack_win) * 35,
        })

    # Penalty discipline
    pen_loss = losses["penalty_yards"].mean()
    pen_win = wins["penalty_yards"].mean() if not wins.empty else pen_loss
    if pen_loss - pen_win > 15:
        findings.append({
            "category": "Discipline",
            "insight": (
                f"Penalty yards spike to {pen_loss:.0f}/game in losses vs {pen_win:.0f} in wins. "
                f"Physicality and tempo get in their heads — play fast and physical."
            ),
            "priority": pen_loss - pen_win,
        })

    findings.sort(key=lambda x: x["priority"], reverse=True)

    win_rate = len(wins) / len(df) * 100
    summary = (
        f"{target_team} goes {len(wins)}-{len(losses)} in this dataset ({win_rate:.0f}% win rate). "
        f"Top {min(3, len(findings))} patterns found in their losses:"
    )

    return {"summary": summary, "findings": findings[:4]}


# ── 3. Momentum Score ─────────────────────────────────────────────────────────

def momentum_score(gamelog: pd.DataFrame, team: str) -> dict:
    """
    Weight recent games more heavily (exponential decay) and compare to
    season baseline. Returns a score, label, and per-game trend data.
    """
    df = gamelog[gamelog["team"] == team].copy().reset_index(drop=True)
    df["total_offense"] = df["passing_yards"] + df["rushing_yards"]
    df["game_num"] = range(1, len(df) + 1)

    if len(df) < 3:
        return {"score": 50, "label": "Insufficient data", "trend": df}

    n = len(df)
    weights = np.exp(np.linspace(0, 1.5, n))  # exponential — recent games weighted more
    weights /= weights.sum()

    # composite score: off yards (positive), turnovers/sacks (negative)
    off_norm = (df["total_offense"] - df["total_offense"].mean()) / (df["total_offense"].std() + 1e-9)
    to_norm = (df["turnovers"] - df["turnovers"].mean()) / (df["turnovers"].std() + 1e-9)
    sack_norm = (df["sacks_taken"] - df["sacks_taken"].mean()) / (df["sacks_taken"].std() + 1e-9)

    composite = off_norm - to_norm - sack_norm
    weighted_recent = np.dot(composite.values, weights)
    season_avg = composite.mean()

    # scale to 0-100
    score = float(np.clip(50 + weighted_recent * 20, 0, 100))

    # trend direction: slope of composite over last 4 games
    recent = composite.values[-4:]
    slope = np.polyfit(range(len(recent)), recent, 1)[0]

    if slope > 0.15:
        label = "Hot — trending up"
        color = "green"
    elif slope < -0.15:
        label = "Cold — trending down"
        color = "red"
    else:
        label = "Steady"
        color = "orange"

    df["composite"] = composite.values
    df["label"] = df["game_num"].astype(str) + ". " + df["opponent"] + " (" + df["result"] + ")"

    return {
        "score": round(score, 1),
        "label": label,
        "color": color,
        "trend_df": df[["game_num", "label", "total_offense", "turnovers", "composite"]],
        "season_avg_offense": round(df["total_offense"].mean(), 1),
        "recent_avg_offense": round(df["total_offense"].tail(3).mean(), 1),
    }


# ── 4. Matchup Exploiter ──────────────────────────────────────────────────────

def matchup_exploiter(agg: pd.DataFrame, home_team: str, away_team: str) -> list[dict]:
    """
    Surface the 3 biggest statistical mismatches and frame them as game-plan items.
    agg must have per-team season averages including offense and defense columns.
    """
    if home_team not in agg["team"].values or away_team not in agg["team"].values:
        return []

    h = agg[agg["team"] == home_team].iloc[0]
    a = agg[agg["team"] == away_team].iloc[0]

    # rank all teams for each metric (lower rank = better)
    def rank(col, ascending=False):
        return agg[col].rank(ascending=ascending, pct=True) if col in agg.columns else None

    matchups = []

    # Home pass offense vs Away pass defense
    if "off_pass_yds" in agg.columns and "def_pass_yds_allowed" in agg.columns:
        h_pass_off_pct = rank("off_pass_yds").loc[agg["team"] == home_team].values[0]
        a_pass_def_pct = rank("def_pass_yds_allowed", ascending=True).loc[agg["team"] == away_team].values[0]
        mismatch = h_pass_off_pct + (1 - a_pass_def_pct)
        matchups.append({
            "title": f"{home_team} Pass Offense vs {away_team} Pass Defense",
            "home_val": f"{h['off_pass_yds']:.0f} pass yds/season",
            "away_val": f"{a['def_pass_yds_allowed']:.0f} allowed/season",
            "edge": home_team if h["off_pass_yds"] > a["def_pass_yds_allowed"] else away_team,
            "recommendation": (
                f"{'Attack through the air' if h['off_pass_yds'] > a['def_pass_yds_allowed'] else 'Contain the pass — force the run'}."
            ),
            "mismatch_score": mismatch,
        })

    # Home rush offense vs Away rush defense
    if "off_rush_yds" in agg.columns and "def_rush_yds_allowed" in agg.columns:
        h_rush_off_pct = rank("off_rush_yds").loc[agg["team"] == home_team].values[0]
        a_rush_def_pct = rank("def_rush_yds_allowed", ascending=True).loc[agg["team"] == away_team].values[0]
        mismatch = h_rush_off_pct + (1 - a_rush_def_pct)
        matchups.append({
            "title": f"{home_team} Run Game vs {away_team} Rush Defense",
            "home_val": f"{h['off_rush_yds']:.0f} rush yds/season",
            "away_val": f"{a['def_rush_yds_allowed']:.0f} allowed/season",
            "edge": home_team if h["off_rush_yds"] > a["def_rush_yds_allowed"] else away_team,
            "recommendation": (
                f"{'Establish the run early — their rush D is exploitable' if h['off_rush_yds'] > a['def_rush_yds_allowed'] else 'Shut down the run, make them one-dimensional'}."
            ),
            "mismatch_score": mismatch,
        })

    # Away pass offense vs Home pass defense
    if "off_pass_yds" in agg.columns and "def_pass_yds_allowed" in agg.columns:
        a_pass_off_pct = rank("off_pass_yds").loc[agg["team"] == away_team].values[0]
        h_pass_def_pct = rank("def_pass_yds_allowed", ascending=True).loc[agg["team"] == home_team].values[0]
        mismatch = a_pass_off_pct + (1 - h_pass_def_pct)
        matchups.append({
            "title": f"{away_team} Pass Offense vs {home_team} Pass Defense",
            "home_val": f"{h['def_pass_yds_allowed']:.0f} allowed/season",
            "away_val": f"{a['off_pass_yds']:.0f} pass yds/season",
            "edge": away_team if a["off_pass_yds"] > h["def_pass_yds_allowed"] else home_team,
            "recommendation": (
                f"{'Expect {away_team} to air it out — play deep coverage' if a['off_pass_yds'] > h['def_pass_yds_allowed'] else 'Press and blitz — their passing game is stoppable'}."
            ).format(away_team=away_team),
            "mismatch_score": mismatch,
        })

    # Away rush offense vs Home rush defense
    if "off_rush_yds" in agg.columns and "def_rush_yds_allowed" in agg.columns:
        a_rush_off_pct = rank("off_rush_yds").loc[agg["team"] == away_team].values[0]
        h_rush_def_pct = rank("def_rush_yds_allowed", ascending=True).loc[agg["team"] == home_team].values[0]
        mismatch = a_rush_off_pct + (1 - h_rush_def_pct)
        matchups.append({
            "title": f"{away_team} Run Game vs {home_team} Rush Defense",
            "home_val": f"{h['def_rush_yds_allowed']:.0f} allowed/season",
            "away_val": f"{a['off_rush_yds']:.0f} rush yds/season",
            "edge": away_team if a["off_rush_yds"] > h["def_rush_yds_allowed"] else home_team,
            "recommendation": (
                f"{'Stack the box — {away_team} will try to run' if a['off_rush_yds'] > h['def_rush_yds_allowed'] else 'Their run game will struggle — force {away_team} to pass'}."
            ).format(away_team=away_team),
            "mismatch_score": mismatch,
        })

    # Turnover margin matchup
    if "turnover_margin" in agg.columns:
        h_tm = h.get("turnover_margin", 0)
        a_tm = a.get("turnover_margin", 0)
        mismatch = abs(h_tm - a_tm)
        if mismatch > 2:
            better = home_team if h_tm > a_tm else away_team
            matchups.append({
                "title": "Turnover Battle",
                "home_val": f"{home_team} margin: {h_tm:+.0f}",
                "away_val": f"{away_team} margin: {a_tm:+.0f}",
                "edge": better,
                "recommendation": (
                    f"{better} has a significant turnover margin advantage ({abs(h_tm - a_tm):.0f}). "
                    f"Ball security will be decisive in this game."
                ),
                "mismatch_score": mismatch * 10,
            })

    matchups.sort(key=lambda x: x["mismatch_score"], reverse=True)
    return matchups[:4]
