"""
Generates realistic fake play-by-play data as if tagged by a coaching staff.
Simulates 3 games of Queen's offensive plays vs Toronto's defence.
Run with: python3 data/manual/generate_fake_plays.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(99)

# Queen's is a run-heavy team — we'll bake that into the probabilities
# They favour I-Form and 21 personnel, run between the tackles, motion TE pre-snap

GAMES = [
    {"game_id": 1, "opponent": "Toronto",  "date": "2024-09-07", "location": "Away"},
    {"game_id": 2, "opponent": "Ottawa",   "date": "2024-09-14", "location": "Home"},
    {"game_id": 3, "opponent": "McMaster", "date": "2024-09-21", "location": "Home"},
]

FORMATIONS = ["I-Form", "Shotgun", "Pistol", "Singleback", "Wildcat", "I-Form Tight"]
PERSONNEL  = ["21 (2RB, 1TE)", "12 (1RB, 2TE)", "11 (1RB, 1TE)", "22 (2RB, 2TE)", "10 (1RB, 0TE)"]
DIRECTIONS = ["Inside Left", "Inside Right", "Off Tackle Left", "Off Tackle Right",
              "Outside Left", "Outside Right", "Up the Middle"]
PASS_DEPTHS = ["Short (0-5)", "Intermediate (6-15)", "Deep (16+)"]
RESULTS_RUN  = ["Gain", "Gain", "Gain", "No Gain", "Loss", "First Down", "Touchdown"]
RESULTS_PASS = ["Complete", "Complete", "Incomplete", "First Down", "Touchdown",
                "Interception", "Scramble"]

def formation_probs(down, distance):
    """Queen's tendencies — heavy run team, I-Form dominant."""
    if down == 1:
        return [0.40, 0.10, 0.15, 0.15, 0.05, 0.15]  # I-Form heavy
    elif down == 2 and distance <= 3:
        return [0.45, 0.05, 0.10, 0.15, 0.10, 0.15]  # even more I-Form short yardage
    elif down == 2 and distance > 6:
        return [0.20, 0.35, 0.20, 0.10, 0.00, 0.15]  # more shotgun on 2nd & long
    elif down == 3 and distance <= 2:
        return [0.35, 0.05, 0.10, 0.10, 0.20, 0.20]  # wildcat/I-Form short yardage
    elif down == 3 and distance > 5:
        return [0.05, 0.60, 0.20, 0.10, 0.00, 0.05]  # shotgun dominant 3rd & long
    else:
        return [0.30, 0.20, 0.15, 0.15, 0.05, 0.15]

def personnel_probs(formation):
    if "I-Form" in formation or formation == "Singleback":
        return [0.50, 0.25, 0.10, 0.12, 0.03]  # 21 dominant
    elif formation == "Shotgun":
        return [0.10, 0.15, 0.55, 0.05, 0.15]  # 11 dominant
    elif formation == "Wildcat":
        return [0.30, 0.10, 0.05, 0.50, 0.05]  # 22 or 21
    else:
        return [0.30, 0.30, 0.20, 0.15, 0.05]

def play_type_probs(formation, down, distance, yard_line):
    """Queen's run-heavy — even more run in short yardage and red zone."""
    base_run = 0.65  # Queen's overall tendency
    if "Shotgun" in formation:
        base_run = 0.25
    elif "Wildcat" in formation:
        base_run = 0.90
    elif "I-Form" in formation:
        base_run = 0.78
    if down == 3 and distance > 5:
        base_run = max(0.10, base_run - 0.50)
    if down == 1:
        base_run = min(0.85, base_run + 0.05)
    if yard_line <= 10:  # red zone
        base_run = min(0.80, base_run + 0.10)
    return ["Run", "Pass"] , [base_run, 1 - base_run]

def motion_prob(formation, play_type):
    """Queen's motions TE pre-snap frequently — especially before runs."""
    if play_type == "Run" and "I-Form" in formation:
        return 0.45
    elif play_type == "Run":
        return 0.25
    else:
        return 0.15

def gain_yards(play_type, formation, down, distance):
    if play_type == "Run":
        if "I-Form" in formation:
            mu = 5.2
        elif "Wildcat" in formation:
            mu = 6.8
        else:
            mu = 4.1
        return int(np.clip(np.random.normal(mu, 4.5), -5, 25))
    else:
        depth = np.random.choice(PASS_DEPTHS, p=[0.45, 0.35, 0.20])
        if "Short" in depth:
            mu = 5
        elif "Intermediate" in depth:
            mu = 11
        else:
            mu = 22
        complete = np.random.rand() < 0.62
        return int(np.clip(np.random.normal(mu, 4), -2, 40)) if complete else 0

def result_label(play_type, gain, down, distance):
    if gain >= 20:
        return "Big Play"
    if play_type == "Pass" and gain == 0:
        if np.random.rand() < 0.08:
            return "Interception"
        return "Incomplete"
    if gain < 0:
        return "Loss"
    if gain == 0:
        return "No Gain"
    if gain >= distance:
        if np.random.rand() < 0.04:
            return "Touchdown"
        return "First Down" if down < 4 else "Conversion"
    return "Gain"

rows = []
play_num = 1

for game in GAMES:
    # simulate ~75 offensive plays per game
    n_plays = np.random.randint(70, 82)
    down = 1
    distance = 10
    yard_line = np.random.randint(25, 45)  # start of each drive
    drive = 1

    for _ in range(n_plays):
        formation = np.random.choice(FORMATIONS, p=formation_probs(down, distance))
        personnel  = np.random.choice(PERSONNEL,  p=personnel_probs(formation))
        pt_choices, pt_probs = play_type_probs(formation, down, distance, yard_line)
        play_type  = np.random.choice(pt_choices, p=pt_probs)
        motion     = "Yes" if np.random.rand() < motion_prob(formation, play_type) else "No"
        quarter    = min(4, max(1, int(((_ / n_plays) * 4) + 1)))
        gain       = gain_yards(play_type, formation, down, distance)
        res        = result_label(play_type, gain, down, distance)

        direction  = ""
        pass_depth = ""
        if play_type == "Run":
            # Queen's loves inside runs — weight accordingly
            dir_probs = [0.28, 0.25, 0.18, 0.14, 0.06, 0.05, 0.04]
            direction = np.random.choice(DIRECTIONS, p=dir_probs)
        else:
            pass_depth = np.random.choice(PASS_DEPTHS, p=[0.45, 0.35, 0.20])

        hash_mark = np.random.choice(["Left", "Middle", "Right"], p=[0.35, 0.30, 0.35])

        rows.append({
            "Game_ID":    game["game_id"],
            "Date":       game["date"],
            "Opponent":   game["opponent"],
            "Location":   game["location"],
            "Drive":      drive,
            "Play_#":     play_num,
            "Quarter":    quarter,
            "Down":       down,
            "Distance":   distance,
            "Yard_Line":  min(99, max(1, yard_line)),
            "Hash":       hash_mark,
            "Formation":  formation,
            "Personnel":  personnel,
            "Motion":     motion,
            "Play_Type":  play_type,
            "Direction":  direction,
            "Pass_Depth": pass_depth,
            "Gain":       gain,
            "Result":     res,
        })

        play_num += 1

        # update down & distance
        if res in ("First Down", "Touchdown", "Conversion", "Big Play"):
            down, distance = 1, 10
            yard_line = min(99, yard_line + max(gain, 10))
            if res == "Touchdown":
                yard_line = np.random.randint(20, 45)
                drive += 1
        elif res in ("Interception",):
            down, distance = 1, 10
            yard_line = np.random.randint(20, 45)
            drive += 1
        else:
            down += 1
            distance = max(1, distance - gain)
            yard_line = min(99, yard_line + gain)
            if down > 4:
                down, distance = 1, 10
                yard_line = np.random.randint(20, 45)
                drive += 1

df = pd.DataFrame(rows)
out = Path(__file__).parent / "queens_plays_3games.xlsx"
df.to_excel(out, index=False)
print(f"Generated {len(df)} plays across {len(GAMES)} games → {out}")
print(f"\nPlay type breakdown:\n{df['Play_Type'].value_counts()}")
print(f"\nFormation breakdown:\n{df['Formation'].value_counts()}")
print(f"\nTop run directions:\n{df[df['Play_Type']=='Run']['Direction'].value_counts().head()}")
