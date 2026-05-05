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
    {"game_id": 1, "opponent": "Toronto",  "date": "2024-09-07", "location": "Away",  "arc": "win_lead"},
    {"game_id": 2, "opponent": "Ottawa",   "date": "2024-09-14", "location": "Home",  "arc": "lose_lead"},
    {"game_id": 3, "opponent": "McMaster", "date": "2024-09-21", "location": "Home",  "arc": "comeback"},
]

FORMATIONS  = ["I-Form", "Shotgun", "Pistol", "Singleback", "Wildcat", "I-Form Tight"]
PERSONNEL   = ["21 (2RB, 1TE)", "12 (1RB, 2TE)", "11 (1RB, 1TE)", "22 (2RB, 2TE)", "10 (1RB, 0TE)"]
DIRECTIONS  = ["Inside Left", "Inside Right", "Off Tackle Left", "Off Tackle Right",
               "Outside Left", "Outside Right", "Up the Middle"]
PASS_DEPTHS = ["Short (0-5)", "Intermediate (6-15)", "Deep (16+)"]
SITUATIONS  = ["Leading", "Close", "Trailing"]


def formation_probs(down, distance):
    """Queen's tendencies — heavy run team, I-Form dominant."""
    if down == 1:
        return [0.40, 0.10, 0.15, 0.15, 0.05, 0.15]
    elif down == 2 and distance <= 3:
        return [0.45, 0.05, 0.10, 0.15, 0.10, 0.15]
    elif down == 2 and distance > 6:
        return [0.20, 0.35, 0.20, 0.10, 0.00, 0.15]
    elif down == 3 and distance <= 2:
        return [0.35, 0.05, 0.10, 0.10, 0.20, 0.20]
    elif down == 3 and distance > 5:
        return [0.05, 0.60, 0.20, 0.10, 0.00, 0.05]
    else:
        return [0.30, 0.20, 0.15, 0.15, 0.05, 0.15]


def personnel_probs(formation):
    if "I-Form" in formation or formation == "Singleback":
        return [0.50, 0.25, 0.10, 0.12, 0.03]
    elif formation == "Shotgun":
        return [0.10, 0.15, 0.55, 0.05, 0.15]
    elif formation == "Wildcat":
        return [0.30, 0.10, 0.05, 0.50, 0.05]
    else:
        return [0.30, 0.30, 0.20, 0.15, 0.05]


def play_type_probs(formation, down, distance, yard_line, situation):
    """Queen's run-heavy — even more run in short yardage, red zone, and when leading."""
    base_run = 0.65
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
    if yard_line >= 90:  # red zone — inside opponent's 10
        base_run = min(0.80, base_run + 0.10)
    # game situation adjustments
    if situation == "Leading":
        base_run = min(0.88, base_run + 0.08)  # bleed clock when ahead
    elif situation == "Trailing":
        base_run = max(0.15, base_run - 0.20)  # throw more when behind
    return ["Run", "Pass"], [base_run, 1 - base_run]


def motion_prob(formation, play_type):
    """Queen's motions TE pre-snap frequently — especially before runs."""
    if play_type == "Run" and "I-Form" in formation:
        return 0.45
    elif play_type == "Run":
        return 0.25
    else:
        return 0.15


def play_action_prob(formation, down, situation):
    """Play action is most effective on 1st down out of I-Form."""
    if "I-Form" in formation and down == 1:
        return 0.30
    elif "I-Form" in formation and down == 2:
        return 0.20
    elif situation == "Leading" and "I-Form" in formation:
        return 0.25
    else:
        return 0.08


def gain_yards(play_type, formation, down, distance, pass_depth):
    if play_type == "Run":
        if "I-Form" in formation:
            mu = 5.2
        elif "Wildcat" in formation:
            mu = 6.8
        else:
            mu = 4.1
        return int(np.clip(np.random.normal(mu, 4.5), -5, 25))
    else:
        if "Short" in pass_depth:
            mu = 5
        elif "Intermediate" in pass_depth:
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
    n_plays = np.random.randint(70, 82)
    down     = 1
    distance = 10
    # yard_line = yards from own end zone (25 = own 25, 99 = opponent's 1)
    yard_line = np.random.randint(20, 40)
    drive = 1

    # game situation arc: pre-assigned per game for realistic coverage
    arc = game["arc"]
    # each arc defines how score_diff evolves over 4 quarters
    arc_map = {
        "win_lead":  [ 3,  7, 10, 14],   # Queen's leads all game
        "close":     [ 0,  3, -3,  0],   # tight throughout
        "lose_lead": [-3, -7, -10, -14], # they get blown out
        "comeback":  [-7, -7,  0,  7],   # come from behind
    }

    def situation_from_score(diff):
        if diff > 7:   return "Leading"
        if diff < -7:  return "Trailing"
        return "Close"

    quarter_diffs = arc_map[arc]

    for i in range(n_plays):
        # derive situation from arc
        q_idx = min(3, int((i / n_plays) * 4))
        situation = situation_from_score(quarter_diffs[q_idx])

        formation  = np.random.choice(FORMATIONS, p=formation_probs(down, distance))
        personnel  = np.random.choice(PERSONNEL,  p=personnel_probs(formation))
        pt_choices, pt_probs = play_type_probs(formation, down, distance, yard_line, situation)
        play_type  = np.random.choice(pt_choices, p=pt_probs)

        # play action flag (only on pass plays where motion/I-Form could sell it)
        use_pa = False
        if play_type == "Pass" and np.random.rand() < play_action_prob(formation, down, situation):
            use_pa = True
            play_type_label = "Play Action Pass"
        else:
            play_type_label = play_type

        motion  = "Yes" if np.random.rand() < motion_prob(formation, play_type) else "No"
        quarter = min(4, max(1, int((i / n_plays) * 4) + 1))

        direction  = ""
        pass_depth = ""
        if play_type == "Run":
            dir_probs = [0.28, 0.25, 0.18, 0.14, 0.06, 0.05, 0.04]
            direction = np.random.choice(DIRECTIONS, p=dir_probs)
        else:
            # trailing teams throw deeper, leading teams throw short
            if situation == "Trailing":
                pass_depth = np.random.choice(PASS_DEPTHS, p=[0.25, 0.40, 0.35])
            elif situation == "Leading":
                pass_depth = np.random.choice(PASS_DEPTHS, p=[0.60, 0.30, 0.10])
            else:
                pass_depth = np.random.choice(PASS_DEPTHS, p=[0.45, 0.35, 0.20])

        gain = gain_yards(play_type, formation, down, distance, pass_depth)
        res  = result_label(play_type, gain, down, distance)

        hash_mark = np.random.choice(["Left", "Middle", "Right"], p=[0.35, 0.30, 0.35])

        rows.append({
            "Game_ID":        game["game_id"],
            "Date":           game["date"],
            "Opponent":       game["opponent"],
            "Location":       game["location"],
            "Drive":          drive,
            "Play_#":         play_num,
            "Quarter":        quarter,
            "Down":           down,
            "Distance":       distance,
            "Yard_Line":      min(99, max(1, yard_line)),
            "Hash":           hash_mark,
            "Formation":      formation,
            "Personnel":      personnel,
            "Motion":         motion,
            "Play_Type":      play_type_label,
            "Direction":      direction,
            "Pass_Depth":     pass_depth,
            "Gain":           gain,
            "Result":         res,
            "Game_Situation": situation,
        })

        play_num += 1

        # (score is tracked via the arc, no per-play bookkeeping needed)

        # advance down & distance
        # yard_line increases as offense gains ground (toward opponent's end zone)
        if res in ("First Down", "Touchdown", "Conversion", "Big Play"):
            down, distance = 1, 10
            yard_line = min(99, yard_line + max(gain, 10))
            if res == "Touchdown":
                yard_line = np.random.randint(20, 40)  # new drive, kick off from own ~25
                drive += 1
        elif res == "Interception":
            down, distance = 1, 10
            yard_line = np.random.randint(20, 40)
            drive += 1
            pass  # score tracked via arc
        else:
            down += 1
            distance = max(1, distance - gain)
            yard_line = min(99, yard_line + gain)
            if down > 4:  # turnover on downs
                down, distance = 1, 10
                yard_line = np.random.randint(20, 40)
                drive += 1

df = pd.DataFrame(rows)
out = Path(__file__).parent / "queens_plays_3games.xlsx"
df.to_excel(out, index=False)
print(f"Generated {len(df)} plays across {len(GAMES)} games → {out}")
print(f"\nPlay type breakdown:\n{df['Play_Type'].value_counts()}")
print(f"\nFormation breakdown:\n{df['Formation'].value_counts()}")
print(f"\nGame situation breakdown:\n{df['Game_Situation'].value_counts()}")
print(f"\nTop run directions:\n{df[df['Play_Type']=='Run']['Direction'].value_counts().head()}")
