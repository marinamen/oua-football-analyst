"""
Run this once to generate the sample play-by-play template Excel.
python3 data/manual/play_template.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

N = 80  # ~80 offensive plays per game

downs       = np.random.choice([1,2,3,4], N, p=[0.40,0.32,0.24,0.04])
distances   = np.where(downs == 1, 10,
              np.where(downs == 2, np.random.randint(1,11,N),
              np.where(downs == 3, np.random.randint(1,15,N),
                                   np.random.randint(1,5,N))))

formations  = np.random.choice(
    ["Shotgun", "I-Form", "Pistol", "Singleback", "Wildcat"],
    N, p=[0.35, 0.30, 0.15, 0.15, 0.05]
)
personnel   = np.random.choice(
    ["11 (1 RB, 1 TE)", "12 (1 RB, 2 TE)", "21 (2 RB, 1 TE)", "22 (2 RB, 2 TE)", "10 (1 RB, 0 TE)"],
    N, p=[0.40, 0.25, 0.20, 0.10, 0.05]
)
play_types  = np.where(
    formations == "Shotgun",
    np.random.choice(["Pass", "Run", "Screen", "Draw"], N, p=[0.65,0.20,0.10,0.05]),
    np.random.choice(["Run", "Pass", "Play Action"], N, p=[0.55,0.30,0.15])
)
directions  = np.where(
    play_types == "Run",
    np.random.choice(["Left", "Right", "Middle", "Outside Left", "Outside Right"], N, p=[0.25,0.25,0.30,0.10,0.10]),
    ""
)
motions     = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])
quarters    = np.random.choice([1,2,3,4], N, p=[0.25,0.25,0.25,0.25])
yard_lines  = np.random.randint(1, 100, N)
gains       = np.random.normal(5, 7, N).astype(int)
gains       = np.clip(gains, -10, 40)

results = []
for g, pt in zip(gains, play_types):
    if g >= 20:
        results.append("Big Play")
    elif "Pass" in pt and np.random.rand() < 0.15:
        results.append("Incomplete")
    elif np.random.rand() < 0.03:
        results.append("Turnover")
    elif g == 0:
        results.append("No Gain")
    else:
        results.append("Complete" if "Pass" in pt else "Gain")

df = pd.DataFrame({
    "Quarter":    quarters,
    "Down":       downs,
    "Distance":   distances,
    "Yard_Line":  yard_lines,
    "Formation":  formations,
    "Personnel":  personnel,
    "Motion":     motions,
    "Play_Type":  play_types,
    "Direction":  directions,
    "Gain":       gains,
    "Result":     results,
})

out = Path(__file__).parent / "play_template.xlsx"
df.to_excel(out, index=False)
print(f"Saved {len(df)}-play template → {out}")
