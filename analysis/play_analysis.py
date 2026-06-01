"""
Play-level situational analysis engine.
Ingests a tagged play-by-play Excel/CSV and answers:
  - What does a team do on a given down & distance?
  - What formations/personnel do they favour in each situation?
  - What are their run tendencies by direction and field zone?
  - What are their pass tendencies by depth and coverage situation?
  - What is their success rate and efficiency by play type / formation?
  - How does game situation (leading/trailing) shape their play-calling?
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
    "play_#": "play_num", "play #": "play_num",
    # formation
    "form": "formation", "off formation": "formation",
    # personnel
    "pers": "personnel", "personnel group": "personnel",
    # direction
    "dir": "direction", "run dir": "direction", "run direction": "direction",
    # pass depth
    "pass_depth": "pass_depth", "depth": "pass_depth",
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
    # game situation
    "game situation": "game_situation", "situation": "game_situation",
    "score_situation": "game_situation",
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

    # field zone from yard_line (yards from own end zone: 1=own goal, 99=opponent's 1-yd line)
    # Red zone = yard_line >= 90 (inside opponent's 10)
    if "yard_line" in df.columns:
        df["yard_line"] = pd.to_numeric(df["yard_line"], errors="coerce")
        df["field_zone"] = pd.cut(
            df["yard_line"],
            bins=[-1, 25, 50, 75, 89, 100],
            labels=["Own Territory (1-25)", "Own Half (26-50)",
                    "Opp Half (51-75)", "Scoring Zone (76-89)", "Red Zone (90+)"],
        )

    # normalise play_type to Run / Pass / Other
    if "play_type" in df.columns:
        df["play_type"] = df["play_type"].astype(str).str.strip().str.title()
        run_keywords = ["run", "rush", "draw", "sweep", "option", "qb sneak", "kneel"]
        pass_keywords = ["pass", "screen", "play action", "pa", "scramble", "throw"]
        play_action_keywords = ["play action", "play-action", "pa pass", "pa run", "playaction"]

        def classify(p):
            pl = p.lower()
            if any(k in pl for k in run_keywords):
                return "Run"
            if any(k in pl for k in pass_keywords):
                return "Pass"
            return p  # keep original if unrecognised

        def is_play_action(p):
            pl = p.lower()
            return any(k in pl for k in play_action_keywords)

        df["play_category"] = df["play_type"].apply(classify)
        df["is_play_action"] = df["play_type"].apply(is_play_action)

    # gain
    if "gain" in df.columns:
        df["gain"] = pd.to_numeric(df["gain"], errors="coerce")

    # success flag — standard football success rate definition
    # 1st down: gain ≥ 40% of distance needed; 2nd down: gain ≥ 50% distance; 3rd/4th: convert
    def compute_success(row):
        try:
            gain     = float(row.get("gain", 0) or 0)
            down     = int(row.get("down", 1) or 1)
            distance = float(row.get("distance", 10) or 10)
            result   = str(row.get("result", "")).lower()
            if "touchdown" in result:
                return True
            if down == 1:
                return gain >= distance * 0.40
            elif down == 2:
                return gain >= distance * 0.50
            else:  # 3rd or 4th
                return "first down" in result or gain >= distance
        except Exception:
            return False

    if "gain" in df.columns and "down" in df.columns:
        df["success"] = df.apply(compute_success, axis=1)

    return df


def _repair_excel_bytes(data: bytes) -> bytes:
    """
    Fix corrupted OUA coach Excel files where cells have t='s' (shared-string)
    type but store a literal value like '0,1,3' that isn't a valid integer index.
    Strategy: add the literal string to the shared-strings table, then replace
    the cell value with the correct integer index.
    """
    import zipfile, io, re

    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        names = zin.namelist()
        files = {n: zin.read(n) for n in names}

    ss_path = "xl/sharedStrings.xml"
    ws_path = "xl/worksheets/sheet1.xml"
    if ss_path not in files or ws_path not in files:
        return data  # can't repair, return as-is

    ss = files[ss_path].decode("utf-8")
    ws = files[ws_path].decode("utf-8")

    # Find numeric cells (no t= attr) whose <v> contains commas — invalid for int()
    bad_vals = set(re.findall(
        r'<c r="[^"]+">(?:\s*<[^/v][^>]*>)*\s*<v>([^<]*,[^<]*)</v>', ws
    ))
    if not bad_vals:
        return data  # nothing to fix

    # Add each bad value to the shared-strings table and remap cell to t="s"
    existing = re.findall(r'<si>', ss)
    next_idx = len(existing)
    replacements = {}
    for val in sorted(bad_vals):
        replacements[val] = next_idx
        ss = ss.replace("</sst>", f"  <si><t>{val}</t></si>\n</sst>")
        ss = re.sub(r'(count=")(\d+)(")', lambda m: m.group(1) + str(int(m.group(2)) + 1) + m.group(3), ss, count=1)
        ss = re.sub(r'(uniqueCount=")(\d+)(")', lambda m: m.group(1) + str(int(m.group(2)) + 1) + m.group(3), ss, count=1)
        next_idx += 1

    for val, idx in replacements.items():
        # Replace: <c r="XX99">\n  <v>0,1,3</v>  →  <c r="XX99" t="s">\n  <v>278</v>
        ws = re.sub(
            rf'(<c r="[^"]+")(\s*>)(\s*<v>){re.escape(val)}(</v>)',
            lambda m, i=idx: m.group(1) + ' t="s"' + m.group(2) + m.group(3) + str(i) + m.group(4),
            ws,
        )

    files[ss_path] = ss.encode("utf-8")
    files[ws_path] = ws.encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in names:
            zout.writestr(name, files[name])
    return buf.getvalue()


def load_play_file(uploaded_file) -> pd.DataFrame:
    """Load an uploaded Excel or CSV file and normalise it."""
    import io
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        raw = uploaded_file.read()
        try:
            df = pd.read_excel(io.BytesIO(raw))
        except (ValueError, Exception):
            # Attempt to repair corrupted shared-string references
            fixed = _repair_excel_bytes(raw)
            df = pd.read_excel(io.BytesIO(fixed))
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
    result = runs["direction"].value_counts(normalize=True).mul(100).round(1).reset_index()
    result.columns = ["direction", "pct"]
    return result


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
    rz = df[df["field_zone"].astype(str).str.startswith("Red Zone", na=False)]
    if rz.empty:
        return pd.DataFrame()
    result = rz["play_category"].value_counts(normalize=True).mul(100).round(1).reset_index()
    result.columns = ["play_type", "pct"]
    return result


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


# ── NEW: Run Game functions ────────────────────────────────────────────────────

def direction_by_personnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each personnel grouping, what % of runs go in each direction?
    Returns: personnel, direction, count, pct
    """
    if "direction" not in df.columns or "personnel" not in df.columns or df.empty:
        return pd.DataFrame()
    runs = df[(df["play_category"] == "Run") & (df["direction"].astype(str).str.strip() != "")]
    if runs.empty:
        return pd.DataFrame()
    t = runs.groupby(["personnel", "direction"]).size().reset_index(name="count")
    total = t.groupby("personnel")["count"].transform("sum")
    t["pct"] = (t["count"] / total * 100).round(1)
    return t.sort_values(["personnel", "count"], ascending=[True, False])


def hash_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run/Pass split by hash mark (Left / Middle / Right).
    Tells the defense whether they align or attack differently by hash.
    """
    if "hash" not in df.columns or "play_category" not in df.columns or df.empty:
        return pd.DataFrame()
    t = df.groupby(["hash", "play_category"]).size().reset_index(name="count")
    total = t.groupby("hash")["count"].transform("sum")
    t["pct"] = (t["count"] / total * 100).round(1)
    return t


def run_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Success rate per run direction.
    Success on run = meets the standard down-specific threshold.
    """
    if "direction" not in df.columns or "success" not in df.columns or df.empty:
        return pd.DataFrame()
    runs = df[(df["play_category"] == "Run") & (df["direction"].astype(str).str.strip() != "")]
    if runs.empty:
        return pd.DataFrame()
    t = runs.groupby("direction")["success"].agg(
        success_rate=lambda x: round(x.mean() * 100, 1),
        plays="count"
    ).reset_index().sort_values("success_rate", ascending=False)
    return t


# ── NEW: Pass Game functions ───────────────────────────────────────────────────

def pass_depth_by_situation(df: pd.DataFrame) -> pd.DataFrame:
    """
    On pass plays, what depth do they throw to, broken down by down × dist_bucket?
    Requires a Pass_Depth column: Short / Intermediate / Deep.
    """
    if "pass_depth" not in df.columns or "play_category" not in df.columns or df.empty:
        return pd.DataFrame()
    passes = df[(df["play_category"] == "Pass") & (df["pass_depth"].astype(str).str.strip() != "")]
    if passes.empty:
        return pd.DataFrame()

    if "dist_bucket" in passes.columns and "down" in passes.columns:
        t = passes.groupby(["down", "dist_bucket", "pass_depth"]).size().reset_index(name="count")
        total = t.groupby(["down", "dist_bucket"])["count"].transform("sum")
        t["pct"] = (t["count"] / total * 100).round(1)
    else:
        t = passes["pass_depth"].value_counts(normalize=True).mul(100).round(1).reset_index()
        t.columns = ["pass_depth", "pct"]
    return t


def play_action_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """
    How often do they use play action, and on which downs?
    Returns a summary of: all pass plays where play action = True.
    """
    if "is_play_action" not in df.columns or df.empty:
        return pd.DataFrame()
    passes = df[df["play_category"] == "Pass"]
    if passes.empty:
        return pd.DataFrame()

    # overall rate
    overall_rate = round(passes["is_play_action"].mean() * 100, 1)

    if "down" not in passes.columns:
        return pd.DataFrame([{"down": "All", "play_action_pct": overall_rate, "plays": len(passes)}])

    t = passes.groupby("down")["is_play_action"].agg(
        play_action_pct=lambda x: round(x.mean() * 100, 1),
        plays="count"
    ).reset_index()
    # append overall row
    overall_row = pd.DataFrame([{"down": "All", "play_action_pct": overall_rate, "plays": len(passes)}])
    return pd.concat([overall_row, t], ignore_index=True)


def pass_success_by_depth(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each pass depth zone, what is the success rate?
    """
    if "pass_depth" not in df.columns or "success" not in df.columns or df.empty:
        return pd.DataFrame()
    passes = df[(df["play_category"] == "Pass") & (df["pass_depth"].astype(str).str.strip() != "")]
    if passes.empty:
        return pd.DataFrame()
    t = passes.groupby("pass_depth")["success"].agg(
        success_rate=lambda x: round(x.mean() * 100, 1),
        plays="count"
    ).reset_index().sort_values("success_rate", ascending=False)
    return t


def completion_rate_by_depth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer completion rate from the result column.
    A completion = result contains 'complete', 'first down', 'touchdown', 'big play'.
    """
    if "pass_depth" not in df.columns or "result" not in df.columns or df.empty:
        return pd.DataFrame()
    passes = df[(df["play_category"] == "Pass") & (df["pass_depth"].astype(str).str.strip() != "")]
    if passes.empty:
        return pd.DataFrame()
    complete_keywords = ["complete", "first down", "touchdown", "big play", "gain"]

    def is_complete(r):
        return any(k in str(r).lower() for k in complete_keywords)

    passes = passes.copy()
    passes["completed"] = passes["result"].apply(is_complete)
    t = passes.groupby("pass_depth")["completed"].agg(
        completion_pct=lambda x: round(x.mean() * 100, 1),
        plays="count"
    ).reset_index().sort_values("completion_pct", ascending=False)
    return t


# ── NEW: Situational functions ─────────────────────────────────────────────────

def third_down_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    On 3rd down: what do they call by distance bucket, and what's the conversion rate?
    Conversion = result contains 'first down' or 'touchdown'.
    """
    if "down" not in df.columns or df.empty:
        return pd.DataFrame()
    third = df[df["down"] == 3]
    if third.empty:
        return pd.DataFrame()

    conversion_keywords = ["first down", "touchdown", "big play", "conversion"]

    def is_converted(r):
        return any(k in str(r).lower() for k in conversion_keywords)

    third = third.copy()
    third["converted"] = third["result"].apply(is_converted) if "result" in third.columns else False

    rows = []
    for bucket in third["dist_bucket"].dropna().unique() if "dist_bucket" in third.columns else ["All"]:
        sub = third[third["dist_bucket"] == bucket] if bucket != "All" else third
        if sub.empty:
            continue
        rp_split = sub["play_category"].value_counts(normalize=True).mul(100).round(1).to_dict() if "play_category" in sub.columns else {}
        rows.append({
            "dist_bucket": bucket,
            "plays": len(sub),
            "run_pct": rp_split.get("Run", 0.0),
            "pass_pct": rp_split.get("Pass", 0.0),
            "conversion_rate": round(sub["converted"].mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def situation_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """
    How does play-calling shift when they're Leading / Trailing / Close?
    Requires a 'game_situation' column with values like 'Leading', 'Trailing', 'Close'.
    """
    if "game_situation" not in df.columns or "play_category" not in df.columns or df.empty:
        return pd.DataFrame()
    t = df.groupby(["game_situation", "play_category"]).size().reset_index(name="count")
    total = t.groupby("game_situation")["count"].transform("sum")
    t["pct"] = (t["count"] / total * 100).round(1)
    return t


def redzone_tendencies_detail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full red zone breakdown: play type, formation, personnel, success rate.
    """
    if "field_zone" not in df.columns:
        return pd.DataFrame()
    rz = df[df["field_zone"].astype(str).str.startswith("Red Zone", na=False)]
    if rz.empty:
        return pd.DataFrame()

    rows = []
    for cat in ["Run", "Pass"]:
        sub = rz[rz["play_category"] == cat] if "play_category" in rz.columns else pd.DataFrame()
        if sub.empty:
            continue
        top_form = sub["formation"].value_counts().index[0] if "formation" in sub.columns and not sub["formation"].dropna().empty else "N/A"
        top_pers = sub["personnel"].value_counts().index[0] if "personnel" in sub.columns and not sub["personnel"].dropna().empty else "N/A"
        succ = round(sub["success"].mean() * 100, 1) if "success" in sub.columns else None
        rows.append({
            "play_type": cat,
            "plays": len(sub),
            "pct": round(len(sub) / len(rz) * 100, 1),
            "top_formation": top_form,
            "top_personnel": top_pers,
            "success_rate": succ,
        })
    return pd.DataFrame(rows)


# ── NEW: Efficiency functions ──────────────────────────────────────────────────

def success_rate_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    % of plays that were 'successful' (meet down-specific gain threshold) grouped by any column.
    """
    if group_col not in df.columns or "success" not in df.columns or df.empty:
        return pd.DataFrame()
    t = df.groupby(group_col)["success"].agg(
        success_rate=lambda x: round(x.mean() * 100, 1),
        plays="count",
    ).reset_index().sort_values("success_rate", ascending=False)
    return t


def yards_per_play_by_situation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average yards per play split by: down, dist_bucket, play_category.
    """
    if "gain" not in df.columns or df.empty:
        return pd.DataFrame()
    groups = [c for c in ["down", "dist_bucket", "play_category"] if c in df.columns]
    if not groups:
        return pd.DataFrame()
    return (
        df.groupby(groups)["gain"]
        .agg(avg_gain="mean", plays="count")
        .round(2)
        .reset_index()
        .sort_values("avg_gain", ascending=False)
    )


def explosive_plays(df: pd.DataFrame, run_threshold: int = 10, pass_threshold: int = 20) -> pd.DataFrame:
    """
    Explosive play breakdown: runs ≥ run_threshold yards, passes ≥ pass_threshold yards.
    Shows which formations/personnel generate chunk plays.
    """
    if "gain" not in df.columns or "play_category" not in df.columns or df.empty:
        return pd.DataFrame()
    mask = (
        ((df["play_category"] == "Run")  & (df["gain"] >= run_threshold)) |
        ((df["play_category"] == "Pass") & (df["gain"] >= pass_threshold))
    )
    exp = df[mask].copy()
    if exp.empty:
        return pd.DataFrame()
    rows = []
    for cat in ["Run", "Pass"]:
        sub = exp[exp["play_category"] == cat]
        if sub.empty:
            continue
        top_form = sub["formation"].value_counts().index[0] if "formation" in sub.columns and not sub["formation"].dropna().empty else "N/A"
        top_dir  = sub["direction"].value_counts().index[0] if cat == "Run" and "direction" in sub.columns and not sub["direction"].dropna().empty else ""
        top_dep  = sub["pass_depth"].value_counts().index[0] if cat == "Pass" and "pass_depth" in sub.columns and not sub["pass_depth"].dropna().empty else ""
        rows.append({
            "play_type":      cat,
            "count":          len(sub),
            "avg_gain":       round(sub["gain"].mean(), 1),
            "top_formation":  top_form,
            "top_direction":  top_dir if cat == "Run" else top_dep,
        })
    return pd.DataFrame(rows)
