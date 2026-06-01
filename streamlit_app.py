"""
Varsity Blues Football Analytics
Built for the University of Toronto football program.
Run with: streamlit run streamlit_app.py
"""

from pathlib import Path
import hashlib
import hmac
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis.team_stats import (
    load_data, compute_team_aggregates, compute_sos_adjusted_aggregates,
    weakness_scores, season_trend
)
from analysis.play_analysis import (
    load_play_file, apply_filters, normalize_columns, derive_buckets,
    # overview
    run_pass_split, tendency_by_down_distance,
    # run game
    formation_tendency, personnel_tendency,
    direction_tendency, direction_by_personnel,
    hash_tendency, run_success_rate,
    avg_gain_by_situation,
    # pass game
    pass_depth_by_situation, play_action_tendency,
    completion_rate_by_depth, pass_success_by_depth,
    # situational
    redzone_tendencies, redzone_tendencies_detail,
    motion_tendency, third_down_breakdown, situation_tendency,
    # efficiency
    success_rate_by_group, yards_per_play_by_situation, explosive_plays,
)
from analysis.predictor import train, predict_matchup
from analysis.scouting import (
    win_condition_fingerprint, how_to_beat, momentum_score, matchup_exploiter
)
import datetime

TORONTO = "Toronto"
UFT_BLUE  = "#003E7E"
UFT_RED   = "#C8102E"

st.set_page_config(
    page_title="Varsity Blues Football Analytics",
    page_icon="🏈",
    layout="wide",
)

# ── Authentication — zero external dependencies ────────────────────────────────
def _secrets_configured() -> bool:
    try:
        return "auth" in st.secrets and "users" in st.secrets
    except Exception:
        return False

def _hash_password(salt: str, password: str) -> str:
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

def _check_credentials(username: str, password: str) -> str | None:
    """Return display name on success, None on failure."""
    try:
        salt  = st.secrets["auth"]["salt"]
        users = st.secrets["users"]
        if username not in users:
            return None
        stored = users[username]["password"]
        computed = _hash_password(salt, password)
        if hmac.compare_digest(computed, stored):
            return users[username]["name"]
        return None
    except Exception:
        return None

def _show_login_wall():
    st.markdown(
        "<h2 style='text-align:center;margin-top:3rem'>🏈 Varsity Blues Football Analytics</h2>"
        "<p style='text-align:center;color:#888'>University of Toronto staff only</p>",
        unsafe_allow_html=True,
    )
    col = st.columns([1, 1.2, 1])[1]
    with col:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="e.g. vblues")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")

        if submitted:
            name = _check_credentials(username.strip().lower(), password)
            if name:
                st.session_state["auth_user"]    = username.strip().lower()
                st.session_state["auth_name"]    = name
                st.session_state["auth_time"]    = time.time()
                st.rerun()
            else:
                st.error("Incorrect username or password.")

SESSION_EXPIRY = 60 * 60 * 24 * 7  # 7 days

if _secrets_configured():
    # check existing session
    logged_in = (
        "auth_user" in st.session_state
        and time.time() - st.session_state.get("auth_time", 0) < SESSION_EXPIRY
    )
    if not logged_in:
        _show_login_wall()
        st.stop()

    auth_name = st.session_state["auth_name"]

    # logout button lives at top of sidebar
    with st.sidebar:
        st.markdown(f"👤 **{auth_name}**")
        if st.button("Sign out", key="logout"):
            for k in ("auth_user", "auth_name", "auth_time"):
                st.session_state.pop(k, None)
            st.rerun()
        st.markdown("---")
else:
    auth_name = "Dev"  # no secrets → open access for local development

st.title("🏈 Varsity Blues Football Analytics")
st.caption("University of Toronto")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/04/Toronto_Varsity_Blues_logo.svg/200px-Toronto_Varsity_Blues_logo.svg.png", width=120)
    st.markdown("---")

    if st.button("Refresh Data from OUA", type="primary"):
        with st.spinner("Scraping oua.ca..."):
            try:
                from scraper.oua_scraper import scrape_season
                from scraper.stats_scraper import scrape_team_gamelog
                from scraper.coaches_scraper import scrape_team_coaches
                import time

                _TEAMS = ["carleton","guelph","laurier","mcmaster","ottawa",
                          "queens","toronto","waterloo","western","windsor","york"]
                RAW = Path(__file__).parent / "data" / "raw"

                all_games = []
                for yr in [2025, 2024, 2023, 2022]:
                    try:
                        all_games.append(scrape_season(yr)); time.sleep(0.5)
                    except Exception: pass
                if all_games:
                    pd.concat(all_games, ignore_index=True).to_csv(RAW / "all_games.csv", index=False)

                gl_rows = []
                for season in ["2025-26", "2024-25", "2023-24"]:
                    for t in _TEAMS:
                        try:
                            gl_rows.append(scrape_team_gamelog(t, season)); time.sleep(0.4)
                        except Exception: pass
                if gl_rows:
                    pd.concat(gl_rows, ignore_index=True).to_csv(RAW / "team_stats_all.csv", index=False)

                c_rows = []
                for season in ["2025-26", "2024-25"]:
                    for t in _TEAMS:
                        try:
                            row = scrape_team_coaches(t, season)
                            if row: c_rows.append(row)
                            time.sleep(0.4)
                        except Exception: pass
                if c_rows:
                    pd.DataFrame(c_rows).to_csv(RAW / "coaches_stats_all.csv", index=False)

                st.cache_data.clear()
                st.success("Done! Reload the page.")
            except Exception as e:
                st.error(f"Scrape failed: {e}")

    st.markdown("---")
    sos_on = st.toggle(
        "SOS-Adjusted Stats",
        value=False,
        help="Normalizes offensive stats by opponent defensive quality. "
             "York's numbers drop when adjusted for weak opponents.",
    )
    st.caption("SOS adjusted" if sos_on else "Raw stats")
    st.markdown("---")
    st.caption("Data: oua.ca")


# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    games, gamelog, coaches = load_data()
    agg_raw = compute_team_aggregates(coaches)
    agg_sos = compute_sos_adjusted_aggregates(gamelog, coaches)
    return games, gamelog, coaches, agg_raw, agg_sos

games, gamelog, coaches, agg_raw, agg_sos = get_data()
agg  = agg_sos if sos_on else agg_raw
weak = weakness_scores(agg)
opponents = sorted([t for t in agg["team"].unique() if t != TORONTO])
all_teams  = sorted(agg["team"].unique().tolist())

# ── Load 2026-27 Toronto schedule ─────────────────────────────────────────────
SCHEDULE_PATH = Path(__file__).parent / "data" / "manual" / "toronto_schedule_2026.csv"

@st.cache_data
def load_schedule():
    if SCHEDULE_PATH.exists():
        return pd.read_csv(SCHEDULE_PATH, parse_dates=["date"])
    return pd.DataFrame()

schedule_df = load_schedule()

# Find next upcoming game
today = datetime.date.today()
next_game = None
if not schedule_df.empty:
    upcoming = schedule_df[schedule_df["date"].dt.date >= today]
    if not upcoming.empty:
        next_game = upcoming.iloc[0]

# ── Next Game Banner ───────────────────────────────────────────────────────────
if next_game is not None:
    days_away = (next_game["date"].date() - today).days
    loc_icon  = "🏠" if next_game["location"] == "Home" else "✈️"
    time_str  = datetime.datetime.strptime(str(next_game["time_et"]), "%H:%M").strftime("%I:%M %p ET").lstrip("0")
    banner_bg = UFT_BLUE
    st.markdown(
        f"""
        <div style="background:{banner_bg};border-radius:10px;padding:18px 24px;margin-bottom:16px;display:flex;align-items:center;gap:32px;flex-wrap:wrap;">
          <div>
            <div style="color:#aac8f5;font-size:0.75rem;font-weight:600;letter-spacing:1px;text-transform:uppercase">Next Game - Week {int(next_game['week'])}</div>
            <div style="color:white;font-size:1.5rem;font-weight:700;margin-top:2px">🏈 {TORONTO} vs {next_game['opponent']}</div>
          </div>
          <div style="display:flex;gap:28px;flex-wrap:wrap">
            <div><div style="color:#aac8f5;font-size:0.7rem;letter-spacing:1px">DATE</div><div style="color:white;font-weight:600">{next_game['date'].strftime('%b %d, %Y')}</div></div>
            <div><div style="color:#aac8f5;font-size:0.7rem;letter-spacing:1px">TIME</div><div style="color:white;font-weight:600">{time_str}</div></div>
            <div><div style="color:#aac8f5;font-size:0.7rem;letter-spacing:1px">LOCATION</div><div style="color:white;font-weight:600">{loc_icon} {next_game['location']}</div></div>
            <div><div style="color:#aac8f5;font-size:0.7rem;letter-spacing:1px">DAYS AWAY</div><div style="color:white;font-weight:600">{days_away}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Global opponent selector (auto-selects next opponent) ─────────────────────
default_opp_idx = 0
if next_game is not None and next_game["opponent"] in opponents:
    default_opp_idx = opponents.index(next_game["opponent"])

st.markdown("### Select Upcoming Opponent")
opponent = st.selectbox("Opponent", opponents, index=default_opp_idx, label_visibility="collapsed", key="global_opp")
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab0, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📅 Schedule",
    "🎯 Opponent Breakdown",
    "📈 Toronto Trends",
    "📋 Game Plan",
    "🏈 Opponent Play Tendencies",
    "🔭 League Intel",
    "🗂 Raw Data",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 0 — Schedule
# ══════════════════════════════════════════════════════════════════════════════
with tab0:
    st.subheader("2026-27 Schedule")

    if schedule_df.empty:
        st.info("Schedule not loaded.")
    else:
        # Merge results from all_games.csv where available
        tor_results = pd.DataFrame()
        if not games.empty:
            tor_games = games[
                (games["season"] == 2026) &
                ((games["home_team"] == TORONTO) | (games["away_team"] == TORONTO))
            ].copy()
            if not tor_games.empty:
                def _result_row(r):
                    is_home = r["home_team"] == TORONTO
                    opp = r["away_team"] if is_home else r["home_team"]
                    tor_score = int(r["home_score"]) if is_home else int(r["away_score"])
                    opp_score = int(r["away_score"]) if is_home else int(r["home_score"])
                    result = "W" if tor_score > opp_score else "L"
                    return pd.Series({"opponent": opp, "result": result,
                                      "score": f"{tor_score}–{opp_score}"})
                tor_results = tor_games.apply(_result_row, axis=1)

        # Build display table
        rows = []
        for _, g in schedule_df.iterrows():
            is_next = (next_game is not None and g["date"] == next_game["date"]
                       and g["opponent"] == next_game["opponent"])
            loc_icon = "🏠 Home" if g["location"] == "Home" else "✈️ Away"
            time_fmt = datetime.datetime.strptime(str(g["time_et"]), "%H:%M").strftime("%I:%M %p ET").lstrip("0")
            result_str, score_str = "", ""
            if not tor_results.empty:
                match = tor_results[tor_results["opponent"] == g["opponent"]]
                if not match.empty:
                    result_str = match.iloc[0]["result"]
                    score_str  = match.iloc[0]["score"]

            rows.append({
                "Wk": int(g["week"]),
                "Date": g["date"].strftime("%a %b %d"),
                "Opponent": ("▶ " if is_next else "") + str(g["opponent"]),
                "Location": loc_icon,
                "Time": time_fmt,
                "Result": result_str,
                "Score": score_str,
            })

        sched_display = pd.DataFrame(rows)

        # Colour next game row
        def _style_sched(row):
            if row["Opponent"].startswith("▶"):
                return [f"background-color:{UFT_BLUE};color:white"] * len(row)
            elif row["Result"] == "W":
                return ["color:#1a7a4a;font-weight:600"] * len(row)
            elif row["Result"] == "L":
                return ["color:#c0392b;font-weight:600"] * len(row)
            return [""] * len(row)

        st.dataframe(
            sched_display.style.apply(_style_sched, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Season record so far
        if not tor_results.empty:
            wins   = (tor_results["result"] == "W").sum()
            losses = (tor_results["result"] == "L").sum()
            st.markdown(f"**Record: {wins}–{losses}** &nbsp;|&nbsp; {len(schedule_df) - wins - losses} games remaining")

        st.caption("▶ = next game · scores update after scraping")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Opponent Breakdown
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"{opponent} Scouting Report")

    # Weakness radar
    if opponent in weak["team"].values:
        opp_row = weak[weak["team"] == opponent].iloc[0]
        weakness_cols = [c for c in weak.columns if c.startswith("weakness_")]
        labels = [c.replace("weakness_", "").replace("_", " ").title() for c in weakness_cols]
        values = [float(opp_row[c]) for c in weakness_cols]

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"#### {opponent} Weakness Radar")
            st.caption(f"Higher = weaker. {len(all_teams)} teams.")
            fig = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor=f"rgba(200,16,46,0.2)",
                line=dict(color=UFT_RED),
                name=opponent,
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, len(all_teams)])),
                showlegend=False, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Toronto vs Their Weaknesses")
            exploit_rows = []
            stat_map = {
                "weakness_pass_defense":   ("passing_yards",         "off_pass_yds",          "Toronto pass offense vs their pass D"),
                "weakness_run_defense":    ("rushing_yards",         "off_rush_yds",          "Toronto run game vs their rush D"),
                "weakness_ball_security":  ("turnovers",             "off_pass_int",          "Their turnover liability"),
                "weakness_pass_offense":   ("passing_yards_allowed", "def_pass_yds_allowed",  "Their pass offense vs Toronto's pass D"),
                "weakness_run_offense":    ("rushing_yards",         "off_rush_yds",          "Their run offense vs Toronto's run D"),
            }
            for weakness, (tor_col, opp_col, label) in stat_map.items():
                if weakness not in opp_row.index:
                    continue
                rank = int(opp_row[weakness])
                if rank >= len(all_teams) - 2:  # top 3 weakest
                    tor_agg = agg[agg["team"] == TORONTO]
                    if not tor_agg.empty and tor_col in tor_agg.columns:
                        exploit_rows.append({
                            "Matchup": label,
                            f"{opponent} Rank (weakness)": f"{rank}/{len(all_teams)}",
                            "Action": "🎯 Attack here",
                        })
            if exploit_rows:
                st.dataframe(pd.DataFrame(exploit_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No major weaknesses in the data.")

    # League-wide comparison highlighting opponent
    st.markdown("---")
    st.markdown(f"#### {opponent} vs League")
    metric_opts = {
        "Total Offense (yds/season)": "off_total_yds",
        "Total Defense Allowed":      "def_total_yds_allowed",
        "Passing Yards Offense":      "off_pass_yds",
        "Rushing Yards Offense":      "off_rush_yds",
        "Passing Yards Allowed":      "def_pass_yds_allowed",
        "Rushing Yards Allowed":      "def_rush_yds_allowed",
        "Sacks Made (defense)":       "def_sacks",
        "Turnover Margin":            "turnover_margin",
    }
    sel_metric_label = st.selectbox("Metric", list(metric_opts.keys()), key="opp_metric")
    sel_metric = metric_opts[sel_metric_label]

    if not coaches.empty and sel_metric in coaches.columns:
        latest = coaches[coaches["season"] == coaches["season"].max()].copy()
        latest["color"] = latest["team"].apply(
            lambda t: UFT_RED if t == opponent else (UFT_BLUE if t == TORONTO else "#AAAAAA")
        )
        fig_bar = px.bar(
            latest.sort_values(sel_metric, ascending=False),
            x="team", y=sel_metric,
            color="team",
            color_discrete_map={opponent: UFT_RED, TORONTO: UFT_BLUE},
            labels={"team": "", sel_metric: sel_metric_label},
            title=f"{sel_metric_label} ({coaches['season'].max()})",
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # How to beat them
    if not gamelog.empty:
        st.markdown("---")
        st.markdown(f"#### How Toronto Beats {opponent}")
        htb = how_to_beat(gamelog, opponent)
        st.caption(htb.get("summary", ""))
        for finding in htb.get("findings", []):
            with st.expander(f"**{finding['category']}**", expanded=True):
                st.write(finding["insight"])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Toronto Trends
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Toronto Season Stats")

    vs_opp = st.toggle(f"vs {opponent} only", value=False, key="trend_vs_opp")

    if gamelog.empty:
        st.info("Game log data not loaded.")
    else:
        if vs_opp:
            trend_log = gamelog[
                (gamelog["team"] == TORONTO) & (gamelog["opponent"] == opponent)
            ]
            if trend_log.empty:
                st.info(f"No game log data for Toronto vs {opponent}.")
                st.stop()
        else:
            trend_log = gamelog

        mom = momentum_score(trend_log, TORONTO)
        m1, m2, m3 = st.columns(3)
        m1.metric("Momentum Score", f"{mom['score']} / 100")
        m2.metric("Current Form", mom["label"])
        m3.metric("Recent Offense (last 3)", f"{mom['recent_avg_offense']} yds",
                  delta=f"{mom['recent_avg_offense'] - mom['season_avg_offense']:+.0f} vs season avg")

        if "trend_df" in mom:
            fig_mom = px.line(
                mom["trend_df"], x="label", y="composite",
                title="Toronto Performance (weighted by recency)",
                markers=True, labels={"label": "Game", "composite": "Score"},
                color_discrete_sequence=[UFT_BLUE],
            )
            fig_mom.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Season avg")
            fig_mom.update_xaxes(tickangle=30)
            st.plotly_chart(fig_mom, use_container_width=True)

        trend = season_trend(trend_log, TORONTO)
        if not trend.empty:
            trend["label"] = trend["game_num"].astype(str) + ". " + trend["opponent"] + " (" + trend["result"] + ")"
            fig2 = px.line(
                trend, x="label", y="total_offense", markers=True,
                title="Toronto Offensive Yards Per Game",
                labels={"label": "Game", "total_offense": "Total Yards"},
                color_discrete_sequence=[UFT_BLUE],
            )
            fig2.update_xaxes(tickangle=30)
            st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.bar(
                trend, x="label", y=["turnovers", "sacks_taken", "penalty_yards"],
                barmode="group",
                title="Toronto Turnovers, Sacks & Penalty Yards",
                labels={"label": "Game", "value": "Count / Yards"},
            )
            fig3.update_xaxes(tickangle=30)
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Toronto Win Condition Fingerprint")
        st.caption("What's different in wins vs losses")
        wcf = win_condition_fingerprint(trend_log, TORONTO)
        if wcf.empty:
            st.info("Not enough wins and losses to compare yet.")
        else:
            top = wcf.iloc[0]
            st.info(
                f"**Biggest swing:** {top['Stat']}: "
                f"{top['Avg in Wins']} in wins vs {top['Avg in Losses']} in losses ({top['Swing']}). "
                f"Toronto needs to be *{top['better_when']}* in this stat to win."
            )
            display_wcf = wcf.drop(columns=["better_when", "_raw_diff"], errors="ignore")
            st.dataframe(display_wcf, use_container_width=True, hide_index=True)
            fig_wcf = px.bar(
                wcf.head(5), x="Stat", y=["Avg in Wins", "Avg in Losses"],
                barmode="group",
                color_discrete_map={"Avg in Wins": UFT_BLUE, "Avg in Losses": UFT_RED},
                title="Toronto: Wins vs Losses",
                labels={"value": "Average", "variable": ""},
            )
            st.plotly_chart(fig_wcf, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Game Plan
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"Game Plan: Toronto vs {opponent}")

    if not gamelog.empty:
        mom_opp = momentum_score(gamelog, opponent)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{opponent} Momentum", f"{mom_opp['score']} / 100")
        m2.metric("Their Form", mom_opp["label"])
        m3.metric(f"{opponent} Recent Offense", f"{mom_opp['recent_avg_offense']} yds",
                  delta=f"{mom_opp['recent_avg_offense'] - mom_opp['season_avg_offense']:+.0f} vs their season avg")

    st.markdown("---")
    st.markdown("#### Matchup Exploiter")
    st.caption("🟢 = Toronto advantage, 🔴 = their advantage")
    exploits = matchup_exploiter(agg, TORONTO, opponent)
    if not exploits:
        st.info("Not enough data to compute mismatches.")
    else:
        for i, ex in enumerate(exploits):
            is_toronto = ex["edge"] == TORONTO
            with st.expander(
                f"{'🟢' if is_toronto else '🔴'} **{ex['title']}** ({ex['edge']})",
                expanded=(i < 2),
            ):
                c1, c2 = st.columns(2)
                c1.metric("Toronto", ex["home_val"])
                c2.metric(opponent, ex["away_val"])
                st.markdown(f"**Game plan:** {ex['recommendation']}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Opponent Play Tendencies
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader(f"{opponent} Play Tendencies")
    st.caption(f"Upload {opponent}'s play-by-play. Run Game, Pass Game, Situational, Efficiency.")

    TEMPLATE_PATH  = Path(__file__).parent / "data" / "manual" / "play_template.xlsx"
    FAKE_DATA_PATH = Path(__file__).parent / "data" / "manual" / "queens_plays_3games.xlsx"

    col_up, col_dl = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            f"Upload {opponent} play-by-play (Excel or CSV)",
            type=["xlsx", "xls", "csv"], key="opp_plays"
        )
    with col_dl:
        st.markdown("<br>", unsafe_allow_html=True)
        if TEMPLATE_PATH.exists():
            with open(TEMPLATE_PATH, "rb") as f:
                st.download_button(
                    "⬇ Download Template",
                    f.read(),
                    file_name="play_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    play_df = None
    if uploaded:
        try:
            play_df = load_play_file(uploaded)
            st.success(f"✅ Loaded {len(play_df)} plays from {opponent}.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    elif FAKE_DATA_PATH.exists():
        import openpyxl  # noqa
        _raw = pd.read_excel(FAKE_DATA_PATH)
        play_df = derive_buckets(normalize_columns(_raw))
        st.info(
            f"Showing Queen's data ({len(play_df)} plays). "
            f"Upload {opponent}'s file above to switch."
        )

    if play_df is not None and not play_df.empty:

        # ── Sidebar filters ────────────────────────────────────────────────────
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"**Play Filters**")
            f_down = st.selectbox("Down", ["All", 1, 2, 3, 4], key="pb_down")
            f_dist = st.selectbox("Distance", ["All", "Short (1-2)", "Medium (3-6)", "Long (7+)"], key="pb_dist")
            f_zone = st.selectbox("Field Zone", ["All"] + (
                [str(z) for z in play_df["field_zone"].dropna().unique()] if "field_zone" in play_df.columns else []
            ), key="pb_zone")
            f_form = st.selectbox("Formation", ["All"] + (
                sorted(play_df["formation"].dropna().unique().tolist()) if "formation" in play_df.columns else []
            ), key="pb_form")
            f_pers = st.selectbox("Personnel", ["All"] + (
                sorted(play_df["personnel"].dropna().unique().tolist()) if "personnel" in play_df.columns else []
            ), key="pb_pers")
            f_qtr  = st.selectbox("Quarter", ["All", 1, 2, 3, 4], key="pb_qtr")
            f_sit  = st.selectbox("Game Situation", ["All"] + (
                sorted(play_df["game_situation"].dropna().unique().tolist()) if "game_situation" in play_df.columns else []
            ), key="pb_sit")

        filtered = apply_filters(play_df, {k: v for k, v in {
            "down":           None if f_down == "All" else f_down,
            "dist_bucket":    None if f_dist == "All" else f_dist,
            "field_zone":     None if f_zone == "All" else f_zone,
            "formation":      None if f_form == "All" else f_form,
            "personnel":      None if f_pers == "All" else f_pers,
            "quarter":        None if f_qtr  == "All" else f_qtr,
            "game_situation": None if f_sit  == "All" else f_sit,
        }.items() if v})

        # ── Overview header ────────────────────────────────────────────────────
        ov1, ov2, ov3 = st.columns(3)
        split = run_pass_split(filtered)
        run_pct  = split.get("Run", 0)
        pass_pct = split.get("Pass", 0)
        ov1.metric("Total Plays (filtered)", len(filtered))
        ov2.metric("Run %", f"{run_pct:.1f}%")
        ov3.metric("Pass %", f"{pass_pct:.1f}%")

        # ── Overview: Run/Pass + Down/Distance heatmap ─────────────────────────
        ov_c1, ov_c2 = st.columns(2)
        with ov_c1:
            if not split.empty:
                fig_split = px.pie(
                    values=split.values, names=split.index, hole=0.45,
                    color=split.index,
                    color_discrete_map={"Run": UFT_RED, "Pass": UFT_BLUE, "Play Action Pass": "#7B3F9E"},
                    title="Overall Run / Pass Split",
                )
                fig_split.update_traces(texttemplate="%{label}<br>%{value:.1f}%")
                fig_split.update_layout(showlegend=False, margin=dict(t=40, b=10))
                st.plotly_chart(fig_split, use_container_width=True)
        with ov_c2:
            dd = tendency_by_down_distance(filtered)
            if not dd.empty:
                run_dd = dd[dd["play_category"] == "Run"].pivot_table(
                    index="down", columns="dist_bucket", values="pct", aggfunc="first"
                ).fillna(0)
                fig_dd = px.imshow(
                    run_dd, text_auto=".0f",
                    color_continuous_scale=["white", UFT_RED],
                    labels={"color": "Run %"},
                    title="Run % by Down x Distance",
                )
                fig_dd.update_layout(margin=dict(t=40, b=10))
                st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 1: RUN GAME
        # ══════════════════════════════════════════════════════════════════════
        with st.expander("🏃 Run Game", expanded=True):
            rg1, rg2 = st.columns(2)

            with rg1:
                st.markdown("##### Run Direction")
                dt = direction_tendency(filtered)
                if not dt.empty:
                    fig_dir = px.bar(
                        dt, x="direction", y="pct",
                        color_discrete_sequence=[UFT_RED],
                        labels={"pct": "% of runs", "direction": ""},
                        title="Run Direction",
                    )
                    fig_dir.update_layout(margin=dict(t=40, b=10))
                    st.plotly_chart(fig_dir, use_container_width=True)
                else:
                    st.info("Add a Direction column to see run direction breakdown.")

            with rg2:
                st.markdown("##### Run Success Rate by Direction")
                rsr = run_success_rate(filtered)
                if not rsr.empty:
                    fig_rsr = px.bar(
                        rsr, x="direction", y="success_rate", text="plays",
                        color_discrete_sequence=[UFT_BLUE],
                        labels={"success_rate": "Success %", "direction": ""},
                        title="Run Success % by Direction",
                    )
                    fig_rsr.update_traces(texttemplate="%{text} plays", textposition="outside")
                    fig_rsr.update_layout(margin=dict(t=40, b=10))
                    st.plotly_chart(fig_rsr, use_container_width=True)
                else:
                    st.info("Need Gain + Down + Distance columns for success rate.")

            rg3, rg4 = st.columns(2)

            with rg3:
                st.markdown("##### Run Direction by Personnel")
                st.caption("Run direction by personnel group")
                dbp = direction_by_personnel(filtered)
                if not dbp.empty:
                    fig_dbp = px.bar(
                        dbp, x="personnel", y="pct", color="direction",
                        barmode="stack",
                        labels={"pct": "% of runs", "personnel": "", "direction": "Direction"},
                        title="Run direction by personnel group",
                    )
                    fig_dbp.update_layout(
                        margin=dict(t=40, b=10), legend=dict(orientation="h"),
                        xaxis_tickangle=15,
                    )
                    st.plotly_chart(fig_dbp, use_container_width=True)
                else:
                    st.info("Need Direction + Personnel columns.")

            with rg4:
                st.markdown("##### Hash Mark Tendencies")
                st.caption("Does hash affect run/pass split?")
                ht = hash_tendency(filtered)
                if not ht.empty:
                    fig_ht = px.bar(
                        ht, x="hash", y="pct", color="play_category",
                        barmode="group",
                        color_discrete_map={"Run": UFT_RED, "Pass": UFT_BLUE},
                        labels={"pct": "% of plays", "hash": "Hash", "play_category": ""},
                        title="Run vs Pass by hash mark",
                    )
                    fig_ht.update_layout(margin=dict(t=40, b=10), legend=dict(orientation="h"))
                    st.plotly_chart(fig_ht, use_container_width=True)
                else:
                    st.info("Add a Hash column (Left / Middle / Right).")

            rg5, rg6 = st.columns(2)

            with rg5:
                st.markdown("##### Formation Tendencies")
                ft = formation_tendency(filtered)
                if not ft.empty:
                    fig_ft = px.bar(
                        ft, x="formation", y="pct", color="play_category",
                        barmode="stack",
                        color_discrete_map={"Run": UFT_RED, "Pass": UFT_BLUE},
                        labels={"pct": "% of plays", "formation": "", "play_category": ""},
                        title="Run/Pass % by formation",
                    )
                    fig_ft.update_layout(margin=dict(t=40, b=10), legend=dict(orientation="h"))
                    st.plotly_chart(fig_ft, use_container_width=True)

            with rg6:
                st.markdown("##### Personnel Tendencies")
                pte = personnel_tendency(filtered)
                if not pte.empty:
                    fig_pte = px.bar(
                        pte, x="personnel", y="pct", color="play_category",
                        barmode="stack",
                        color_discrete_map={"Run": UFT_RED, "Pass": UFT_BLUE},
                        labels={"pct": "% of plays", "personnel": "", "play_category": ""},
                        title="Run/Pass % by personnel package",
                    )
                    fig_pte.update_layout(
                        margin=dict(t=40, b=10), legend=dict(orientation="h"),
                        xaxis_tickangle=15,
                    )
                    st.plotly_chart(fig_pte, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 2: PASS GAME
        # ══════════════════════════════════════════════════════════════════════
        with st.expander("🎯 Pass Game", expanded=True):
            pg1, pg2 = st.columns(2)

            with pg1:
                st.markdown("##### Pass Depth by Down & Distance")
                st.caption("Pass depth by down")
                pdbs = pass_depth_by_situation(filtered)
                if not pdbs.empty and "down" in pdbs.columns:
                    # summarise to down × pass_depth
                    summ = filtered[filtered["play_category"] == "Pass"].copy()
                    if "pass_depth" in summ.columns and "down" in summ.columns:
                        grp = summ.groupby(["down", "pass_depth"]).size().reset_index(name="count")
                        total = grp.groupby("down")["count"].transform("sum")
                        grp["pct"] = (grp["count"] / total * 100).round(1)
                        fig_pd = px.bar(
                            grp, x="down", y="pct", color="pass_depth",
                            barmode="stack",
                            color_discrete_sequence=[UFT_BLUE, "#005BAC", "#88B0D8"],
                            labels={"pct": "% of passes", "down": "Down", "pass_depth": "Depth"},
                            title="Pass Depth by Down",
                        )
                        fig_pd.update_layout(margin=dict(t=40, b=10), legend=dict(orientation="h"))
                        st.plotly_chart(fig_pd, use_container_width=True)
                elif not pdbs.empty:
                    fig_pd2 = px.bar(
                        pdbs, x="pass_depth", y="pct",
                        color_discrete_sequence=[UFT_BLUE],
                        labels={"pct": "% of passes", "pass_depth": "Depth"},
                        title="Overall pass depth distribution",
                    )
                    st.plotly_chart(fig_pd2, use_container_width=True)
                else:
                    st.info("Add a Pass_Depth column (Short / Intermediate / Deep).")

            with pg2:
                st.markdown("##### Completion Rate by Pass Depth")
                st.caption("Where are they completing passes?")
                crd = completion_rate_by_depth(filtered)
                if not crd.empty:
                    fig_crd = px.bar(
                        crd, x="pass_depth", y="completion_pct", text="plays",
                        color_discrete_sequence=[UFT_BLUE],
                        labels={"completion_pct": "Completion %", "pass_depth": ""},
                        title="Completion % per depth zone",
                    )
                    fig_crd.update_traces(texttemplate="%{text} att", textposition="outside")
                    fig_crd.update_layout(margin=dict(t=40, b=10))
                    st.plotly_chart(fig_crd, use_container_width=True)
                else:
                    st.info("Need Pass_Depth + Result columns.")

            pg3, pg4 = st.columns(2)

            with pg3:
                st.markdown("##### Play Action Usage")
                st.caption("Play action usage by down")
                pat = play_action_tendency(filtered)
                if not pat.empty:
                    fig_pa = px.bar(
                        pat, x="down", y="play_action_pct", text="plays",
                        color_discrete_sequence=["#7B3F9E"],
                        labels={"play_action_pct": "Play Action %", "down": "Down"},
                        title="Play Action Rate by Down",
                    )
                    fig_pa.update_traces(texttemplate="%{text} pass plays", textposition="outside")
                    fig_pa.update_layout(margin=dict(t=40, b=10))
                    st.plotly_chart(fig_pa, use_container_width=True)
                else:
                    st.info("Tag Play_Type as 'Play Action Pass' to track play action.")

            with pg4:
                st.markdown("##### Pass Success Rate by Depth")
                st.caption("Which depth converts?")
                psd = pass_success_by_depth(filtered)
                if not psd.empty:
                    fig_psd = px.bar(
                        psd, x="pass_depth", y="success_rate", text="plays",
                        color_discrete_sequence=[UFT_BLUE],
                        labels={"success_rate": "Success %", "pass_depth": ""},
                        title="Pass success rate by depth zone",
                    )
                    fig_psd.update_traces(texttemplate="%{text} plays", textposition="outside")
                    fig_psd.update_layout(margin=dict(t=40, b=10))
                    st.plotly_chart(fig_psd, use_container_width=True)
                else:
                    st.info("Need Pass_Depth + Gain + Down + Distance columns.")

            st.markdown("##### Pre-Snap Motion Impact on Pass")
            mt = motion_tendency(filtered)
            if not mt.empty:
                fig_mt = px.bar(
                    mt, x="motion", y="pct", color="play_category",
                    barmode="group",
                    color_discrete_map={"Run": UFT_RED, "Pass": UFT_BLUE},
                    labels={"pct": "% of plays", "motion": "", "play_category": ""},
                    title="Motion vs Run/Pass Split",
                )
                fig_mt.update_layout(margin=dict(t=40, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig_mt, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 3: SITUATIONAL
        # ══════════════════════════════════════════════════════════════════════
        with st.expander("🎲 Situational", expanded=True):
            sit1, sit2 = st.columns(2)

            with sit1:
                st.markdown("##### 3rd Down Breakdown")
                st.caption("Conversion rate and run/pass split by distance")
                tdb = third_down_breakdown(filtered)
                if not tdb.empty:
                    fig_3d = px.bar(
                        tdb, x="dist_bucket", y=["run_pct", "pass_pct"],
                        barmode="group",
                        color_discrete_map={"run_pct": UFT_RED, "pass_pct": UFT_BLUE},
                        labels={"value": "% of 3rd downs", "dist_bucket": "Distance", "variable": ""},
                        title="3rd down play type by distance",
                    )
                    fig_3d.update_layout(margin=dict(t=40, b=10), legend=dict(orientation="h"))
                    st.plotly_chart(fig_3d, use_container_width=True)

                    # conversion rate table
                    tdb_disp = tdb.rename(columns={
                        "dist_bucket": "Distance", "plays": "Plays",
                        "run_pct": "Run %", "pass_pct": "Pass %",
                        "conversion_rate": "Conversion %",
                    })
                    st.dataframe(tdb_disp, use_container_width=True, hide_index=True)
                else:
                    st.info("Need Down + Distance + Result columns for 3rd down breakdown.")

            with sit2:
                st.markdown("##### Red Zone Tendencies")
                st.caption("Inside the 10")
                rzd = redzone_tendencies_detail(play_df)
                if not rzd.empty:
                    fig_rz = px.bar(
                        rzd, x="play_type", y="pct",
                        color="play_type",
                        color_discrete_map={"Run": UFT_RED, "Pass": UFT_BLUE},
                        text="plays",
                        labels={"pct": "% of red zone plays", "play_type": ""},
                        title="Red zone run vs pass",
                    )
                    fig_rz.update_traces(texttemplate="%{text} plays", textposition="outside")
                    fig_rz.update_layout(showlegend=False, margin=dict(t=40, b=10))
                    st.plotly_chart(fig_rz, use_container_width=True)

                    # formation/personnel table
                    rzd_disp = rzd[["play_type", "plays", "pct", "top_formation", "top_personnel", "success_rate"]].rename(columns={
                        "play_type": "Play", "plays": "Plays", "pct": "%",
                        "top_formation": "Top Formation", "top_personnel": "Top Personnel",
                        "success_rate": "Success %",
                    })
                    st.dataframe(rzd_disp, use_container_width=True, hide_index=True)
                else:
                    st.info("Add a Yard_Line column for red zone analysis.")

            st.markdown("##### Play-Calling by Game Situation")
            st.caption("Run/pass split by game situation")
            situ = situation_tendency(play_df)
            if not situ.empty:
                fig_sit = px.bar(
                    situ, x="game_situation", y="pct", color="play_category",
                    barmode="group",
                    color_discrete_map={"Run": UFT_RED, "Pass": UFT_BLUE},
                    labels={"pct": "% of plays", "game_situation": "Situation", "play_category": ""},
                    title="Run vs Pass by Game Situation",
                    category_orders={"game_situation": ["Leading", "Close", "Trailing"]},
                )
                fig_sit.update_layout(margin=dict(t=40, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig_sit, use_container_width=True)
            else:
                st.info("Add a Game_Situation column (Leading / Close / Trailing) to unlock situational analysis.")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 4: EFFICIENCY
        # ══════════════════════════════════════════════════════════════════════
        with st.expander("📊 Efficiency", expanded=True):
            ef1, ef2 = st.columns(2)

            with ef1:
                st.markdown("##### Success Rate by Formation")
                st.caption("Which formation is working?")
                srf = success_rate_by_group(filtered, "formation")
                if not srf.empty:
                    fig_srf = px.bar(
                        srf, x="formation", y="success_rate", text="plays",
                        color_discrete_sequence=[UFT_BLUE],
                        labels={"success_rate": "Success %", "formation": ""},
                        title="Formation Success Rate",
                    )
                    fig_srf.update_traces(texttemplate="%{text} plays", textposition="outside")
                    fig_srf.add_hline(y=50, line_dash="dash", line_color="gray",
                                      annotation_text="50% benchmark")
                    fig_srf.update_layout(margin=dict(t=40, b=10))
                    st.plotly_chart(fig_srf, use_container_width=True)
                else:
                    st.info("Need Gain + Down + Distance + Formation columns.")

            with ef2:
                st.markdown("##### Success Rate by Personnel")
                st.caption("Which personnel group is most effective?")
                srp = success_rate_by_group(filtered, "personnel")
                if not srp.empty:
                    fig_srp = px.bar(
                        srp, x="personnel", y="success_rate", text="plays",
                        color_discrete_sequence=[UFT_BLUE],
                        labels={"success_rate": "Success %", "personnel": ""},
                        title="Personnel group success rate",
                    )
                    fig_srp.update_traces(texttemplate="%{text} plays", textposition="outside")
                    fig_srp.add_hline(y=50, line_dash="dash", line_color="gray",
                                      annotation_text="50% benchmark")
                    fig_srp.update_layout(margin=dict(t=40, b=10), xaxis_tickangle=15)
                    st.plotly_chart(fig_srp, use_container_width=True)
                else:
                    st.info("Need Gain + Down + Distance + Personnel columns.")

            ef3, ef4 = st.columns(2)

            with ef3:
                st.markdown("##### Avg Gain by Formation")
                ag = avg_gain_by_situation(filtered, "formation")
                if not ag.empty:
                    fig_ag = px.bar(
                        ag, x="formation", y="avg_gain", text="plays",
                        color_discrete_sequence=[UFT_RED],
                        labels={"avg_gain": "Avg Yards/Play", "formation": ""},
                        title="Yards Per Play by Formation",
                    )
                    fig_ag.update_traces(texttemplate="%{text} plays", textposition="outside")
                    fig_ag.update_layout(margin=dict(t=40, b=10))
                    st.plotly_chart(fig_ag, use_container_width=True)

            with ef4:
                st.markdown("##### Explosive Plays")
                st.caption("Runs 10+ yds, passes 20+ yds")
                exp = explosive_plays(filtered)
                if not exp.empty:
                    fig_exp = px.bar(
                        exp, x="play_type", y="count",
                        color="play_type",
                        color_discrete_map={"Run": UFT_RED, "Pass": UFT_BLUE},
                        text="avg_gain",
                        labels={"count": "# Explosive Plays", "play_type": ""},
                        title="Explosive Plays",
                    )
                    fig_exp.update_traces(texttemplate="avg %{text} yds", textposition="outside")
                    fig_exp.update_layout(showlegend=False, margin=dict(t=40, b=10))
                    st.plotly_chart(fig_exp, use_container_width=True)

                    # detail table
                    exp_disp = exp.rename(columns={
                        "play_type": "Type", "count": "Plays", "avg_gain": "Avg Gain",
                        "top_formation": "Top Formation", "top_direction": "Top Dir / Depth",
                    })
                    st.dataframe(exp_disp, use_container_width=True, hide_index=True)
                else:
                    st.info("Not enough data or no chunk plays in filtered sample.")

            st.markdown("##### Yards Per Play by Down × Distance × Play Type")
            ypp = yards_per_play_by_situation(filtered)
            if not ypp.empty and len(ypp) > 1:
                fig_ypp = px.bar(
                    ypp, x="avg_gain", y=ypp.apply(
                        lambda r: f"D{int(r['down']) if 'down' in ypp.columns else ''} "
                                  f"{r.get('dist_bucket','')} "
                                  f"({r.get('play_category','')})", axis=1
                    ) if "down" in ypp.columns else ypp.index,
                    color="play_category" if "play_category" in ypp.columns else None,
                    color_discrete_map={"Run": UFT_RED, "Pass": UFT_BLUE},
                    orientation="h",
                    labels={"x": "Avg Yards/Play", "y": "Situation"},
                    title="Yards per play across all down/distance/type combos",
                )
                fig_ypp.update_layout(margin=dict(t=40, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig_ypp, use_container_width=True)

        with st.expander("📋 Full Play Log"):
            st.dataframe(filtered, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 — League Intel
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("League Intel")

    non_toronto = [t for t in all_teams if t != TORONTO]
    li1, li2 = st.columns(2)
    with li1:
        team_a = st.selectbox("Team A", non_toronto, key="li_a")
    with li2:
        team_b = st.selectbox("Team B", [t for t in non_toronto if t != team_a], key="li_b")

    st.markdown(f"### {team_a} vs {team_b}")

    # Head-to-head from game results
    if not games.empty:
        h2h = games[
            ((games["home_team"] == team_a) & (games["away_team"] == team_b)) |
            ((games["home_team"] == team_b) & (games["away_team"] == team_a))
        ].copy()
        if not h2h.empty:
            st.markdown(f"#### Head-to-Head Results ({len(h2h)} games)")
            h2h["Winner"] = h2h.apply(
                lambda r: r["home_team"] if r["home_score"] > r["away_score"] else r["away_team"], axis=1
            )
            st.dataframe(h2h[["season","home_team","home_score","away_score","away_team","Winner"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No direct matchup data in the dataset.")

    # Season stats comparison
    st.markdown("#### Season Stats Comparison")
    compare_cols = {
        "Passing Yards (offense)": "off_pass_yds",
        "Rushing Yards (offense)": "off_rush_yds",
        "Pass Yards Allowed":       "def_pass_yds_allowed",
        "Rush Yards Allowed":       "def_rush_yds_allowed",
        "Sacks Made":               "def_sacks",
        "Forced Fumbles":           "def_forced_fumbles",
        "Turnover Margin":          "turnover_margin",
    }
    if not coaches.empty:
        latest = coaches[coaches["season"] == coaches["season"].max()]
        a_row = latest[latest["team"] == team_a]
        b_row = latest[latest["team"] == team_b]
        if not a_row.empty and not b_row.empty:
            comp_rows = []
            for label, col in compare_cols.items():
                if col in latest.columns:
                    a_val = round(float(a_row[col].values[0]), 1)
                    b_val = round(float(b_row[col].values[0]), 1)
                    comp_rows.append({"Stat": label, team_a: a_val, team_b: b_val})
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Matchup exploiter between the two non-Toronto teams
    st.markdown("#### Matchup Exploiter")
    st.caption(f"Key mismatches: {team_a} vs {team_b}")
    li_exploits = matchup_exploiter(agg, team_a, team_b)
    if li_exploits:
        for ex in li_exploits[:3]:
            with st.expander(f"**{ex['title']}** ({ex['edge']})", expanded=False):
                c1, c2 = st.columns(2)
                c1.metric(team_a, ex["home_val"])
                c2.metric(team_b, ex["away_val"])
                st.markdown(f"**Tendency:** {ex['recommendation']}")

    # Upload play data for non-Toronto team
    st.markdown("---")
    st.markdown(f"#### Upload {team_a} or {team_b} Play Data")
    st.caption("Upload their film data to see tendencies")
    li_uploaded = st.file_uploader("Play-by-play Excel or CSV", type=["xlsx","xls","csv"], key="li_plays")
    if li_uploaded:
        try:
            li_df = load_play_file(li_uploaded)
            st.success(f"Loaded {len(li_df)} plays.")
            split = run_pass_split(li_df)
            if not split.empty:
                lc1, lc2 = st.columns(2)
                with lc1:
                    fig_li = px.pie(values=split.values, names=split.index, hole=0.4,
                                    title="Overall Run/Pass Split")
                    st.plotly_chart(fig_li, use_container_width=True)
                with lc2:
                    ft_li = formation_tendency(li_df)
                    if not ft_li.empty:
                        fig_ft = px.bar(ft_li, x="formation", y="pct", color="play_category",
                                        barmode="stack", title="Formation Tendencies")
                        st.plotly_chart(fig_ft, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read file: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 7 — Raw Data
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("Raw Data")
    st.markdown("#### Coaches-View Season Stats")
    st.dataframe(coaches, use_container_width=True)
    st.markdown("#### Aggregated Averages")
    st.dataframe(agg, use_container_width=True)
    st.markdown("#### Game Results")
    st.dataframe(games, use_container_width=True)
