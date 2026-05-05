"""
OUA Football Analyst — Streamlit app
Run with: streamlit run streamlit_app.py
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis.team_stats import (
    load_data, compute_team_aggregates, compute_sos_adjusted_aggregates,
    weakness_scores, season_trend
)
from analysis.play_analysis import (
    load_play_file, load_play_csv, apply_filters,
    run_pass_split, tendency_by_down_distance, formation_tendency,
    personnel_tendency, direction_tendency, avg_gain_by_situation,
    redzone_tendencies, motion_tendency,
)
from analysis.predictor import train, predict_matchup
from analysis.scouting import (
    win_condition_fingerprint, how_to_beat, momentum_score, matchup_exploiter
)

st.set_page_config(page_title="OUA Football Analyst", page_icon="🏈", layout="wide")
st.title("🏈 OUA Football Analyst")
st.caption("Pattern analysis & game prediction for Ontario University Athletics football")

# ── Sidebar: data refresh ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data")
    st.caption("Scraped from oua.ca — free, no login required.")
    if st.button("Refresh Data from OUA", type="primary"):
        with st.spinner("Scraping oua.ca..."):
            try:
                from scraper.oua_scraper import scrape_season
                from scraper.stats_scraper import scrape_team_gamelog
                from scraper.coaches_scraper import scrape_team_coaches
                import time

                TEAMS = ["carleton","guelph","laurier","mcmaster","ottawa",
                         "queens","toronto","waterloo","western","windsor","york"]
                SEASONS_COACHES = ["2025-26","2024-25"]
                SEASONS_GAMELOG = ["2024-25","2023-24"]

                RAW = Path(__file__).parent / "data" / "raw"

                # games
                all_games = []
                for yr in [2024, 2023, 2022]:
                    try:
                        all_games.append(scrape_season(yr))
                        time.sleep(0.5)
                    except Exception:
                        pass
                if all_games:
                    import pandas as pd
                    pd.concat(all_games, ignore_index=True).to_csv(RAW / "all_games.csv", index=False)

                # gamelog
                import pandas as pd
                gl_rows = []
                TEAM_NAME_MAP = {"carleton":"Carleton","guelph":"Guelph","laurier":"Laurier",
                                 "mcmaster":"McMaster","ottawa":"Ottawa","queens":"Queen's",
                                 "toronto":"Toronto","waterloo":"Waterloo","western":"Western",
                                 "windsor":"Windsor","york":"York"}
                for season in SEASONS_GAMELOG:
                    for t in TEAMS:
                        try:
                            gl_rows.append(scrape_team_gamelog(t, season))
                            time.sleep(0.4)
                        except Exception:
                            pass
                if gl_rows:
                    pd.concat(gl_rows, ignore_index=True).to_csv(RAW / "team_stats_all.csv", index=False)

                # coaches
                c_rows = []
                for season in SEASONS_COACHES:
                    for t in TEAMS:
                        try:
                            row = scrape_team_coaches(t, season)
                            if row:
                                c_rows.append(row)
                            time.sleep(0.4)
                        except Exception:
                            pass
                if c_rows:
                    pd.DataFrame(c_rows).to_csv(RAW / "coaches_stats_all.csv", index=False)

                st.cache_data.clear()
                st.success("Done! Reload the page to see updated data.")
            except Exception as e:
                st.error(f"Scrape failed: {e}")

    st.markdown("---")
    sos_on = st.toggle(
        "Strength-of-Schedule Adjusted",
        value=False,
        help=(
            "Normalizes each team's offensive stats by the defensive quality "
            "of their opponents. 300 yards against Queen's (top defense) is "
            "worth more than 300 yards against York (weak defense)."
        ),
    )
    if sos_on:
        st.caption("SOS adjusted — stats scaled by opponent defensive quality.")
    else:
        st.caption("Raw stats — no opponent adjustment.")

    st.markdown("---")
    st.caption("Built for OUA Football analysis. Data: oua.ca")


# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    games, gamelog, coaches = load_data()
    agg_raw = compute_team_aggregates(coaches)
    agg_sos = compute_sos_adjusted_aggregates(gamelog, coaches)
    return games, gamelog, coaches, agg_raw, agg_sos

games, gamelog, coaches, agg_raw, agg_sos = get_data()
agg = agg_sos if sos_on else agg_raw
weak = weakness_scores(agg)
teams = sorted(agg["team"].unique().tolist())

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Game Predictor", "Team Weaknesses", "Season Trends", "Scouting Report", "Play Breakdown", "Raw Stats"]
)

# ── Tab 1: Game Predictor ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Win Probability Predictor")
    st.caption(f"Using {'SOS-adjusted' if sos_on else 'raw'} stats — toggle in sidebar to switch.")
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home Team", teams, key="home")
    with col2:
        away_opts = [t for t in teams if t != home]
        away = st.selectbox("Away Team", away_opts, key="away")

    if st.button("Train Model & Predict", type="primary"):
        with st.spinner("Training on historical data..."):
            _, _, _, acc = train(games, agg)
            st.success(f"Model trained — cross-val accuracy: {acc:.1%}")

        result = predict_matchup(home, away, agg)

        if "error" in result:
            st.error(result["error"])
        else:
            st.markdown("### Prediction")
            c1, c2 = st.columns(2)
            c1.metric(f"{home} win probability", f"{result['home_win_prob']:.1%}")
            c2.metric(f"{away} win probability", f"{result['away_win_prob']:.1%}")

            st.markdown("### Key Factor Breakdown")
            breakdown = result["breakdown"]
            rows = []
            for feat, vals in breakdown.items():
                edge = home if vals["home"] > vals["away"] else away
                # for defensive stats, lower is better — flip the edge
                if feat in ("yards_allowed_per_game", "passing_yards_allowed",
                            "rushing_yards_allowed", "turnovers"):
                    edge = away if vals["home"] > vals["away"] else home
                rows.append({
                    "Factor": feat.replace("_", " ").title(),
                    home: round(vals["home"], 1),
                    away: round(vals["away"], 1),
                    "Edge": edge,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Tab 2: Team Weaknesses ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Team Weakness Dashboard")
    if sos_on:
        st.info("**SOS Adjusted** — offensive rankings account for opponent defensive quality. A team that dominated weak defenses will rank lower here than raw stats suggest.")
    else:
        st.caption("Higher rank = weaker in the league for that dimension")

    weakness_cols = [c for c in weak.columns if c.startswith("weakness_")]
    display = weak[["team"] + weakness_cols + ["overall_weakness_score"]].sort_values(
        "overall_weakness_score", ascending=False
    ).copy()
    display.columns = [c.replace("weakness_", "").replace("_", " ").title() for c in display.columns]
    st.dataframe(display, use_container_width=True)

    st.markdown("### Weakness Radar — Select a Team")
    selected = st.selectbox("Team", teams, key="radar_team")
    row = weak[weak["team"] == selected].iloc[0]
    labels = [c.replace("weakness_", "").replace("_", " ").title() for c in weakness_cols]
    values = [float(row[c]) for c in weakness_cols]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=selected,
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, len(teams)])), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### League-Wide Comparison")
    metric = st.selectbox("Metric", [
        "off_total_yds", "def_total_yds_allowed", "turnover_margin",
        "def_sacks", "off_pass_yds", "off_rush_yds",
        "passing_yards_allowed", "rushing_yards_allowed",
    ])
    if metric in coaches.columns:
        latest = coaches[coaches["season"] == coaches["season"].max()]
        fig3 = px.bar(
            latest.sort_values(metric, ascending=False),
            x="team", y=metric,
            title=f"{metric.replace('_', ' ').title()} — {coaches['season'].max()} Season",
            color="team",
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── Tab 3: Season Trends ───────────────────────────────────────────────────────
with tab3:
    st.subheader("Season Performance Trends")
    trend_team = st.selectbox("Select Team", teams, key="trend_team")

    if gamelog.empty:
        st.info("No game-by-game data loaded.")
    else:
        trend = season_trend(gamelog, trend_team)
        if trend.empty:
            st.info(f"No game log data for {trend_team}.")
        else:
            trend["label"] = trend["game_num"].astype(str) + ". " + trend["opponent"] + " (" + trend["result"] + ")"
            fig = px.line(
                trend, x="label", y="total_offense",
                labels={"label": "Game", "total_offense": "Total Yards"},
                title=f"{trend_team} — Offensive Yards Per Game",
                markers=True,
            )
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(
                trend, x="label", y=["turnovers", "sacks_taken", "penalty_yards"],
                barmode="group",
                labels={"label": "Game", "value": "Count / Yards"},
                title=f"{trend_team} — Turnovers, Sacks & Penalty Yards Per Game",
            )
            fig2.update_xaxes(tickangle=30)
            st.plotly_chart(fig2, use_container_width=True)

# ── Tab 4: Scouting Report ────────────────────────────────────────────────────
with tab4:
    st.subheader("Scouting Report")
    st.caption("Data-driven insights beyond the box score")

    sc1, sc2 = st.columns(2)
    with sc1:
        scout_team = st.selectbox("Team to Scout", teams, key="scout_team")
    with sc2:
        opponent_team = st.selectbox("Their Upcoming Opponent", [t for t in teams if t != scout_team], key="scout_opp")

    if gamelog.empty:
        st.warning("Game log data not loaded — run the stats scraper first.")
    else:
        # ── Momentum Score ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### Momentum — {scout_team}")
        mom = momentum_score(gamelog, scout_team)
        m1, m2, m3 = st.columns(3)
        m1.metric("Momentum Score", f"{mom['score']} / 100")
        m2.metric("Form", mom["label"])
        m3.metric(
            "Recent Offense (last 3 games)",
            f"{mom['recent_avg_offense']} yds",
            delta=f"{mom['recent_avg_offense'] - mom['season_avg_offense']:+.0f} vs season avg",
        )
        if "trend_df" in mom:
            fig_mom = px.line(
                mom["trend_df"], x="label", y="composite",
                title="Performance Composite (weighted — higher = better)",
                markers=True,
                labels={"label": "Game", "composite": "Composite Score"},
            )
            fig_mom.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Season avg")
            fig_mom.update_xaxes(tickangle=30)
            st.plotly_chart(fig_mom, use_container_width=True)

        # ── Win Condition Fingerprint ─────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### Win Condition Fingerprint — {scout_team}")
        st.caption("What statistically separates their wins from their losses")
        wcf = win_condition_fingerprint(gamelog, scout_team)
        if wcf.empty:
            st.info("Not enough wins and losses to compare.")
        else:
            # highlight the top insight
            top = wcf.iloc[0]
            st.info(
                f"**Biggest swing:** {top['Stat']} — "
                f"{top['Avg in Wins']} in wins vs {top['Avg in Losses']} in losses ({top['Swing']}). "
                f"They need to be *{top['better_when']}* in this stat to win."
            )
            display_wcf = wcf.drop(columns=["better_when", "_raw_diff"], errors="ignore")
            st.dataframe(display_wcf, use_container_width=True, hide_index=True)

            fig_wcf = px.bar(
                wcf.head(5), x="Stat",
                y=["Avg in Wins", "Avg in Losses"],
                barmode="group",
                title=f"{scout_team} — Key Stats: Wins vs Losses",
                labels={"value": "Average", "variable": ""},
            )
            st.plotly_chart(fig_wcf, use_container_width=True)

        # ── How to Beat Them ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### How to Beat {scout_team}")
        htb = how_to_beat(gamelog, scout_team)
        st.caption(htb.get("summary", ""))
        for finding in htb.get("findings", []):
            with st.expander(f"**{finding['category']}**", expanded=True):
                st.write(finding["insight"])

        # ── Matchup Exploiter ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### Matchup Exploiter — {scout_team} vs {opponent_team}")
        st.caption("Biggest statistical mismatches in this specific game")
        exploits = matchup_exploiter(agg, scout_team, opponent_team)
        if not exploits:
            st.info("Not enough data to compute matchup mismatches.")
        else:
            for i, ex in enumerate(exploits):
                edge_color = "green" if ex["edge"] == scout_team else "red"
                with st.expander(
                    f"{'🟢' if ex['edge'] == scout_team else '🔴'} **{ex['title']}** — Edge: {ex['edge']}",
                    expanded=(i == 0),
                ):
                    c1, c2 = st.columns(2)
                    c1.metric(scout_team, ex["home_val"])
                    c2.metric(opponent_team, ex["away_val"])
                    st.markdown(f"**Game plan:** {ex['recommendation']}")


# ── Tab 5: Play Breakdown ─────────────────────────────────────────────────────
with tab5:
    st.subheader("Play-Level Breakdown")
    st.caption("Upload a tagged play-by-play Excel to see situational tendencies by down, distance, formation, and personnel.")

    TEMPLATE_PATH = Path(__file__).parent / "data" / "manual" / "play_template.xlsx"

    col_up, col_dl = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader("Upload play-by-play Excel or CSV", type=["xlsx", "xls", "csv"])
    with col_dl:
        st.markdown("<br>", unsafe_allow_html=True)
        if TEMPLATE_PATH.exists():
            with open(TEMPLATE_PATH, "rb") as f:
                st.download_button(
                    "Download Template",
                    f.read(),
                    file_name="play_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    # load data — uploaded file or fall back to template
    play_df = None
    if uploaded:
        try:
            play_df = load_play_file(uploaded)
            st.success(f"Loaded {len(play_df)} plays.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    elif TEMPLATE_PATH.exists():
        play_df = load_play_csv(TEMPLATE_PATH) if str(TEMPLATE_PATH).endswith(".csv") else (
            lambda: (lambda df: df)(pd.read_excel(TEMPLATE_PATH).pipe(
                lambda df: df.rename(columns={c: c.strip().lower() for c in df.columns})
            ))
        )()
        # simpler load for template
        import openpyxl  # noqa
        _raw = pd.read_excel(TEMPLATE_PATH)
        from analysis.play_analysis import normalize_columns, derive_buckets
        play_df = derive_buckets(normalize_columns(_raw))
        st.info(f"Showing sample data ({len(play_df)} plays). Upload your own file above.")

    if play_df is not None and not play_df.empty:

        # ── Sidebar filters ───────────────────────────────────────────────────
        with st.sidebar:
            st.markdown("---")
            st.markdown("**Play Breakdown Filters**")
            f_down = st.selectbox("Down", ["All", 1, 2, 3, 4], key="pb_down")
            f_dist = st.selectbox("Distance", ["All", "Short (1-2)", "Medium (3-6)", "Long (7+)"], key="pb_dist")
            f_zone = st.selectbox("Field Zone", ["All"] + (
                play_df["field_zone"].dropna().unique().tolist() if "field_zone" in play_df.columns else []
            ), key="pb_zone")
            f_form = st.selectbox("Formation", ["All"] + (
                sorted(play_df["formation"].dropna().unique().tolist()) if "formation" in play_df.columns else []
            ), key="pb_form")
            f_pers = st.selectbox("Personnel", ["All"] + (
                sorted(play_df["personnel"].dropna().unique().tolist()) if "personnel" in play_df.columns else []
            ), key="pb_pers")
            f_qtr = st.selectbox("Quarter", ["All", 1, 2, 3, 4], key="pb_qtr")

        filters = {
            "down": None if f_down == "All" else f_down,
            "dist_bucket": None if f_dist == "All" else f_dist,
            "field_zone": None if f_zone == "All" else f_zone,
            "formation": None if f_form == "All" else f_form,
            "personnel": None if f_pers == "All" else f_pers,
            "quarter": None if f_qtr == "All" else f_qtr,
        }
        filtered = apply_filters(play_df, {k: v for k, v in filters.items() if v})

        st.markdown(f"**{len(filtered)} plays** match current filters")

        # ── Row 1: Run/Pass split + Down & Distance heatmap ──────────────────
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            st.markdown("#### Run / Pass Split")
            split = run_pass_split(filtered)
            if not split.empty:
                fig = px.pie(
                    values=split.values,
                    names=split.index,
                    color=split.index,
                    color_discrete_map={"Run": "#003E7E", "Pass": "#C8102E"},
                    hole=0.4,
                )
                fig.update_traces(texttemplate="%{label}<br>%{value:.1f}%")
                fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            st.markdown("#### Run % by Down & Distance")
            dd = tendency_by_down_distance(filtered)
            if not dd.empty:
                run_dd = dd[dd["play_category"] == "Run"].pivot_table(
                    index="down", columns="dist_bucket", values="pct", aggfunc="first"
                ).fillna(0)
                fig2 = px.imshow(
                    run_dd,
                    text_auto=".0f",
                    color_continuous_scale=["white", "#003E7E"],
                    labels={"color": "Run %"},
                    title="Run % (darker = more likely to run)",
                )
                fig2.update_layout(margin=dict(t=40, b=10))
                st.plotly_chart(fig2, use_container_width=True)

        # ── Row 2: Formation + Personnel ──────────────────────────────────────
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            st.markdown("#### Formation Tendencies")
            ft = formation_tendency(filtered)
            if not ft.empty:
                fig3 = px.bar(
                    ft, x="formation", y="pct", color="play_category",
                    barmode="stack",
                    color_discrete_map={"Run": "#003E7E", "Pass": "#C8102E"},
                    labels={"pct": "% of plays", "formation": "", "play_category": ""},
                )
                fig3.update_layout(margin=dict(t=10, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig3, use_container_width=True)

        with r2c2:
            st.markdown("#### Personnel Tendencies")
            pt = personnel_tendency(filtered)
            if not pt.empty:
                fig4 = px.bar(
                    pt, x="personnel", y="pct", color="play_category",
                    barmode="stack",
                    color_discrete_map={"Run": "#003E7E", "Pass": "#C8102E"},
                    labels={"pct": "% of plays", "personnel": "", "play_category": ""},
                )
                fig4.update_layout(margin=dict(t=10, b=10), legend=dict(orientation="h"),
                                   xaxis_tickangle=20)
                st.plotly_chart(fig4, use_container_width=True)

        # ── Row 3: Run direction + Avg gain ──────────────────────────────────
        r3c1, r3c2 = st.columns(2)

        with r3c1:
            st.markdown("#### Run Direction")
            dt = direction_tendency(filtered)
            if not dt.empty:
                fig5 = px.bar(
                    dt, x="direction", y="pct",
                    color_discrete_sequence=["#003E7E"],
                    labels={"pct": "% of runs", "direction": ""},
                )
                fig5.update_layout(margin=dict(t=10, b=10))
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("No run direction data in current filter.")

        with r3c2:
            st.markdown("#### Avg Gain by Formation")
            ag = avg_gain_by_situation(filtered, "formation")
            if not ag.empty:
                fig6 = px.bar(
                    ag, x="formation", y="avg_gain", text="plays",
                    color_discrete_sequence=["#C8102E"],
                    labels={"avg_gain": "Avg Yards", "formation": "", "plays": "# plays"},
                )
                fig6.update_traces(texttemplate="%{text} plays", textposition="outside")
                fig6.update_layout(margin=dict(t=10, b=10))
                st.plotly_chart(fig6, use_container_width=True)

        # ── Row 4: Red zone + Motion ──────────────────────────────────────────
        r4c1, r4c2 = st.columns(2)

        with r4c1:
            st.markdown("#### Red Zone Tendencies")
            rz = redzone_tendencies(play_df)  # always full dataset for red zone
            if not rz.empty:
                fig7 = px.pie(
                    rz, values="pct", names="play_type",
                    color_discrete_map={"Run": "#003E7E", "Pass": "#C8102E"},
                    hole=0.4,
                )
                fig7.update_traces(texttemplate="%{label}<br>%{value:.1f}%")
                fig7.update_layout(showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig7, use_container_width=True)
            else:
                st.info("No red zone data (need Yard_Line column).")

        with r4c2:
            st.markdown("#### Pre-Snap Motion Impact")
            mt = motion_tendency(filtered)
            if not mt.empty:
                fig8 = px.bar(
                    mt, x="motion", y="pct", color="play_category",
                    barmode="group",
                    color_discrete_map={"Run": "#003E7E", "Pass": "#C8102E"},
                    labels={"pct": "% of plays", "motion": "", "play_category": ""},
                )
                fig8.update_layout(margin=dict(t=10, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig8, use_container_width=True)
            else:
                st.info("No motion data in current filter.")

        # ── Raw play log ──────────────────────────────────────────────────────
        with st.expander("View raw play log"):
            st.dataframe(filtered, use_container_width=True)


# ── Tab 6: Raw Stats ───────────────────────────────────────────────────────────
with tab6:
    st.subheader("Coaches-View Season Stats (all teams)")
    st.dataframe(coaches, use_container_width=True)
    st.subheader("Aggregated Averages")
    st.dataframe(agg, use_container_width=True)
    st.subheader("Game Results")
    st.dataframe(games, use_container_width=True)
