"""
Varsity Blues Football Analytics
Built for the University of Toronto football program.
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
    load_play_file, apply_filters,
    run_pass_split, tendency_by_down_distance, formation_tendency,
    personnel_tendency, direction_tendency, avg_gain_by_situation,
    redzone_tendencies, motion_tendency, normalize_columns, derive_buckets,
)
from analysis.predictor import train, predict_matchup
from analysis.scouting import (
    win_condition_fingerprint, how_to_beat, momentum_score, matchup_exploiter
)

TORONTO = "Toronto"
UFT_BLUE  = "#003E7E"
UFT_RED   = "#C8102E"

st.set_page_config(
    page_title="Varsity Blues Football Analytics",
    page_icon="🏈",
    layout="wide",
)
st.title("🏈 Varsity Blues Football Analytics")
st.caption("University of Toronto — internal analysis tool")

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
                for yr in [2024, 2023, 2022]:
                    try:
                        all_games.append(scrape_season(yr)); time.sleep(0.5)
                    except Exception: pass
                if all_games:
                    pd.concat(all_games, ignore_index=True).to_csv(RAW / "all_games.csv", index=False)

                gl_rows = []
                for season in ["2024-25", "2023-24"]:
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
             "York's numbers drop significantly — they padded stats vs weak defenses.",
    )
    st.caption("SOS adjusted" if sos_on else "Raw stats")
    st.markdown("---")
    st.caption("Data: oua.ca — no paid services")


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

# ── Global opponent selector (used by all UofT tabs) ──────────────────────────
st.markdown("### Select Upcoming Opponent")
opponent = st.selectbox("", opponents, label_visibility="collapsed", key="global_opp")
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 This Week's Game",
    "🎯 Opponent Breakdown",
    "📈 Toronto Trends",
    "📋 Game Plan",
    "🏈 Opponent Play Tendencies",
    "🔭 League Intel",
    "🗂 Raw Data",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — This Week's Game
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Toronto vs {opponent}")
    st.caption(f"Using {'SOS-adjusted' if sos_on else 'raw'} stats")

    loc = st.radio("Game location", ["Toronto at Home", "Toronto Away"], horizontal=True)
    home_team = TORONTO if loc == "Toronto at Home" else opponent
    away_team = opponent if loc == "Toronto at Home" else TORONTO

    if st.button("Run Prediction", type="primary"):
        with st.spinner("Training model..."):
            try:
                _, _, _, acc = train(games, agg)
                st.success(f"Model accuracy: {acc:.1%}")
            except Exception as e:
                st.warning(f"Model note: {e}")

        result = predict_matchup(home_team, away_team, agg)
        if "error" in result:
            st.error(result["error"])
        else:
            c1, c2 = st.columns(2)
            tor_prob = result["home_win_prob"] if home_team == TORONTO else result["away_win_prob"]
            opp_prob = 1 - tor_prob
            c1.metric("Toronto win probability", f"{tor_prob:.1%}",
                      delta="Favoured" if tor_prob > 0.5 else "Underdog")
            c2.metric(f"{opponent} win probability", f"{opp_prob:.1%}")

            st.markdown("### Statistical Edge Breakdown")
            st.caption("Where Toronto has the advantage and where they don't")
            rows = []
            for feat, vals in result["breakdown"].items():
                h_val = vals["home"]
                a_val = vals["away"]
                tor_val = h_val if home_team == TORONTO else a_val
                opp_val = a_val if home_team == TORONTO else h_val
                defensive = feat in ("yards_allowed_per_game", "passing_yards_allowed",
                                     "rushing_yards_allowed", "turnovers")
                tor_edge = (tor_val < opp_val) if defensive else (tor_val > opp_val)
                rows.append({
                    "Factor": feat.replace("_", " ").title(),
                    "Toronto": round(tor_val, 1),
                    opponent: round(opp_val, 1),
                    "Edge": "✅ Toronto" if tor_edge else f"⚠️ {opponent}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Opponent Breakdown
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Scouting {opponent} — What Toronto Can Exploit")

    # Weakness radar
    if opponent in weak["team"].values:
        opp_row = weak[weak["team"] == opponent].iloc[0]
        weakness_cols = [c for c in weak.columns if c.startswith("weakness_")]
        labels = [c.replace("weakness_", "").replace("_", " ").title() for c in weakness_cols]
        values = [float(opp_row[c]) for c in weakness_cols]

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"#### {opponent} Weakness Radar")
            st.caption(f"Higher = weaker. Rank out of {len(all_teams)} teams.")
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
            st.markdown("#### Toronto Strengths vs Their Weaknesses")
            st.caption("Where Toronto's offense meets their defensive gaps")
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
                st.info("No glaring weaknesses detected — this is a well-rounded opponent.")

    # League-wide comparison highlighting opponent
    st.markdown("---")
    st.markdown(f"#### {opponent} vs League — Key Metrics")
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
            title=f"{sel_metric_label} — {coaches['season'].max()}",
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # How to beat them
    if not gamelog.empty:
        st.markdown("---")
        st.markdown(f"#### How Toronto Beats {opponent} — Pattern Analysis")
        htb = how_to_beat(gamelog, opponent)
        st.caption(htb.get("summary", ""))
        for finding in htb.get("findings", []):
            with st.expander(f"**{finding['category']}**", expanded=True):
                st.write(finding["insight"])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Toronto Trends
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Toronto — Season Performance")

    if gamelog.empty:
        st.info("Game log data not loaded.")
    else:
        mom = momentum_score(gamelog, TORONTO)
        m1, m2, m3 = st.columns(3)
        m1.metric("Momentum Score", f"{mom['score']} / 100")
        m2.metric("Current Form", mom["label"])
        m3.metric("Recent Offense (last 3)", f"{mom['recent_avg_offense']} yds",
                  delta=f"{mom['recent_avg_offense'] - mom['season_avg_offense']:+.0f} vs season avg")

        if "trend_df" in mom:
            fig_mom = px.line(
                mom["trend_df"], x="label", y="composite",
                title="Toronto Performance Composite — weighted by recency",
                markers=True, labels={"label": "Game", "composite": "Score"},
                color_discrete_sequence=[UFT_BLUE],
            )
            fig_mom.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Season avg")
            fig_mom.update_xaxes(tickangle=30)
            st.plotly_chart(fig_mom, use_container_width=True)

        trend = season_trend(gamelog, TORONTO)
        if not trend.empty:
            trend["label"] = trend["game_num"].astype(str) + ". " + trend["opponent"] + " (" + trend["result"] + ")"
            fig2 = px.line(
                trend, x="label", y="total_offense", markers=True,
                title="Toronto — Offensive Yards Per Game",
                labels={"label": "Game", "total_offense": "Total Yards"},
                color_discrete_sequence=[UFT_BLUE],
            )
            fig2.update_xaxes(tickangle=30)
            st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.bar(
                trend, x="label", y=["turnovers", "sacks_taken", "penalty_yards"],
                barmode="group",
                title="Toronto — Turnovers, Sacks & Penalty Yards",
                labels={"label": "Game", "value": "Count / Yards"},
            )
            fig3.update_xaxes(tickangle=30)
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Toronto Win Condition Fingerprint")
        st.caption("What statistically separates Toronto wins from losses")
        wcf = win_condition_fingerprint(gamelog, TORONTO)
        if wcf.empty:
            st.info("Not enough wins and losses to compare yet.")
        else:
            top = wcf.iloc[0]
            st.info(
                f"**Biggest swing:** {top['Stat']} — "
                f"{top['Avg in Wins']} in wins vs {top['Avg in Losses']} in losses ({top['Swing']}). "
                f"Toronto needs to be *{top['better_when']}* in this stat to win."
            )
            display_wcf = wcf.drop(columns=["better_when", "_raw_diff"], errors="ignore")
            st.dataframe(display_wcf, use_container_width=True, hide_index=True)
            fig_wcf = px.bar(
                wcf.head(5), x="Stat", y=["Avg in Wins", "Avg in Losses"],
                barmode="group",
                color_discrete_map={"Avg in Wins": UFT_BLUE, "Avg in Losses": UFT_RED},
                title="Toronto — Key Stats: Wins vs Losses",
                labels={"value": "Average", "variable": ""},
            )
            st.plotly_chart(fig_wcf, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Game Plan
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"Game Plan — Toronto vs {opponent}")
    st.caption("Specific statistical mismatches to exploit this week")

    if not gamelog.empty:
        mom_opp = momentum_score(gamelog, opponent)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{opponent} Momentum", f"{mom_opp['score']} / 100")
        m2.metric("Their Form", mom_opp["label"])
        m3.metric(f"{opponent} Recent Offense", f"{mom_opp['recent_avg_offense']} yds",
                  delta=f"{mom_opp['recent_avg_offense'] - mom_opp['season_avg_offense']:+.0f} vs their season avg")

    st.markdown("---")
    st.markdown("#### Matchup Exploiter")
    st.caption("Biggest statistical mismatches — 🟢 = Toronto advantage, 🔴 = their advantage")
    exploits = matchup_exploiter(agg, TORONTO, opponent)
    if not exploits:
        st.info("Not enough data to compute mismatches.")
    else:
        for i, ex in enumerate(exploits):
            is_toronto = ex["edge"] == TORONTO
            with st.expander(
                f"{'🟢' if is_toronto else '🔴'} **{ex['title']}** — Edge: {ex['edge']}",
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
    st.subheader(f"{opponent} — Play Tendencies")
    st.caption(f"Upload {opponent}'s tagged play-by-play so Toronto's defence knows what's coming")

    TEMPLATE_PATH = Path(__file__).parent / "data" / "manual" / "play_template.xlsx"

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
                    "Download Template",
                    f.read(),
                    file_name="play_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    play_df = None
    if uploaded:
        try:
            play_df = load_play_file(uploaded)
            st.success(f"Loaded {len(play_df)} plays from {opponent}.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    elif TEMPLATE_PATH.exists():
        import openpyxl  # noqa
        _raw = pd.read_excel(TEMPLATE_PATH)
        play_df = derive_buckets(normalize_columns(_raw))
        st.info(f"Showing sample data — upload {opponent}'s actual file above.")

    if play_df is not None and not play_df.empty:
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"**{opponent} Play Filters**")
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
            f_qtr  = st.selectbox("Quarter", ["All", 1, 2, 3, 4], key="pb_qtr")

        filtered = apply_filters(play_df, {k: v for k, v in {
            "down": None if f_down == "All" else f_down,
            "dist_bucket": None if f_dist == "All" else f_dist,
            "field_zone": None if f_zone == "All" else f_zone,
            "formation": None if f_form == "All" else f_form,
            "personnel": None if f_pers == "All" else f_pers,
            "quarter": None if f_qtr == "All" else f_qtr,
        }.items() if v})

        st.markdown(f"**{len(filtered)} plays** match current filters")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown(f"#### {opponent} Run / Pass Split")
            split = run_pass_split(filtered)
            if not split.empty:
                fig = px.pie(values=split.values, names=split.index, hole=0.4,
                             color=split.index,
                             color_discrete_map={"Run": UFT_RED, "Pass": "#555555"})
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
                fig2 = px.imshow(run_dd, text_auto=".0f",
                                 color_continuous_scale=["white", UFT_RED],
                                 labels={"color": "Run %"},
                                 title="Darker = more likely to run")
                st.plotly_chart(fig2, use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("#### Formation Tendencies")
            ft = formation_tendency(filtered)
            if not ft.empty:
                fig3 = px.bar(ft, x="formation", y="pct", color="play_category",
                              barmode="stack",
                              color_discrete_map={"Run": UFT_RED, "Pass": "#555555"},
                              labels={"pct": "% of plays", "formation": "", "play_category": ""})
                fig3.update_layout(margin=dict(t=10, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig3, use_container_width=True)

        with r2c2:
            st.markdown("#### Personnel Tendencies")
            pt = personnel_tendency(filtered)
            if not pt.empty:
                fig4 = px.bar(pt, x="personnel", y="pct", color="play_category",
                              barmode="stack",
                              color_discrete_map={"Run": UFT_RED, "Pass": "#555555"},
                              labels={"pct": "% of plays", "personnel": "", "play_category": ""})
                fig4.update_layout(margin=dict(t=10, b=10), legend=dict(orientation="h"),
                                   xaxis_tickangle=20)
                st.plotly_chart(fig4, use_container_width=True)

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            st.markdown("#### Run Direction")
            dt = direction_tendency(filtered)
            if not dt.empty:
                fig5 = px.bar(dt, x="direction", y="pct",
                              color_discrete_sequence=[UFT_RED],
                              labels={"pct": "% of runs", "direction": ""})
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("No direction data — add a Direction column to the upload.")

        with r3c2:
            st.markdown("#### Avg Gain by Formation")
            ag = avg_gain_by_situation(filtered, "formation")
            if not ag.empty:
                fig6 = px.bar(ag, x="formation", y="avg_gain", text="plays",
                              color_discrete_sequence=[UFT_BLUE],
                              labels={"avg_gain": "Avg Yards", "formation": ""})
                fig6.update_traces(texttemplate="%{text} plays", textposition="outside")
                st.plotly_chart(fig6, use_container_width=True)

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            st.markdown("#### Red Zone Tendencies")
            rz = redzone_tendencies(play_df)
            if not rz.empty:
                fig7 = px.pie(rz, values="pct", names="play_type", hole=0.4,
                              color_discrete_map={"Run": UFT_RED, "Pass": "#555555"})
                fig7.update_traces(texttemplate="%{label}<br>%{value:.1f}%")
                fig7.update_layout(showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig7, use_container_width=True)
            else:
                st.info("Add a Yard_Line column for red zone analysis.")

        with r4c2:
            st.markdown("#### Pre-Snap Motion Impact")
            mt = motion_tendency(filtered)
            if not mt.empty:
                fig8 = px.bar(mt, x="motion", y="pct", color="play_category",
                              barmode="group",
                              color_discrete_map={"Run": UFT_RED, "Pass": "#555555"},
                              labels={"pct": "% of plays", "motion": "", "play_category": ""})
                fig8.update_layout(margin=dict(t=10, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig8, use_container_width=True)
            else:
                st.info("Add a Motion column for motion analysis.")

        with st.expander("View full play log"):
            st.dataframe(filtered, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 — League Intel
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("League Intel")
    st.caption("Watch how other OUA teams play against each other — scout future opponents before you face them")

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
    st.caption(f"Biggest mismatches if {team_a} hosts {team_b} — useful intel if Toronto faces either of them")
    li_exploits = matchup_exploiter(agg, team_a, team_b)
    if li_exploits:
        for ex in li_exploits[:3]:
            with st.expander(f"**{ex['title']}** — Edge: {ex['edge']}", expanded=False):
                c1, c2 = st.columns(2)
                c1.metric(team_a, ex["home_val"])
                c2.metric(team_b, ex["away_val"])
                st.markdown(f"**Tendency:** {ex['recommendation']}")

    # Upload play data for non-Toronto team
    st.markdown("---")
    st.markdown(f"#### Upload {team_a} or {team_b} Play Data")
    st.caption("If you have their tagged film, upload it here to see their tendencies")
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
