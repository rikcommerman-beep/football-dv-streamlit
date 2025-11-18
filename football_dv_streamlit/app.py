import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

st.set_page_config(
    page_title="Football & Domestic Violence – River/Boca",
    page_icon="⚽",
    layout="wide"
)

DV_URL = "https://datos.jus.gob.ar/dataset/linea-137-victimas-de-violencia-familiar/archivo/21b615fc-001d-43d1-9396-e61f804a32cc"
LIGA_PATH = "data/liga_argentina_futbol.csv"  # replace this file with the Kaggle CSV

@st.cache_data(show_spinner=True)
def load_dv_data():
    df = pd.read_csv(DV_URL, low_memory=False)
    date_cols = [c for c in df.columns if "fecha" in c.lower()]
    if not date_cols:
        st.error("No date column found in DV dataset. Please inspect the CSV.")
        st.stop()
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["date"] = df[date_col].dt.date
    dv_daily = (
        df.groupby("date")
          .size()
          .reset_index(name="dv_calls")
    )
    return dv_daily

@st.cache_data(show_spinner=True)
def load_match_data():
    try:
        m = pd.read_csv(LIGA_PATH)
    except FileNotFoundError:
        st.error(
            "Football CSV not found.\n\n"
            "Download the 'Liga Argentina Fútbol' CSV from Kaggle and save it as:\n"
            "data/liga_argentina_futbol.csv"
        )
        st.stop()

    date_candidate_cols = [c for c in m.columns if "date" in c.lower() or "fecha" in c.lower()]
    if not date_candidate_cols:
        st.error("Could not find a date column in Liga Argentina dataset.")
        st.stop()
    date_col = date_candidate_cols[0]
    m[date_col] = pd.to_datetime(m[date_col], errors="coerce")
    m = m.dropna(subset=[date_col])
    m["date"] = m[date_col].dt.date

    home_cols = [c for c in m.columns if "home" in c.lower() and "team" in c.lower()]
    away_cols = [c for c in m.columns if "away" in c.lower() and "team" in c.lower()]
    if not home_cols or not away_cols:
        st.error("Could not find home_team / away_team columns in Liga Argentina dataset.")
        st.stop()
    home_col = home_cols[0]
    away_col = away_cols[0]

    home_goal_cols = [c for c in m.columns if "home" in c.lower() and "goal" in c.lower()]
    away_goal_cols = [c for c in m.columns if "away" in c.lower() and "goal" in c.lower()]
    if not home_goal_cols or not away_goal_cols:
        st.error("Could not find goals columns in Liga Argentina dataset.")
        st.stop()
    hg_col = home_goal_cols[0]
    ag_col = away_goal_cols[0]

    m[home_col] = m[home_col].astype(str).str.strip()
    m[away_col] = m[away_col].astype(str).str.strip()

    m["river_played"] = ((m[home_col] == "River Plate") | (m[away_col] == "River Plate")).astype(int)
    m["boca_played"] = ((m[home_col] == "Boca Juniors") | (m[away_col] == "Boca Juniors")).astype(int)

    m["superclasico"] = (
        ((m[home_col] == "River Plate") & (m[away_col] == "Boca Juniors")) |
        ((m[home_col] == "Boca Juniors") & (m[away_col] == "River Plate"))
    ).astype(int)

    def result_for(team, row):
        if not (row[home_col] == team or row[away_col] == team):
            return None
        home_goals = row[hg_col]
        away_goals = row[ag_col]
        if row[home_col] == team:
            if home_goals > away_goals:
                return "win"
            elif home_goals < away_goals:
                return "loss"
            else:
                return "draw"
        else:
            if away_goals > home_goals:
                return "win"
            elif away_goals < home_goals:
                return "loss"
            else:
                return "draw"

    m["river_result"] = m.apply(lambda r: result_for("River Plate", r), axis=1)
    m["boca_result"] = m.apply(lambda r: result_for("Boca Juniors", r), axis=1)

    return m

@st.cache_data(show_spinner=True)
def build_daily_merged():
    dv = load_dv_data()
    matches = load_match_data()
    start = min(dv["date"].min(), matches["date"].min())
    end = max(dv["date"].max(), matches["date"].max())
    calendar = pd.DataFrame({"date": pd.date_range(start=start, end=end).date})

    agg = (
        matches.groupby("date")
               .agg(
                   river_played=("river_played", "max"),
                   boca_played=("boca_played", "max"),
                   superclasico=("superclasico", "max"),
                   river_win=("river_result", lambda x: 1 if "win" in list(x) else 0),
                   river_loss=("river_result", lambda x: 1 if "loss" in list(x) else 0),
                   boca_win=("boca_result", lambda x: 1 if "win" in list(x) else 0),
                   boca_loss=("boca_result", lambda x: 1 if "loss" in list(x) else 0),
               )
               .reset_index()
    )

    df = (
        calendar
        .merge(dv, on="date", how="left")
        .merge(agg, on="date", how="left")
    )

    df["dv_calls"] = df["dv_calls"].fillna(0).astype(int)
    flag_cols = ["river_played", "boca_played", "superclasico",
                 "river_win", "river_loss", "boca_win", "boca_loss"]
    for c in flag_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    df["any_match"] = ((df["river_played"] == 1) | (df["boca_played"] == 1)).astype(int)
    return df

def page_overview(df):
    st.title("⚽ Domestic Violence & Football in Argentina")
    st.markdown(
        "This dashboard links **Línea 137** domestic-violence hotline calls "
        "with **River Plate** and **Boca Juniors** match days."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total days in dataset", f"{len(df):,}")
    with col2:
        st.metric("Average DV calls / day", f"{df['dv_calls'].mean():.1f}")
    with col3:
        st.metric("Total match days (River or Boca)", f"{df['any_match'].sum():,}")

    st.subheader("Daily domestic-violence calls (Línea 137)")

    min_date = df["date"].min()
    max_date = df["date"].max()
    start_date, end_date = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df_range = df.loc[mask].copy()

    def label_row(row):
        if row["superclasico"] == 1:
            return "Superclásico"
        elif row["any_match"] == 1:
            return "River/Boca match day"
        else:
            return "No match"

    df_range["day_type"] = df_range.apply(label_row, axis=1)

    fig = px.line(
        df_range,
        x="date",
        y="dv_calls",
        color="day_type",
        title="Daily hotline calls, with football days highlighted",
        labels={"dv_calls": "DV calls (Línea 137)", "date": "Date", "day_type": "Day type"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Average DV calls: match days vs non-match days")
    summary = (
        df.assign(
            group=lambda x: np.where(
                x["superclasico"] == 1, "Superclásico",
                np.where(x["any_match"] == 1, "River/Boca match day", "No match")
            )
        )
        .groupby("group")["dv_calls"]
        .agg(["mean", "count"])
        .reset_index()
    )
    summary["mean"] = summary["mean"].round(2)

    fig2 = px.bar(
        summary,
        x="group",
        y="mean",
        text="mean",
        title="Average DV calls by type of day",
        labels={"group": "Day type", "mean": "Average DV calls"},
    )
    fig2.update_traces(textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(summary.rename(columns={"mean": "avg_dv_calls", "count": "number_of_days"}))

def page_teams(df):
    st.title("River Plate & Boca Juniors – Match Effects")

    st.markdown(
        "We compare **hotline calls** on days when River or Boca play, "
        "looking at **wins vs losses** and **Superclásicos**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("River Plate")
        river_days = df[df["river_played"] == 1].copy()
        river_days["result"] = np.select(
            [
                river_days["river_win"] == 1,
                river_days["river_loss"] == 1,
            ],
            ["win", "loss"],
            default="draw"
        )
        river_summary = (
            river_days.groupby("result")["dv_calls"]
            .agg(["mean", "count"])
            .reset_index()
        )
        river_summary["mean"] = river_summary["mean"].round(2)

        fig_r = px.bar(
            river_summary,
            x="result",
            y="mean",
            text="mean",
            title="River match days – DV calls by result",
            labels={"mean": "Average DV calls"},
        )
        fig_r.update_traces(textposition="outside")
        st.plotly_chart(fig_r, use_container_width=True)
        st.dataframe(river_summary.rename(columns={"mean": "avg_dv_calls", "count": "days"}))

    with col2:
        st.subheader("Boca Juniors")
        boca_days = df[df["boca_played"] == 1].copy()
        boca_days["result"] = np.select(
            [
                boca_days["boca_win"] == 1,
                boca_days["boca_loss"] == 1,
            ],
            ["win", "loss"],
            default="draw"
        )
        boca_summary = (
            boca_days.groupby("result")["dv_calls"]
            .agg(["mean", "count"])
            .reset_index()
        )
        boca_summary["mean"] = boca_summary["mean"].round(2)

        fig_b = px.bar(
            boca_summary,
            x="result",
            y="mean",
            text="mean",
            title="Boca match days – DV calls by result",
            labels{"mean": "Average DV calls"},
        )
        fig_b.update_traces(textposition="outside")
        st.plotly_chart(fig_b, use_container_width=True)
        st.dataframe(boca_summary.rename(columns={"mean": "avg_dv_calls", "count": "days"}))

    st.markdown("### Superclásicos (River vs Boca)")

    df_sorted = df.sort_values("date")
    sc_dates = df_sorted.loc[df_sorted["superclasico"] == 1, "date"].tolist()

    if not sc_dates:
        st.info("No Superclásicos found in the overlapping period of both datasets.")
        return

    frames = []
    for d in sc_dates:
        for offset in range(-3, 4):
            day = d + pd.Timedelta(days=offset)
            row = df_sorted[df_sorted["date"] == day]
            if not row.empty:
                tmp = row[["date", "dv_calls"]].copy()
                tmp["match_date"] = d
                tmp["t"] = offset
                frames.append(tmp)

    sc_window = pd.concat(frames, ignore_index=True)
    sc_avg = sc_window.groupby("t")["dv_calls"].mean().reset_index()

    fig_sc = px.line(
        sc_avg,
        x="t",
        y="dv_calls",
        markers=True,
        title="Average DV calls around Superclásicos (t = 0 is match day)",
        labels={"t": "Days relative to Superclásico", "dv_calls": "Average DV calls"},
    )
    st.plotly_chart(fig_sc, use_container_width=True)

def page_methodology(df):
    st.title("Methodology & Notes")

    st.markdown(
        """
        ### Data sources

        **Domestic violence (outcome):**

        - Línea 137 - víctimas de violencia familiar - llamados e intervenciones domiciliarias  
          - Publisher: Ministerio de Justicia, Programa Las Víctimas contra las Violencias  
          - Resource: “Llamados atendidos sobre violencia familiar - línea 137 - 2017 a 2024 - unificado” (CSV)

        **Football (exposure):**

        - Liga Argentina Fútbol – Kaggle dataset containing historical match results for Argentina's top division,
          including River Plate and Boca Juniors. Saved locally as `data/liga_argentina_futbol.csv`.

        ### Variables

        - dv_calls: number of Línea 137 calls per day.
        - river_played / boca_played: 1 if River/Boca appears as home or away team that day.
        - superclasico: 1 if River plays Boca.
        - river_result / boca_result: win / draw / loss based on goals.
        - any_match: 1 if River or Boca plays that day.
        """
    )

    st.markdown(
        """
        ### Limitations

        - Hotline calls are a proxy for incidents and many cases are never reported.
        - Football matches often coincide with weekends or holidays.
        - This is observational data: correlations do not automatically imply causation.
        """
    )

def main():
    df = build_daily_merged()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "River & Boca", "Methodology"])

    if page == "Overview":
        page_overview(df)
    elif page == "River & Boca":
        page_teams(df)
    else:
        page_methodology(df)

if __name__ == "__main__":
    main()
