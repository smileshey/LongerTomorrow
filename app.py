import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(
    page_title="US YPLL Explorer",
    layout="wide",
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 320px;
            max-width: 320px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

UCD_MAP = {
    "Cancer": "cancer",
    "Heart Disease": "heart_disease",
    "Stroke": "stroke",
    "Chronic Lower Respiratory Disease": "lower_resp",
    "Accidents": "accidents",
}

STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

# Did a quick google search regarding projected improvements in deaths for these categories.
# Serves as a starting point so the users aren't served a blank map
starting_trend = {
    "cancer":        -15,   # ≈1.6–1.7%/yr decline
    "heart_disease": -10,   # ≈1.0%/yr decline
    "stroke":        -12,   # ≈1.2%/yr decline
    "lower_resp":    -14,   # ≈1.5%/yr decline
    "accidents":      -5,   # small improvement, but research is less certain about this
}

FEATURE_COLS = [
    "year", "state", "sex", "UCD", "years_from_start",
    "obesity_pct", "uninsured_pct", "income_mean", "employed_pct",
    "diabetes_pct", "educa_z", "smoking_pct_z", "binge_drink_pct_z",
    "seatbelt_always_pct_z", "rural_pct",
]

# not used in the UI right now. This was determined to be a dead end, but keeping it here in case we revisit
ACTION_FEATURES = [
    "obesity_pct",
    "uninsured_pct",
    "diabetes_pct",
    "smoking_pct_z",
    "binge_drink_pct_z",
]

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)


@st.cache_data
def load_data():
    
    df = pd.read_csv("data/future_df.csv")
    df = df.drop_duplicates().reset_index(drop=True)

    for col in ["state", "sex", "UCD"]:
        df[col] = df[col].astype("category")
    
    df["cause_short"] = df["UCD"].map(UCD_MAP)

    return df


def aggregate_by_state(df, improvements, model, target_year=2030):
    """
    df: future_df (all years 2021–2030)
    improvements: dict like {"cancer": -15, "heart_disease": -10, ...}
    model: LightGBM model loaded from model.pkl
    target_year: which year in future_df to map (2030)
    """

    # 1. Take only the target year rows
    d = df[df["year"] == target_year].copy()

    # Make sure cause_short exists
    if "cause_short" not in d.columns:
        d["cause_short"] = d["UCD"].map(UCD_MAP)

    # 2. Predict YPLL for each (state, sex, UCD) row
    X = d[FEATURE_COLS]
    d["ypll_pred"] = model.predict(X)

    # 3. Aggregate by state & cause: baseline YPLL by cause
    cause_totals = (
        d.groupby(["state", "cause_short"], as_index=False)["ypll_pred"]
         .sum()
         .rename(columns={"ypll_pred": "cause_ypll_base"})
    )

    # 4. Aggregate to state totals
    state_totals = (
        cause_totals.groupby("state", as_index=False)["cause_ypll_base"]
                    .sum()
                    .rename(columns={"cause_ypll_base": "baseline_total"})
    )

    # 5. Merge to compute cause shares
    merged = cause_totals.merge(state_totals, on="state", how="left")

    merged["cause_share"] = 0.0
    mask = merged["baseline_total"] > 0
    merged.loc[mask, "cause_share"] = (
        merged.loc[mask, "cause_ypll_base"] / merged.loc[mask, "baseline_total"]
    )

    # 6. Apply slider factors: negative = fewer deaths = fewer YPLL (good)
    factors_by_cause = {k: 1 + (v / 100.0) for k, v in improvements.items()}

    merged["factor"] = (
        merged["cause_short"]
        .astype(str)
        .map(factors_by_cause)
        .astype("float64")
        .fillna(1.0)
    )

    merged["adjusted_cause_ypll"] = merged["cause_ypll_base"] * merged["factor"]

    # 7. Summarize back to state level
    summary = (
        merged.groupby("state", as_index=False)
              .agg(
                  baseline_total=("cause_ypll_base", "sum"),
                  adjusted_total=("adjusted_cause_ypll", "sum"),
              )
    )

    summary["years_gained"] = summary["baseline_total"] - summary["adjusted_total"]
    summary["state_abbrev"] = summary["state"].map(STATE_ABBREV)

    return summary

# Kept for later & not used in the current UI
def aggregate_by_state_actions(df, base_year, feature_changes, model, target_year=2030):
    d_base = df[df["year"] == base_year].copy()
    start_year = df["year"].min()

    d_base["year"] = target_year
    d_base["years_from_start"] = target_year - start_year
    X_base = d_base[FEATURE_COLS]
    d_base["ypll_base"] = model.predict(X_base)

    d_actions = d_base.copy()
    for feat, pct in feature_changes.items():
        if feat in d_actions.columns:
            factor = 1 + pct / 100.0
            d_actions[feat] = d_actions[feat] * factor

    # Clip to 5th–95th percentile for stability
    clip_q = (
        df[df["year"] == base_year][ACTION_FEATURES]
        .quantile([0.05, 0.95])
    )

    for feat in ACTION_FEATURES:
        if feat in d_actions.columns:
            lo = clip_q.loc[0.05, feat]
            hi = clip_q.loc[0.95, feat]
            d_actions[feat] = d_actions[feat].clip(lower=lo, upper=hi)

    X_actions = d_actions[FEATURE_COLS]
    d_actions["ypll_actions"] = model.predict(X_actions)

    baseline_state = (
        d_base.groupby("state", as_index=False)["ypll_base"]
        .sum()
        .rename(columns={"ypll_base": "baseline_total"})
    )
    adjusted_state = (
        d_actions.groupby("state", as_index=False)["ypll_actions"]
        .sum()
        .rename(columns={"ypll_actions": "adjusted_total"})
    )

    summary = baseline_state.merge(adjusted_state, on="state", how="left")
    summary["years_gained"] = summary["baseline_total"] - summary["adjusted_total"]
    summary["state_abbrev"] = summary["state"].map(STATE_ABBREV)

    return summary


df = load_data()
min_year = int(df["year"].min())
max_year = int(df["year"].max())
base_year = max_year  # stick with 2020 as baseline

st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Section",
    (
        "Introduction",
        "Modeling Years of Life Gained",
        "Conclusion",
    ),
)

st.sidebar.markdown("---")

if mode == "Modeling Years of Life Gained":
    st.sidebar.markdown("### 10-year change in deaths by cause")
    st.sidebar.caption("Negative = fewer deaths (good), positive = more deaths (bad).")

    cancer_change = st.sidebar.slider(
        "Cancer deaths change (%)",
        -20,
        20,
        starting_trend["cancer"],
    )
    heart_change = st.sidebar.slider(
        "Heart disease deaths change (%)",
        -20,
        20,
        starting_trend["heart_disease"],
    )
    stroke_change = st.sidebar.slider(
        "Stroke deaths change (%)",
        -20,
        20,
        starting_trend["stroke"],
    )
    lower_resp_change = st.sidebar.slider(
        "Lower respiratory deaths change (%)",
        -20,
        20,
        starting_trend["lower_resp"],
    )
    accidents_change = st.sidebar.slider(
        "Accident deaths change (%)",
        -20,
        20,
        starting_trend["accidents"],
    )

    improvements = {
        "cancer":        cancer_change,
        "heart_disease": heart_change,
        "stroke":        stroke_change,
        "lower_resp":    lower_resp_change,
        "accidents":     accidents_change,
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        **How this map works**

        In this project, we estimate how changes in major causes of death translate into
        **years of potential life gained by 2030**, using **{base_year}** as a baseline year
        for each state's risk factors.

        In this section, you directly adjust **cause-specific death rates**. Negative slider
        values represent **fewer deaths** (improvement), and positive values represent
        **more deaths** (worsening). The model projects how these changes would alter
        years of potential life lost, and the map shows the **net years gained**.
        """
    )

    summary = aggregate_by_state(
        df,
        improvements=improvements,
        model=model,
        target_year=2030,
    )

elif mode == "Introduction":
    improvements = starting_trend.copy()

    st.sidebar.markdown("### Overview")
    st.sidebar.markdown(
        f"""
        In this project, we use the CDC WONDER Underlying Cause of Death (UCD) dataset for 1999–2020
        to estimate Years of Potential Life Lost (YPLL), aggregated at the U.S. state level.
        The UCD data provide yearly state-level counts and rates of deaths for U.S. residents, 
        broken down by underlying cause of death (ICD codes) and demographics such as age, sex, and race.

        We use these data to estimate how changes in major causes of death translate into 
        years of potential life gained by 2030, using {base_year} as a baseline year for each state's
        risk factors. A LightGBM model trained on the historical data generates the projections 
        that drive the map on this page.

        The map shows a trend-based scenario: modest 10-year reductions in deaths 
        from cancer, heart disease, stroke, lower respiratory disease, and accidents, 
        consistent with recent national mortality trends.

        States shaded in green are projected to gain more years of life, 
        while red indicates net losses relative to the baseline.
        """
    )

    summary = aggregate_by_state(
        df,
        improvements=improvements,
        model=model,
        target_year=2030,
    )

elif mode == "Conclusion":
    improvements = starting_trend.copy()

    st.sidebar.markdown("### Conclusion")
    st.sidebar.markdown(
        f"""
        This view summarizes what the model implies about **potential life gained** under
        a modest improvement scenario.

        These numbers are **model-based estimates**, not forecasts. They depend on how
        well the model captures the relationship between state-level risk factors and
        years of potential life lost, and on the assumptions you choose for changes
        in deaths by cause.

        Use this as a tool to compare **relative impact across states** and to
        explore how improvements in different causes of death might translate into
        more years of life for communities.
        """
    )

    summary = aggregate_by_state(
        df,
        improvements=improvements,
        model=model,
        target_year=2030,
    )

max_slider_pct = 20
max_possible_change = float(summary["baseline_total"].max() * (max_slider_pct / 100.0))

st.title("A Longer Tomorrow: Years of Potential Life Gained")
st.caption(
    f"Projected years of potential life gained (or lost) by state in 2030, using {base_year} "
    "as the baseline risk factor year. Green indicates more years gained; red indicates losses."
)

st.subheader(f"Years of potential life gained • 2030 (baseline risk factors: {base_year})")

fig = px.choropleth(
    summary.dropna(subset=["state_abbrev"]),
    locations="state_abbrev",
    locationmode="USA-states",
    color="years_gained",
    scope="usa",
    hover_name="state",
    hover_data={
        "baseline_total": ":,.0f",
        "adjusted_total": ":,.0f",
        "years_gained": ":,.0f",
        "state_abbrev": False,
    },
    labels={"years_gained": "Years of potential life gained (2030)"},
    color_continuous_scale="RdYlGn",
    range_color=(-max_possible_change, max_possible_change),
    color_continuous_midpoint=0,
)

fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    height=650,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
fig.update_geos(
    bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Summary stats")

total_baseline = summary["baseline_total"].sum()
total_adjusted = summary["adjusted_total"].sum()
total_gained = summary["years_gained"].sum()
delta_value = total_adjusted - total_baseline
pct_gained = (total_gained / total_baseline * 100) if total_baseline else 0.0

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Total baseline YPLL (2030)",
        f"{total_baseline:,.0f}",
    )

with c2:
    st.metric(
        "Total adjusted YPLL (2030)",
        f"{total_adjusted:,.0f}",
        delta=f"{delta_value:,.0f}",
    )

with c3:
    label = (
        f"≈ {pct_gained:.1f}% fewer YPLL than baseline"
        if pct_gained >= 0
        else f"≈ {abs(pct_gained):.1f}% more YPLL than baseline"
    )
    st.metric(
        "Total years of potential life gained",
        f"{total_gained:,.0f}",
    )
    st.caption(label)
