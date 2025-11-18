import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# CONFIGURATIONS FOR MAPPING

st.set_page_config(
    page_title="US YPLL Explorer",
    layout="wide",
)

UCD_MAP = {
    '#Malignant neoplasms (C00-C97)': 'cancer',
    '#Diseases of heart (I00-I09,I11,I13,I20-I51)': 'heart_disease',
    '#Cerebrovascular diseases (I60-I69)': 'stroke',
    '#Chronic lower respiratory diseases (J40-J47)': 'lower_resp',
    '#Accidents (unintentional injuries) (V01-X59,Y85-Y86)': 'accidents',
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

FEATURE_COLS = [
    "year", "state", "sex", "UCD", "years_from_start",
    "obesity_pct", "uninsured_pct", "income_mean", "employed_pct",
    "diabetes_pct", "educa_z", "smoking_pct_z", "binge_drink_pct_z",
    "seatbelt_always_pct_z", "rural_pct",
]

# LOADING THE MODEL

with open('model/model.pkl', "rb") as f:
    model = pickle.load(f)

# LOADING IN THE DATA

@st.cache_data
def load_data():
    df = pd.read_csv("df_states.csv")

    start_year = df["year"].min()
    df["years_from_start"] = df["year"] - start_year

    for col in ["state", "sex", "UCD"]:
        df[col] = df[col].astype("category")

    df["cause_short"] = df["UCD"].map(UCD_MAP)

    return df


def aggregate_by_state(df, base_year, improvements, model, target_year=2030):

    d = df[df["year"] == base_year].copy()

    start_year = df["year"].min()
    d["year"] = target_year
    d["years_from_start"] = target_year - start_year

    X = d[FEATURE_COLS]
    d["ypll_pred"] = model.predict(X)

    d["cause_short"] = d["UCD"].map(UCD_MAP)
    factors_by_cause = {k: 1 - v / 100.0 for k, v in improvements.items()}
    d["factor"] = d["cause_short"].map(factors_by_cause).fillna(1.0)
    d["ypll_adj"] = d["ypll_pred"] * d["factor"]

    summary = (
        d.groupby("state", as_index=False)
         .agg(
             baseline_total=("ypll_pred", "sum"),
             adjusted_total=("ypll_adj", "sum"),
         )
    )
    summary["savings"] = summary["baseline_total"] - summary["adjusted_total"]
    summary["state_abbrev"] = summary["state"].map(STATE_ABBREV)

    return summary


# --- SIDEBAR CONTROLS --------------------------------------------------------
df = load_data()
st.sidebar.title("Controls")

base_year = st.sidebar.selectbox(
    "Base year (covariates)",
    options=sorted(df["year"].unique()),
    index=len(sorted(df["year"].unique())) - 1,  # default to latest
)

st.sidebar.markdown("### 10-year improvement assumptions")

cancer_improve = st.sidebar.slider("Cancer deaths reduction (%)", 0, 50, 0)
heart_improve = st.sidebar.slider("Heart disease deaths reduction (%)", 0, 50, 0)
stroke_improve  = st.sidebar.slider("Stroke deaths reduction (%)", 0, 50, 0)
lower_resp_improve  = st.sidebar.slider("Lower respiratory deaths reduction (%)", 0, 50, 0)
accidents_improve = st.sidebar.slider("Accident deaths reduction (%)", 0, 50, 0)

improvements = {
    "cancer":        cancer_improve,
    "heart_disease": heart_improve,
    "stroke":        stroke_improve,
    "lower_resp":    lower_resp_improve,
    "accidents":     accidents_improve,
}

summary = aggregate_by_state(df, base_year, improvements, model, target_year=2030)

# --- MAIN LAYOUT -------------------------------------------------------------

st.title("A Longer Tomorrow: Modelling Years of Potential Life Lost")
st.caption(
    "Model-based projection of YPLL by state in 2030, updated when you adjust "
    "cause-of-death improvements."
)

baseline_min = summary["baseline_total"].min()
baseline_max = summary["baseline_total"].max()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Adjusted YPLL by state • 2030 (base year {base_year})")

    fig = px.choropleth(
        summary.dropna(subset=["state_abbrev"]),
        locations="state_abbrev",
        locationmode="USA-states",
        color="adjusted_total",
        scope="usa",
        hover_name="state",
        hover_data={
            "baseline_total": ":,.0f",
            "adjusted_total": ":,.0f",
            "savings": ":,.0f",
            "state_abbrev": False,
        },
        labels={"adjusted_total": "Adj. YPLL (2030)"},
        color_continuous_scale="Viridis",
        range_color=(baseline_min, baseline_max),
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Adj. YPLL (2030)"),
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Summary stats")

    total_baseline = summary["baseline_total"].sum()
    total_adjusted = summary["adjusted_total"].sum()
    total_savings  = summary["savings"].sum()
    delta_value = total_adjusted - total_baseline
    pct_delta      = total_savings / total_baseline * 100

    st.metric(
        "Total baseline YPLL (2030)",
        f"{total_baseline:,.0f}",
    )
    st.metric(
        "Total adjusted YPLL (2030)",
        f"{total_adjusted:,.0f}",
        delta=f"{delta_value:,.0f}",
    )
    st.metric(
        "Relative reduction",
        f"{pct_delta:.1f} %",
    )