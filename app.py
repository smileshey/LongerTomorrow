import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIG ------------------------------------------------------------------

st.set_page_config(
    page_title="US YPLL Explorer",
    layout="wide",
)

# Map UCD labels in the dataframe to short names we'll use in the UI
UCD_MAP = {
    '#Malignant neoplasms (C00-C97)': 'cancer',
    '#Diseases of heart (I00-I09,I11,I13,I20-I51)': 'heart_disease',
    '#Cerebrovascular diseases (I60-I69)': 'stroke',
    '#Chronic lower respiratory diseases (J40-J47)': 'lower_resp',
    '#Accidents (unintentional injuries) (V01-X59,Y85-Y86)': 'accidents',
}

# Full state name -> USPS abbreviation for Plotly
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


# --- DATA LOADING & PREP -----------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("df_states.csv")

    # Clean deaths: convert "Suppressed" -> NaN -> 0, keep numeric for aggregation
    df["deaths_num"] = pd.to_numeric(df["deaths"], errors="coerce").fillna(0)

    # Keep only the causes we care about for this app
    df = df[df["UCD"].isin(UCD_MAP.keys())].copy()
    df["cause"] = df["UCD"].map(UCD_MAP)

    return df


def aggregate_by_state(df, year, improvements):
    """
    df: full df_states with 'deaths_num' and 'cause' columns
    year: selected year (int)
    improvements: dict like {"cancer": 5, "heart_disease": 10, ...} (percent reductions)
    """

    d = df[df["year"] == year].copy()

    grouped = (
        d.groupby(["state", "cause"], as_index=False)["deaths_num"]
        .sum()
    )

    pivot = grouped.pivot(index="state", columns="cause", values="deaths_num").fillna(0)

    for cause in improvements.keys():
        if cause not in pivot.columns:
            pivot[cause] = 0.0

    pivot["baseline_total"] = pivot[list(improvements.keys())].sum(axis=1)

    for cause, pct in improvements.items():
        reduction_factor = 1 - (pct / 100.0)
        pivot[f"{cause}_adj"] = pivot[cause] * reduction_factor

    adj_cols = [f"{cause}_adj" for cause in improvements.keys()]
    pivot["adjusted_total"] = pivot[adj_cols].sum(axis=1)

    summary = pivot.reset_index()
    summary["state_abbrev"] = summary["state"].map(STATE_ABBREV)

    return summary


# --- SIDEBAR CONTROLS --------------------------------------------------------

df = load_data()

st.sidebar.title("Controls")

year = st.sidebar.selectbox(
    "Year",
    options=sorted(df["year"].unique()),
    index=0
)

st.sidebar.markdown("### 10-year improvement assumptions")

# Pick 4 sliders for now (you can add the 5th easily if you want)
cancer_improve = st.sidebar.slider("Cancer deaths reduction (%)", 0, 50, 0, step=1)
heart_improve = st.sidebar.slider("Heart disease deaths reduction (%)", 0, 50, 0, step=1)
stroke_improve = st.sidebar.slider("Stroke deaths reduction (%)", 0, 50, 0, step=1)
accidents_improve = st.sidebar.slider("Accident deaths reduction (%)", 0, 50, 0, step=1)

improvements = {
    "cancer": cancer_improve,
    "heart_disease": heart_improve,
    "stroke": stroke_improve,
    "accidents": accidents_improve,
    # You can add "lower_resp": slider_value here if you want the 5th slider
}

# --- MAIN LAYOUT -------------------------------------------------------------

st.title("A Longer Tomorrow: Modelling Years of Potential Life Lost")
st.caption(
    "Choropleth map of a YPLL-like metric by state, updated when you adjust "
    "cause-of-death improvements."
)

summary = aggregate_by_state(df, year, improvements)

baseline_min = summary["baseline_total"].min()
baseline_max = summary["baseline_total"].max()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Adjusted YPLL proxy by state • {year}")

    # Plotly choropleth
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
            "state_abbrev": False,
        },
        labels={"adjusted_total": "Adj. YPLL proxy"},
        color_continuous_scale="Viridis",
        range_color=(baseline_min, baseline_max),  # <- key line
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Adj. YPLL\n(proxy)"),
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Summary stats")

    total_baseline = summary["baseline_total"].sum()
    total_adjusted = summary["adjusted_total"].sum()
    delta = total_baseline - total_adjusted
    pct_delta = (delta / total_baseline * 100) if total_baseline > 0 else 0

    st.metric(
        "Total baseline deaths (proxy)",
        f"{total_baseline:,.0f}",
    )
    st.metric(
        "Total adjusted deaths (proxy)",
        f"{total_adjusted:,.0f}",
        delta=f"-{delta:,.0f}",
    )
    st.metric(
        "Relative reduction",
        f"{pct_delta:.1f} %",
    )