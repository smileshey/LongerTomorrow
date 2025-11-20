# LongerTomorrow

# A Longer Tomorrow: Visualizing Future Years of Potential Life Lost

This project explores how changes in major causes of death could translate into years of potential life lost by 2030 across U.S. states.  
It combines CDC mortality data, behavioral risk factor data, a LightGBM model, and an interactive Streamlit UI to let users simulate the projected impacts of changes to those risk factors over the next 10 years.

---

## Overview

We start from historical mortality patterns and risk factors at the **state** level (1999–2020), estimate **Years of Potential Life Lost (YPLL)**, and then train a model to project YPLL for 2030.  

In the Streamlit app:

- A LightGBM model predicts a baseline YPLL in 2030 for each state.
- The model’s predictions are decomposed across five leading causes of death.
- User-adjustable sliders change the assumed 10-year reduction or increase in deaths for each cause.
- The app recomputes a **scaled YPLL projection** and visualizes the result on a U.S. choropleth, as **years of potential life gained or lost** relative to baseline.

---

## Data Sources

### 1. CDC WONDER – Multiple Cause of Death (MCD)

Original mortality data were obtained from the **CDC WONDER Multiple Cause of Death** system for the period **1999–2020**:

- CDC WONDER MCD: <https://wonder.cdc.gov/mcd.html>  

Query settings:

- **Dataset:** Multiple Cause of Death, 1999–2020
- **Underlying Cause of Death:** UCD–ICD–10 113 Cause List
- **Age grouping:** Five-year age groups
- **Geography:** States
- **Strata:** Sex, State, Race
- **Age restriction:** Adults aged **18 years and older**
- **Underlying causes of death restricted to the top 5 UCDs:**
  - `#Diseases of heart (I00-I09,I11,I13,I20-I51)`
  - `#Malignant neoplasms (C00-C97)`
  - `#Cerebrovascular diseases (I60-I69)`
  - `#Chronic lower respiratory diseases (J40-J47)`
  - `#Accidents (unintentional injuries) (V01-X59,Y85-Y86)`

These extracts were cleaned and aggregated into a state-level panel called **`df_states`**.

### 2. CDC BRFSS – Behavioral Risk Factor Surveillance System

To enrich the mortality data with health and behavioral risk factors, we used the **CDC BRFSS** (Behavioral Risk Factor Surveillance System):

- BRFSS data: <https://www.cdc.gov/brfss/annual_data/annual_data.htm>

Relevant features (by state and year) include, for example:

- `obesity_pct`
- `uninsured_pct`
- `income_mean`
- `employed_pct`
- `diabetes_pct`
- `educa_z`
- `smoking_pct_z`
- `binge_drink_pct_z`
- `seatbelt_always_pct_z`

These data were manually downloaded from BRFSS, parsed in Python, and merged into the base mortality dataset in `UCD.py`.

---

## Data Processing (`UCD.py`)

The script **`UCD.py`** handles:

1. Importing the raw CDC WONDER MCD extracts.
2. Aggregating to state–year–cause level and computing YPLL-related quantities.
3. Joining in BRFSS-derived features by state and year.
4. Producing the final modeling dataset `df_states`, which includes:
   - Mortality counts and population,
   - YPLL-related outcome(s),
   - Cause of death (`UCD`),
   - Demographics / geography,
   - Behavioral and socioeconomic covariates,
   - Convenience columns such as `years_from_start`.

The resulting dataframe is saved and later loaded by the modeling notebook.

---

## Modeling (`Phillip's_Code_(Model).ipynb`)

The notebook **`Phillip's_Code_(Model).ipynb`** trains a LightGBM model on `df_states` to predict a projected YPLL-like quantity for 2030 (e.g. `ypll_30`):

- Defines the feature set (including `year`, `state`, `UCD`, and risk factors).
- Encodes categorical variables and ensures consistency with LightGBM’s expectations.
- Fits a LightGBM regressor using historical data.
- Evaluates performance and tunes hyperparameters.
- Serializes the trained model to disk as **`model/model.pkl`**.

This serialized model is what the Streamlit app uses at runtime.

---

## Interactive App (`app.py`)

The file `app.py` serves the user interface via **Streamlit**.

Overall Project Flow:

1. **Load artifacts**
   - Load `df_states` from `df_states.csv`.
   - Add derived features such as `years_from_start`, `cause_short`, and numeric `deaths_num`.
   - Load the trained LightGBM model from `model/model.pkl`.

2. **Baseline projection**
   - Fix a **baseline risk-factor year** (e.g. 2020) and project YPLL forward to 2030.
   - Use the model to compute **baseline 2030 YPLL** (`baseline_total`) by state.

3. **Cause-of-death decomposition**
   - Use observed state-level death counts by cause to compute **cause shares** within each state.
   - Allocate the baseline 2030 YPLL to each of the five causes using those shares.

4. **Slider-driven scenario**
   - The UI exposes sliders for each major cause:
     - Cancer deaths change (%)
     - Heart disease deaths change (%)
     - Stroke deaths change (%)
     - Lower respiratory deaths change (%)
     - Accident deaths change (%)
   - Slider values are interpreted as **10-year percentage changes in deaths for that cause**  
     (negative = fewer deaths, positive = more deaths).

5. **Scaling the model output**
   - For each state and cause, apply a multiplicative **scaler** derived from the slider:
     - `factor = 1 + (slider_value / 100)`
   - Multiply baseline cause-level YPLL by these factors to get **adjusted 2030 YPLL**.
   - Aggregate adjusted YPLL back to the state level.

6. **Map and metrics**
   - Compute **years of potential life gained** as:
     - `years_gained = baseline_total - adjusted_total`
   - Render a U.S. choropleth using Plotly, colored by `years_gained`:
     - Green = more years of life gained (Years of Potential Life Gained),
     - Red = years lost (Years of Potential Life Lost).
   - Display summary metrics:
     - Total baseline YPLL (2030),
     - Total adjusted YPLL (2030),
     - Total years of potential life gained,
     - Approximate percentage change vs. baseline.

---

## Repository Structure (simplified)

```text
.
├── app.py                    # Streamlit UI
├── UCD.py                    # Data cleaning and feature engineering
├── df_states.csv             # Processed modeling dataset
├── model/
│   └── model.pkl             # Trained LightGBM model
├── Phillip's_Code_(Model).ipynb  # Modeling / training notebook
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation (this file)