# A Longer Tomorrow: Visualizing Future Years of Potential Life Lost

---

DESCRIPTION
-----------
"A Longer Tomorrow" is an interactive tool for exploring how changes in major causes of death could affect future Years of Potential Life Lost (YPLL) across U.S. states.

Using historical CDC mortality data (1999–2020) from the CDC WONDER Underlying Cause of Death (UCD) system and state-level behavioral / socioeconomic risk factors (e.g., obesity, smoking, insurance coverage), we train a LightGBM model to predict YPLL rates per 100,000 for 2030 by state, sex, and cause of death. The Streamlit app then loads a future-state dataset, applies the trained model, and decomposes projected YPLL across five leading causes of death:

- Cancer
- Heart Disease
- Stroke
- Chronic Lower Respiratory Disease
- Accidents

In the UI, users adjust sliders that represent 10-year percentage changes in deaths for each cause (e.g., “-15%” cancer deaths). These changes are translated into scaled YPLL projections for 2030, and the app visualizes, for each state, how many years of potential life are gained or lost relative to the baseline model using a U.S. choropleth and summary metrics.

Data Sources
1. CDC WONDER – Underlying Cause of Death (UCD)
   Original mortality data are obtained from the CDC WONDER Underlying Cause of Death system for 1999–2020.
   - Saved CDC WONDER UCD query:
     https://wonder.cdc.gov/controller/saved/D76/D456F892

   Key settings:
   - Dataset: Underlying Cause of Death, 1999–2020
   - Underlying Cause of Death: UCD–ICD–10 113 Cause List
   - Geography: States
   - Strata: Sex, State (no race stratification)
   - Age grouping: 10-year age groups (e.g., 15–24, 25–34, …, 75–84)
   - Age restriction: adult age groups up to 75–84, with YPLL calculated using a 75-year upper age limit
   - Underlying causes restricted to the top 5 UCDs:
     * #Diseases of heart (I00-I09,I11,I13,I20-I51)
     * #Malignant neoplasms (C00-C97)
     * #Cerebrovascular diseases (I60-I69)
     * #Chronic lower respiratory diseases (J40-J47)
     * #Accidents (unintentional injuries) (V01-X59,Y85-Y86)

2. CDC BRFSS – Behavioral Risk Factor Surveillance System
   To enrich mortality data with health and behavioral risk factors, we use:
   - BRFSS data: https://www.cdc.gov/brfss/annual_data/annual_data.htm
   Example state-year features:
   - obesity_pct, uninsured_pct, income_mean, employed_pct, diabetes_pct,
     educa_z, smoking_pct_z, binge_drink_pct_z, seatbelt_always_pct_z

Data Processing (UCD.py)
- Import raw CDC WONDER UCD extracts and aggregate to state–year–cause level.
- Join BRFSS-derived features by state and year.
- Compute YPLL-related outcomes and convenience features such as years_from_start.
- Produce a modeling dataset (df_states) and a future-projection dataset (future_df) that are used by the modeling notebook and the Streamlit app.

Data Access
-----------
The raw and intermediate data files used to build df_states and future_df are stored in a shared data folder:

https://gtvault.sharepoint.com/:f:/s/cse6242dvagroup1/IgCg1dBTRITKQpkpoXQUPKevAbGiPIyYJ86w9uAxCfitTz8?e=hdwaJS

These data are **not required** to run the hosted UI. However, if you wish to:
- Re-run UCD.py, or
- Re-train / modify the model in Phillip's_Code_(Model).ipynb,

you will need to download the data from that location.

---

INSTALLATION
------------
Quick start – hosted application
- If you only want to explore the interactive UI (no local setup required), visit:
  https://longertomorrow-app.streamlit.app/

Local setup (optional)
If you would like to run the application locally or inspect the code:

1. Clone the repository:
   - GitHub repo:
     https://github.com/smileshey/LongerTomorrow/tree/main

   From a terminal:
   - git clone https://github.com/smileshey/LongerTomorrow.git
   - cd LongerTomorrow

2. Ensure you have Python 3.9+ installed.

3. (Recommended) Create and activate a virtual environment:
   - python -m venv .venv
   - On macOS / Linux: source .venv/bin/activate
   - On Windows (PowerShell): .venv\Scripts\activate

4. Install dependencies:
   - pip install -r requirements.txt

Key Files
- UCD.py  
  Handles data cleaning and feature engineering:
  - Reads raw CDC WONDER UCD and BRFSS data.
  - Aggregates to state–year–cause level and computes YPLL-related quantities.
  - Outputs df_states and future_df for modeling and visualization.

- Phillip's_Code_(Model).ipynb  
  Trains the LightGBM model on df_states:
  - Defines features (year, state, UCD, risk factors, etc.).
  - Encodes categorical variables and trains a LightGBM regressor.
  - Evaluates performance and serializes the model to model/model.pkl.

- model/model.pkl  
  The trained LightGBM model loaded by the Streamlit app.

- data/future_df.csv  
  The future projection dataset (2021–2030) used as input to model/model.pkl
  to generate 2030 YPLL predictions for each (state, sex, cause) combination.

- app.py  
  The Streamlit application that loads future_df and model.pkl,
  computes projected YPLL for 2030, applies user-selected scenarios,
  and renders the interactive visualization.

---

EXECUTION
---------
Using the hosted app (recommended):
- Open the deployed Streamlit app in your browser:
  https://longertomorrow-app.streamlit.app/

- The app has three main sections accessible from the left sidebar:
  1) Introduction
     - Provides background on YPLL, data sources, and the modeling approach.
  2) Modeling Years of Life Gained
     - Shows the interactive choropleth and sliders for:
       * Cancer deaths change (%)
       * Heart disease deaths change (%)
       * Stroke deaths change (%)
       * Lower respiratory deaths change (%)
       * Accident deaths change (%)
     - Sliders represent 10-year percentage changes in deaths for each cause
       (negative = fewer deaths → lower YPLL; positive = more deaths → higher YPLL).
  3) Conclusion
     - Summarizes key insights from the model and visualization.

Running locally (optional):
1. From the project root (the directory containing app.py), run:
   - streamlit run app.py

2. Streamlit will print a local URL (typically http://localhost:8501). Open it in a browser.

Interactive behavior (app.py):
- Load data
  * Read data/future_df.csv and model/model.pkl (LightGBM regressor).

- Baseline projection (2030)
  * Use the model to predict baseline 2030 YPLL rates per 100,000 for each state, sex, and cause.
  * Aggregate to get baseline_total YPLL per state.

- Cause-of-death decomposition
  * Decompose each state’s baseline_total into cause-level contributions using cause_short
    (Cancer, Heart Disease, Stroke, Chronic Lower Respiratory Disease, Accidents).

- Sliders
  * Apply user-selected percentage changes in deaths by cause to scale cause-specific YPLL.

- Scaling
  * For each state and cause, compute:
    factor = 1 + (slider_value / 100)
  * Multiply baseline cause-level YPLL by factor to obtain adjusted 2030 YPLL.
  * Aggregate adjusted YPLL back to the state level.

- Map and metrics
  * Compute:
    years_gained = baseline_total - adjusted_total
  * Render a U.S. choropleth where:
    - Green = more years of life gained (lower YPLL vs baseline),
    - Red = years of life lost (higher YPLL vs baseline).
  * Show summary metrics:
    - Total baseline YPLL (2030),
    - Total adjusted YPLL (2030),
    - Total years of potential life gained,
    - Percent change relative to baseline.

---

DEMO VIDEO
---------------------
In place of a demo video, our team has assembled an interactive UI that abstracts away most of the setup and allows you to explore the model directly:

https://longertomorrow-app.streamlit.app/
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