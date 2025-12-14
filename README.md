# Business Data Science Project – Inside Airbnb

This repository contains the material for the Business Data Science project carried out by Anastasia Bouev-Dombre, Arthur Morvan, Aliénor Sabourdin, and Elise Deyris. We work with the multi-city **Inside Airbnb** listings, calendar, and review feeds and focus on the city of **Bordeaux, France**. The goal is to build a data pipeline that cleans the raw data, engineers demand-driven features, and trains predictive models that power a dynamic pricing assistant for hosts.

## Repository layout
```
Business Data Science Project Airbnb.ipynb  # end-to-end exploration, cleaning, modeling, and pricing tool
pyproject.toml / requirements.txt           # Python dependencies (uv/PEP 621 or pip requirements)
data/                                       # Inside Airbnb CSV exports (+ gzipped copies)
data_transorm.py                            # helper to gzip large CSVs for faster loading
data.zip                                    # compressed copy of the raw data folder
```

## Data
We use the Inside Airbnb data dump (listings, calendar, reviews) restricted to Bordeaux. The raw CSVs live in `data/` alongside `.gz` copies for faster reads from pandas.

| File | Description |
| --- | --- |
| `data/Airbnb-Listings-Data.csv[.gz]` | Listing-level metadata, host profile, amenities, prices. |
| `data/Airbnb-Calendar-Data.csv[.gz]` | Daily availability and advertised price per listing. |
| `data/Airbnb-Reviews.csv[.gz]` | Textual reviews with timestamps and reviewer metadata. |

> To work with another city, download the corresponding Inside Airbnb dump and drop the three CSVs into `data/`. Update the `bordeaux_bound` helper in the notebook to match the city limits you care about.

## Environment setup
Python 3.10+ works well (the `pyproject` targets 3.13 but the analysis was developed with 3.10/3.11). Choose either workflow:

**Using uv (recommended)**
```bash
uv sync
uv run jupyter lab  # or: uv run jupyter notebook
```

**Using pip**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=airbnb-bds
jupyter lab  # or: jupyter notebook
```

## Reproducing the analysis
1. Install the environment (see above).
2. (Optional) Gzip large CSVs with `python data_transorm.py data/Airbnb-Listings-Data.csv data/Airbnb-Calendar-Data.csv data/Airbnb-Reviews.csv`.
3. Launch Jupyter and open `Business Data Science Project Airbnb.ipynb`.
4. Run all cells (Kernel → Restart & Run All) to recreate the cleaned datasets, train the models, and render the pricing tool case studies.

The notebook is organized as follows:
- **Cleaning** – format prices/percentages, convert booleans, parse dates, impute missing values, remove duplicates, and clip listings to the Bordeaux bounding box.
- **Merging & Feature Engineering** – join calendar/review aggregates onto each listing, build demand proxies (availability windows, historical occupancy, revenue, seasonality flags), host reliability flags, and textual sentiment signals.
- **Modelisation and Prediction** – two tasks: (1) predicting the price at which a listing gets booked and (2) predicting the probability that a given date is booked.
- **Valuable Business Insights** – wrap the classification model in a pricing simulation that scans price grids and surfaces the revenue-maximizing choice. Includes high-demand, low-demand, and seasonality case studies.

## Modeling summary
### 1. Price of booked listings (regression)
- **Target**: `price_clean` for rows known to be booked.
- **Features**: normalized property/host metadata (capacity, amenities, superhost status), spatial buckets, review volumes and ratings, and demand proxies such as `availability_30`, calendar seasonality, and historic revenue.
- **Baseline**: `LinearRegression` after removing collinear columns with Variance Inflation Factor (MAE **€35.40** on the held-out set).
- **Final model**: `HistGradientBoostingRegressor` with monotonic-friendly features (MAE **€13.69**). This captures nonlinear interactions between capacity, location, and host reputation.

### 2. Booking probability per date (classification)
- **Target**: `is_booked` derived from the calendar (`availability_365 == 0` ⇒ booked).
- **Features**: price, lead time, rolling availability, demand from nearby listings, host signals, historical reviews, and engineered seasonality.
- **Baseline**: `LogisticRegression` (`ROC AUC = 0.652`, `Log loss = 0.629`).
- **Improved model**: `HistGradientBoostingClassifier` plus sigmoid calibration on a validation fold (`ROC AUC = 0.841`, `PR AUC = 0.915`, `Log loss = 0.537`). Permutation importance in the notebook highlights price levels, short-term availability, review volume, and superhost status as key drivers.

### Pricing assistant
We combine the calibrated booking model with a simple simulator (`simulate_pricing_curve`) that:
1. Builds a grid of candidate prices (±30–40% around the current price).
2. Scores booking probability for each price.
3. Computes expected revenue (price × booking probability) and picks the argmax.

The notebook contains three case studies (high demand, low demand, and seasonality shift) to illustrate how the tool raises or lowers prices while keeping expected occupancy in check. Visual diagnostics include booking-probability vs. price and expected-revenue vs. price curves.

## Helper script: `data_transorm.py`
`data_transorm.py` is a tiny utility that gzips one or multiple CSV files in-place:
```bash
python data_transorm.py data/Airbnb-Listings-Data.csv data/Airbnb-Calendar-Data.csv
```
Each input file is copied to `<file>.gz`. Use it whenever you refresh the raw Inside Airbnb dump.

## Next steps
- Extend the geographic filters and feature engineering helpers to support multiple cities simultaneously.
- Persist the cleaned datasets and trained models (e.g., Parquet + `joblib`) so the pricing tool can be served outside of the notebook.
- Experiment with alternative calibration strategies (isotonic) and cost-sensitive objectives that better reflect hosts’ risk tolerance.
