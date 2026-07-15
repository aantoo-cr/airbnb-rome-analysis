# Airbnb Rome — Price Prediction

Predicting the nightly price of Airbnb listings in Rome using listing, host, review, and
geographic features. Built for the Data Science for Business case study.

## Business Understanding

Airbnb hosts and the platform itself need a reliable way to estimate what a listing "should"
cost given its characteristics — its size, location, host reputation, and guest demand signals.
This project builds a data pipeline and a first predictive model that takes the raw, messy
Inside Airbnb export for Rome and turns it into a clean, feature-rich dataset that can explain
and predict nightly price.

## Data Source

Data comes from [Inside Airbnb](https://insideairbnb.com/get-the-data/) for Rome:

| File | Description |
|---|---|
| `listings.csv.gz` | One row per listing: property, host, review, and availability attributes, including price. |
| `reviews.csv.gz` | One row per individual guest review, with the review text and date. |
| `calendar.csv.gz` | Date-level availability per listing. **Does not contain price/adjusted_price for Rome** — see Notebook 01. |
| `neighbourhoods.geojson` | Neighbourhood boundary shapes (not currently used in the modeling pipeline). |

CSVs are not committed to the repo (they're too large) — each contributor keeps their own copy
in a local `data/` folder, per the group's git workflow.

## Repository Structure

```
airbnb-rome-analysis/
├── data/                        # local only, not committed (raw + generated CSVs)
├── notebooks/
│   ├── 01_calendar_cleaning.ipynb
│   ├── 02_listings_cleaning.ipynb
│   ├── 03_reviews_analysis.ipynb
│   ├── 04_time_analysis.ipynb
│   ├── 05_final_dataset.ipynb
│   └── 06_random_forest.ipynb
├── Case Study DSFB.pdf          # assignment brief
├── requirements.txt
└── README.md
```

## Pipeline Overview

```
01_calendar_cleaning ──────────────────────────┐
                                                 │ (confirms target = listings.price)
02_listings_cleaning ──► listings_features.csv ─┤
                                                 ├─► 05_final_dataset ──► final_dataset.csv ──► 06_random_forest
03_reviews_analysis ───► review_features.csv ───┘

04_time_analysis (external tourism seasonality — exploratory, not merged into the modeling dataset)
```

Each notebook reads from `../data/` and, where relevant, writes its output back to `../data/`
as a CSV, so the next notebook in the chain can pick it up without re-running everything
upstream.

---

## Notebook 01 — Calendar Cleaning

Quick exploration of `calendar.csv.gz`. The key outcome: **this file has no usable price
information for Rome** — `price` and `adjusted_price` are entirely missing, it only reliably
contains `listing_id`, `date`, `available`, `minimum_nights`, and `maximum_nights`. This
notebook is what settled the target variable decision for the whole project: **price prediction
uses the static `price` from `listings.csv`, not a date-level calendar price.**

## Notebook 02 — Listings Cleaning & Feature Engineering

The core data preparation notebook. Starting from the raw `listings.csv.gz` (75+ columns), it:

1. **Cleans the target** — parses `price` out of its currency-string format (`$1,234.00` → `1234.0`) and drops rows with no price.
2. **Selects candidate features** — 28 columns chosen based on domain knowledge (Airbnb's own pricing documentation) and the project's focus areas, explicitly excluding identifiers, free text, images, and admin fields that don't explain price.
3. **Handles missing values** — median imputation for numeric fields (bedrooms, bathrooms, review scores, etc.), `"Unknown"` category for missing categoricals.
4. **Engineers new features**, grouped by theme:
   - **Property**: `beds_per_guest`, `bathrooms_per_guest` — capacity normalized by group size.
   - **Amenities**: `amenities_count` plus binary flags (`has_wifi`, `has_kitchen`, `has_parking`, etc.) extracted from the free-text amenities list.
   - **Host**: `host_experience_days` (from `host_since`), `professional_host` (hosts with 5+ listings), superhost encoded as 0/1.
   - **Property type**: dozens of raw categories consolidated into 4 broad classes (Apartment, Private Room, Shared Room, Hotel, Entire Home).
   - **Reviews**: `review_quality_index` (composite of the 4 review sub-scores), `listing_age_days`, `review_recency_days`, `review_intensity`.
   - **Availability**: `occupancy_rate`, `demand_proxy` (combines past occupancy with future booking pressure).
   - **Geographic**: one-hot encoded neighbourhoods, `distance_to_colosseum` (geodesic distance to Rome's most iconic landmark), plus **K-Means and DBSCAN clustering** on standardized lat/long to capture spatial pricing patterns that raw neighbourhood names miss. K-Means (k=5) was chosen for modeling over DBSCAN, since DBSCAN mostly finds one dense central cluster plus noise, while K-Means gives a more usable, balanced segmentation.
5. **Exports** the final feature set to `data/listings_features.csv` (id, price, and ~50 engineered features, one row per listing).

## Notebook 03 — Reviews Analysis

Transforms the raw, review-level `reviews.csv.gz` into listing-level features:

- `review_length` per review (word count of the comment).
- Aggregated per `listing_id`: `review_count`, `avg_review_length`, and `latest_review_date`.

Exports to `data/review_features.csv`.

## Notebook 04 — Time Analysis

Exploratory look at external tourism seasonality data for Rome (arrivals/overnights by month),
used to build a 0–1 seasonality index. This is contextual/business-case material for the
presentation rather than a per-listing feature — it doesn't get merged into the modeling
dataset, since the static `price` target has no date dimension for it to attach to.

## Notebook 05 — Final Dataset Construction

Merges the two feature sets produced above into one modeling-ready table:

- Left join of `listings_features.csv` (key: `id`) with `review_features.csv` (key:
  `listing_id`), keeping all listings even if they have no reviews.
- Listings without reviews get `review_count` / `avg_review_length` filled with `0` (absence of
  review activity, not missing data).
- Adds `days_since_latest_review`, with a placeholder of `9999` for listings with no reviews.
- Exports the result to `data/final_dataset.csv`, the single input used by the modeling
  notebooks from here on.

## Notebook 06 — Random Forest Model

The first predictive model, using `final_dataset.csv`:

**Preprocessing:**
- Median-imputes any remaining missing numeric values.
- Decomposes `latest_review_date` into year/month/day; derives `days_since_last_review` from `last_review`.
- Drops the listing identifier.
- One-hot encodes `property_type`, `room_type`, and `instant_bookable` (`drop_first=True`).
- Trims `price` to the 1st–99th percentile to reduce the influence of extreme luxury/outlier listings.
- 80/20 train/test split (`random_state=42`).

**Modeling:** a baseline `RandomForestRegressor`, then hyperparameter tuning via
`RandomizedSearchCV` (20 iterations, 5-fold CV, tuning `n_estimators`, `max_depth`,
`min_samples_split`, `min_samples_leaf`, `max_features`).

**Results (test set, tuned model):**

| Metric | Naive baseline (predict the mean) | Random Forest (tuned) |
|---|---|---|
| RMSE | 120.94 | **79.47** |
| MAE | — | **45.46** |
| R² | — | **0.568** |

The model explains roughly 57% of the variance in listing price — a solid first result given
how heterogeneous Airbnb pricing behavior tends to be.

**Top predictors** (by feature importance): `bathrooms`, `bedrooms`, `accommodates`,
`distance_to_colosseum`, `longitude`/`latitude`, the Centro Storico neighbourhood indicator,
`beds`, `review_scores_location`, and `demand_proxy`. In short: **size and capacity, location,
and demand/reputation signals** are the three main drivers the model picked up.

---

## How to Reproduce

1. Download `listings.csv.gz`, `reviews.csv.gz`, `calendar.csv.gz`, and `neighbourhoods.geojson`
   for **Rome** from [Inside Airbnb](https://insideairbnb.com/get-the-data/) into your local
   `data/` folder (not committed to git).
2. `pip install -r requirements.txt`
3. Run the notebooks in order: `01` → `02` → `03` → `05` → `06`. (`04` can be run independently
   at any point — it only depends on an external tourism CSV, not on the other notebooks.)

## Team Workflow Reminder

- `git pull` before starting work, `git add . && git commit -m "..." && git push` when done.
- Keep your own `data/` folder locally; CSVs are gitignored.
- If two people need to edit the same notebook, coordinate first or work on a copy — notebook
  JSON doesn't merge cleanly with git, and unresolved conflicts will corrupt the file.

