# Airbnb Rome — Price Prediction

Predicting the nightly price of Airbnb listings in Rome using listing, host, review, and
geographic features. Built for the Data Science for Business case study.

## Business Understanding

Airbnb hosts and the platform itself need a reliable way to estimate what a listing "should"
cost given its characteristics — its size, location, host reputation, and guest demand signals.
This project builds a data pipeline and three predictive models (Random Forest, Linear
Regression, and XGBoost) that take the raw, messy Inside Airbnb export for Rome and turn it into
a clean, feature-rich dataset that can explain and predict nightly price.

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
│   ├── 06_random_forest.ipynb
│   ├── 07_linear_regression_model.ipynb
│   └── 08_XGBoost_model.ipynb
├── Case Study DSFB.pdf          # assignment brief
├── requirements.txt
└── README.md
```

## Pipeline Overview

```
01_calendar_cleaning ──────────────────────────┐
                                                 │ (confirms target = listings.price)
02_listings_cleaning ──► listings_features.csv ─┤
                                                 ├─► 05_final_dataset ──► final_dataset.csv ──┬─► 06_random_forest
03_reviews_analysis ───► review_features.csv ───┘                                            ├─► 07_linear_regression_model
                                                                                               └─► 08_XGBoost_model

04_time_analysis (external tourism seasonality — exploratory, not merged into the modeling dataset)
```

Each notebook reads from `../data/` and, where relevant, writes its output back to `../data/`
as a CSV, so the next notebook in the chain can pick it up without re-running everything
upstream. Notebooks 06, 07, and 08 all read the same `final_dataset.csv` produced by 05, so
their results are directly comparable.

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

> **Known data-quality caveat:** `host_response_rate`, `host_acceptance_rate`, and
> `host_experience_days` currently come through 100% empty for every listing in the merged
> dataset. This doesn't break Random Forest or XGBoost (both tolerate missing values natively),
> but it does break Linear Regression, which requires these columns to be dropped explicitly
> (see Notebook 07 below). Worth tracing back to the source column/merge in this notebook if
> there's time before submission.

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
- Exports the result to `data/final_dataset.csv`, the single input used by all three modeling
  notebooks (06, 07, 08) from here on.

## Notebook 06 — Random Forest Model

The first predictive model, using `final_dataset.csv`:


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

## Notebook 07 — Linear Regression Model

A linear baseline against Random Forest, using the **same** `final_dataset.csv`, the same 1st–
99th percentile outlier trimming on `price`, and the same 80/20 split (`random_state=42`) as
Notebook 06, so results are directly comparable.


> Because of this, Notebook 07's feature matrix has 3 fewer columns than Notebook 06's (which
> keeps those empty columns — Random Forest tolerates them natively). The effect on Random
> Forest's predictions is negligible, but it's worth knowing the two notebooks aren't running on
> a byte-for-byte identical `X` until that upstream data-quality issue is fixed in Notebook 02.

**Modeling:** a baseline `LinearRegression` (with `StandardScaler`), then — since plain OLS has
no hyperparameters to tune — a **Ridge Regression** (L2-regularized linear model) tuned over its
`alpha` via `RandomizedSearchCV` (9 iterations, 5-fold CV), to give Linear Regression a fair
"optimized" stage comparable to Random Forest and XGBoost's tuning steps while keeping the model
linear.

**Results (test set):**

| Metric | Naive baseline (predict the mean) | Linear Regression (baseline) | Ridge (tuned) |
|---|---|---|---|
| RMSE | *(fill in after final run — see Section 8)* | *(fill in)* | *(fill in)* |
| MAE | — | *(fill in)* | *(fill in)* |
| R² | — | *(fill in)* | *(fill in)* |

*(An earlier, differently-preprocessed run of this notebook produced RMSE = 89.78, MAE = 55.79,
R² = 0.5062 on a plain `LinearRegression` — kept here as a sanity-check reference point, not the
final reported result. Replace the table above with the numbers from the current pipeline once
it's run end-to-end.)*

If Linear Regression's R² comes out clearly lower than Random Forest's, that's expected, not a
bug — it's evidence that price doesn't relate to these features in a purely linear way, which is
part of the argument for also using Random Forest and XGBoost. The notebook's residual analysis
(predicted vs. actual, residuals vs. predicted, residual distribution) is there to check for
exactly this: if residuals fan out at higher prices or look skewed, that confirms a non-linear /
heteroscedastic relationship worth calling out in the presentation.

**Interpretability:** since numeric features are standardized, the Ridge coefficients are
directly comparable in magnitude and serve as the Linear Regression counterpart to the
feature-importance rankings in Notebooks 06 and 08 — useful for cross-checking which features
the three models agree are important, and which ones only the tree-based models pick up (a
signal of non-linear or interaction effects).

## Notebook 08 — XGBoost Model

# XGBoost Regressor: Overview and Parameter Guide

## What is XGBoost Regressor?

**XGBoost** (eXtreme Gradient Boosting) is an optimized, highly efficient implementation of the **Gradient Boosted Decision Trees (GBDT)** algorithm designed for speed, scalability, and state-of-the-art predictive performance.

Unlike **Random Forest** (which trains multiple deep decision trees independently in parallel and averages their predictions), **XGBoost builds shallow decision trees sequentially**. 

### How it works:
1. It starts with a baseline prediction (e.g., the average price of all listings).
2. It calculates the prediction errors (residuals) for every data point.
3. It trains a new shallow tree **specifically to predict those residuals** (correcting the mistakes of the previous trees).
4. It updates the overall prediction by adding the new tree's output, scaled by a learning rate.
5. This process repeats sequentially for a specified number of iterations.

---

## Core Parameters Explained

Below is a precise breakdown of the parameters configured in your model:

### 1. `n_estimators=100` (Number of Trees)
* **What it does:** Sets the total number of sequential decision trees to be built during training.
* **The Mechanism:** Because boosting is additive, trees are built one after another. 
* **The Impact:** 
  * If set **too low**, the model will underfit and fail to capture complex relationships.
  * If set **too high**, the model will overfit by capturing random noise in the training set, and training will take significantly longer.

### 2. `learning_rate=0.1` (Shrinkage / Step Size)
* **What it does:** Scalese the contribution of each new tree added to the model (technically called $\eta$).
* **The Mechanism:** When a new tree is trained to correct the errors of previous trees, its predictions are multiplied by the `learning_rate` (e.g., $0.1$ or $10\%$) before being added to the final score. This forces the model to take small, conservative steps toward the optimal solution.
* **The Impact:** 
  * Lower values (like $0.05$ or $0.1$) make the learning process more robust and less prone to overfitting, but require a higher `n_estimators` to fully converge.

### 3. `max_depth=6` (Maximum Tree Depth)
* **What it does:** Restricts the maximum number of splits (levels) allowed for any single decision tree in the ensemble.
* **The Mechanism:** A tree with a `max_depth=6` can split up to 6 consecutive times, creating a maximum of $2^6 = 64$ terminal leaf nodes.
* **The Impact:** 
  * Higher values allow individual trees to capture highly specific multi-variable interactions (raising the risk of overfitting).
  * Values between $3$ and $9$ are typically optimal for tabular data.

### 4. `random_state=42` (Reproducibility Seed)
* **What it does:** Sets a constant seed for the internal pseudo-random number generator.
* **The Mechanism:** XGBoost uses stochastic processes (such as random row/column sampling) to improve generalization. Fixing this seed ensures that these random choices are identical every time the code runs.
* **The Impact:** Guarantees that your performance metrics (MAE, RMSE, $R^2$) remain completely identical across different runs, notebooks, and machines.

### 5. `n_jobs=-1` (Parallel Execution)
* **What it does:** Controls the number of CPU threads allocated to run the training process.
* **The Mechanism:** Setting `n_jobs=-1` instructs your operating system to utilize all available CPU cores in parallel.
* **The Impact:** Drastically reduces training time—especially during hyperparameter tuning (like `RandomizedSearchCV`)—without affecting the mathematical accuracy of the final model.

---

## Parameter Quick-Reference

| Parameter | Default Value | Standard Tuning Range | Primary Objective |
| :--- | :---: | :---: | :--- |
| **`n_estimators`** | `100` | `100 - 1000` | Controls overall model capacity and training duration. |
| **`learning_rate`** | `0.3` | `0.01 - 0.2` | Slows down learning speed to improve generalization. |
| **`max_depth`** | `6` | `3 - 10` | Limits the complexity of individual trees to prevent overfitting. |
| **`random_state`** | `None` | Any Integer | Ensures reproducibility of results across runs. |
| **`n_jobs`** | `None` | `-1` | Accelerates computations using multi-core processing. |
A gradient-boosted trees model (`XGBRegressor`), evaluated against the same `final_dataset.csv`
and the same 80/20 split (`random_state=42`) as Notebooks 06 and 07.



> **Alignment caveat:** unlike Notebooks 06 and 07, this notebook does **not** trim `price` to
> the 1st–99th percentile before splitting. That means Notebook 08 is training and evaluating on
> a different (larger, outlier-inclusive) set of rows than 06 and 07 — worth fixing before the
> final model comparison table goes into the presentation, or the RMSE/MAE numbers won't be
> apples-to-apples.

**Modeling:** a baseline `XGBRegressor` (`n_estimators=100`, `learning_rate=0.1`, `max_depth=6`),
then hyperparameter tuning via `RandomizedSearchCV` (15 iterations, tuning `n_estimators`,
`learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`,
scored on negative MAE). The notebook also includes an exploratory side-by-side comparison of 5
manually chosen hyperparameter "scenarios" (underfit → overfit) to illustrate the bias-variance
trade-off for the presentation.

**Results (test set, tuned model):**

| Metric | Naive baseline (predict the mean) | XGBoost (tuned) |
|---|---|---|
| RMSE | *(fill in after final run — see Section 7)* | *(fill in)* |
| MAE | — | *(fill in)* |
| R² | — | *(fill in)* |
| MAPE | — | *(fill in)* |

**Interpretability:** feature importances are extracted from the fitted `XGBRegressor` and
compared against the one-hot-encoded feature names produced by the `ColumnTransformer`, giving
a ranked list of the strongest price drivers — to be compared against Notebooks 06 and 07's
rankings in the Feature Selection section of the presentation.

---

## Model Comparison Summary

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Naive baseline (predict the mean) | ~121–128 *(varies slightly by outlier trim / row count)* | — | ~0 |
| Random Forest (tuned) | **79.47** | **45.46** | **0.568** |
| Linear Regression / Ridge (tuned) | *(fill in)* | *(fill in)* | *(fill in)* |
| XGBoost (tuned) | *(fill in)* | *(fill in)* | *(fill in)* |

Fill in the Linear Regression and XGBoost rows once both notebooks have been run end-to-end on
the same, finalized `final_dataset.csv` (and once Notebook 08's outlier trimming is aligned with
06/07 — see the caveat above). Use this table, together with each notebook's feature-importance
/ coefficient ranking, for the "Model Selection" and "Reasoning" parts of the presentation.

## How to Reproduce

1. Download `listings.csv.gz`, `reviews.csv.gz`, `calendar.csv.gz`, and `neighbourhoods.geojson`
   for **Rome** from [Inside Airbnb](https://insideairbnb.com/get-the-data/) into your local
   `data/` folder (not committed to git).
2. `pip install -r requirements.txt`
3. Run the notebooks in order: `01` → `02` → `03` → `05` → `06` → `07` → `08`. (`04` can be run
   independently at any point — it only depends on an external tourism CSV, not on the other
   notebooks. `06`, `07`, and `08` can be run in any order relative to each other, as long as
   `05` has already produced `final_dataset.csv`.)

## Team Workflow Reminder

- `git pull` before starting work, `git add . && git commit -m "..." && git push` when done.
- Keep your own `data/` folder locally; CSVs are gitignored.
- If two people need to edit the same notebook, coordinate first or work on a copy — notebook
  JSON doesn't merge cleanly with git, and unresolved conflicts will corrupt the file.