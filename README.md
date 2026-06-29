# 🇮🇹 Airbnb Market Analysis & Business Optimization — Rome

This repository contains a comprehensive data science and business strategy analysis of the Airbnb marketplace in Rome, Italy. Developed as part of the **Data Science for Businesses** course.

---

## 🎯 Project Objective & Business Case
Rome is one of the most heavily visited tourist destinations globally, making its short-term rental market highly lucrative but intensely competitive. 

The objective of this project is two-fold:
1. **Technical Foundation:** Clean, preprocess, and segment the spatial real estate landscape of Rome using structured data pipelines.
2. **Business Case:** Deliver actionable, data-driven insights to help property management companies and real estate investors maximize yield, optimize availability strategies, and understand geographical pricing power.

---

## 📂 Repository Structure
```text
├── data/                      # Local raw datasets (git-ignored)
├── notebooks/
│   ├── 01_calendar_cleaning.py   # Calendar timeseries preprocessing
│   ├── 02_listings_cleanings.py  # Property attributes exploration
│   ├── 03_rome_geo_analysis.py   # Strategic proximity clustering & spatial EDA
│   └── 04_rome_classification.py # Scikit-learn predictive pipeline & hyperparameter tuning
├── README.md                  # Executive summary and project guide
└── .gitignore                 # Safe deployment configuration

---

## 🛠️ Data Pipeline & Methodology

### 1. Data Preparation & Exploration
* **Data Cleansing:** Standardized financial inputs by stripping currency symbols and parsing raw string objects into clean, operable numeric features.
* **Outlier Mitigation:** Handled premium pricing outliers using a strict 95th percentile clip to ensure robust statistical representations of standard market conditions.

### 2. Feature Engineering: Strategic Proximity Clustering
* Rather than relying on black-box coordinate groupings, we engineered a **Business-Driven Market Segmentation** model centered around the commercial heart of Rome's tourism: **The Colosseum**.
* Properties were classified into **5 Macro Business Zones** based on spatial Euclidean distance, revealing clear pricing power drop-offs and listing density thresholds as properties move away from the historical core.

### 3. Predictive Modeling (Focus Area: Classification & Tuning)
* **Target Objective:** Predict whether a listing operates under a high-availability enterprise strategy (`availability_365 > 180 days`) versus a highly seasonal or organic structure.
* **The Pipeline:** Implemented a formal `scikit-learn Pipeline` that encapsulates automated feature scaling (`StandardScaler`) alongside a `LogisticRegression` classifier.
* **Hyperparameter Optimization:** Utilized `GridSearchCV` with 5-fold cross-validation (`cv=5`) to optimize regularization strengths (C) and penalty types (l1 vs l2), eliminating data leakage and ensuring robust model generalization.

---

## 📈 Key Business Insights & Artifacts
The pipeline automatically generates and exports strategic visualization assets to support our investment deck:
* `roma_top_neighborhoods.png`: Barplot profiling the top 10 premium neighborhoods by median price.
* `roma_price_map.png`: A high-resolution spatial heatmap displaying the geographical clustering of nightly rates.
* `roma_business_segments.png`: Visual boundaries of our 5 engineered macro investment zones relative to the tourism core.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.12+ installed. Install the necessary data science stack via your terminal:
```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
Execution
Run the pipelines sequentially from the project root:

Bash
python notebooks/03_rome_geo_analysis.py
python notebooks/04_rome_classification.py