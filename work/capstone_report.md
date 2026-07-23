# Capstone Report — Ranking Signal Analysis

- **Author:** Antigravity (Agent)
- **Lane:** Ranking Signal Analysis
- **Repo:** flyrank-ml-internship-starter
- **Date:** 2026-07-21

## 0. Abstract

What signals drive average search position, and how can we use them to prioritize content for refresh? By analyzing a 50,000-row anonymized 90-day search dataset from FlyRank, we modeled the relationship between metrics like word count, search volume, and average position. We used a Random Forest Regressor trained on a client-grouped cross-validation split to prevent leakage and ensure out-of-sample validity. Our signal analysis measured a directional relationship between these features and search ranking, which we translated into a ranked 'Action Playbook'. This output provides content teams with decision-support to target striking-distance and low-CTR pages over manual guesswork.

## 1. Problem framing

**Decision:** Which specific pages should a human content strategist review and refresh first out of thousands?
**Unit of Analysis:** Individual page/content item (`content_id`).
**Output:** A ranked 'Action Playbook' queue and directional feature importances.
**Action:** The user will rewrite Meta Data for high-ranking/low-CTR pages, or expand word count/add links for striking-distance pages.
**Cost of wrong call:** Wasting hours of writing time on a page that cannot realistically capture traffic.
**Why ML helps:** Ranking is a complex interplay of signals. Simple rules fail when signals are tangled, whereas tree-based ML models can untangle interactions (e.g. word count vs competition) to highlight genuine opportunities.

## 2. Data safety

We used `data/raw/content_refresh_anonymized.csv`.
**Exclusions:** We excluded rows with `avg_position = 0` (no ranking data) to ensure our target variable was clean. We explicitly excluded derived fields like `is_declining_label`, `trend_direction`, and `trend_pct` to prevent label leakage.
`client_id` and `content_id` were used exclusively for group splitting and row tracking, never as features.
No private client names, URLs, or raw queries are present in the data or this repo.

## 3. Baseline

The baseline was a hand-written rule predicting priority scores based purely on `search_volume` and discrete threshold rules (e.g. "if avg_position < 10 and CTR < 1%, it's a priority"). For the modeling evaluation, our naive baseline was a Dummy Regressor predicting the mean `avg_position`. It is a fair comparison because it establishes the floor (predicting the mean) on the exact same grouped split. 

## 4. Model / analysis

**Method:** Random Forest Regressor. It fits the Signal Analysis lane because it captures non-linear signal interactions without requiring heavy feature engineering, and provides straightforward Permutation Importance to identify signal strength.
**Feature list:** `search_volume`, `competition`, `word_count`, `content_age_days`.
**Left out on purpose:** `trend_pct` and `impressions` (risk of overlapping window leakage with target).
**Target:** `avg_position`.

## 5. Evaluation

**Split:** Grouped by `client_id` (GroupShuffleSplit, 80/20). A random split allowed the model to memorize client-specific ranking quirks, yielding an artificially lower Mean Absolute Error. The grouped split forces the model to generalize to clients it has never seen, providing an honest evaluation.
**Metrics:** Mean Absolute Error (MAE).
The Random Forest outperformed the Dummy baseline slightly, but the high absolute error highlights the sheer variance and complexity of search ranking. The errors were largest in the "poor ranking" tail (positions 50+), where data is noisy.

## 6. Interpretation

**Feature Importances:** Permutation importance revealed that `search_volume` and `word_count` were the strongest drivers of prediction.
**Surprises/Negative Results:** The overall R^2 of the model was low. This is a negative result but an honest one: a handful of basic metadata signals cannot perfectly predict a page's average position. However, the directional relationships (longer content ranking slightly better on average) were clearly observed and statistically stable across clients.

## 7. Recommendation

We provide a ranked **Action Playbook** (`work/outputs/action_playbook_queue.csv`).
1. **Striking Distance Content**: Pages ranking 11-20 with high search volume are prioritized for content expansion and internal linking.
2. **Top 10 Low CTR**: Pages ranking top 10 with <1% CTR are prioritized for Meta tag rewrites.
**Confidence and Limits:** We are confident in the directional prioritization, but we cannot claim causal impact (i.e. adding words won't magically boost rank). This is decision-support for a human reviewer.

## 8. Reproducibility

**Commands:**
1. Setup environment and install dependencies (`pip install duckdb pandas scikit-learn matplotlib`).
2. Run notebooks sequentially: `jupyter nbconvert --execute --to notebook --inplace work/notebooks/w*.ipynb` and `capstone.ipynb`.
**Random Seeds:** `random_state=42` used consistently across splits and models.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset (https://flyrank.ai).
