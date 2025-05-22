# Satellite-Informed Heating Oil Trading Strategy

This project, developed for the Oregon Quant Group, explores the predictive power of satellite temperature data in forecasting heating oil prices. We use Z-score anomaly detection, seasonal demeaning, and machine learning (ElasticNet and LightGBM) to construct long/short strategies and validate them with robust cross-validation and walk-forward testing.

---

## Project Objective

- Use remote-sensing temperature data (MODIS LST) to anticipate demand-driven price movements in heating oil.
- Construct a **systematic trading strategy** based on anomalies in lagged land surface temperature.
- Validate strategy performance using **PurgedKFold**, **walk-forward backtests**, and **buy-and-hold benchmarks**.

---

## Project Structure

| File | Purpose |
|------|---------|
| `RunFeatures.ipynb` | Parses and engineers lag-based and rolling temperature features from MODIS data. |
| `Train_Structured.ipynb` | Trains LightGBM and ElasticNet models using PurgedKFold and conducts return-based strategy evaluations. |
| `OQG_Spring_Project.pdf` | Final presentation summarizing methodology, key results, drawdowns, and model performance. |

---

## Key Techniques

- **MODIS Satellite Features**: Extract `lst_day_1km_mean_lagX`, rolling features, and anomalies over day-of-year.
- **Z-score Anomaly Signal**: Signal is triggered when temperature deviates sharply from seasonal baseline.
- **LightGBM Model**: Gradient-boosted trees with randomized bagging, noise injection, and column dropout.
- **ElasticNet Model**: Linear regression with both L1 and L2 penalties; selected via randomized grid search.
- **Cross-Validation**: 
  - `PurgedKFold` prevents leakage across time-split folds.
  - `RepeatedPurgedKFold` used in `RandomizedSearchCV` to tune hyperparameters.
- **Walk-Forward Testing**: Retrain models in rolling fashion on expanding time windows to simulate real-time deployment.

---

## Results Summary

- **Best Sharpe Ratio**: ~0.91 using Z-score anomaly strategy.
- **LightGBM Total Return**: Significantly outperformed Buy & Hold in targeted windows.
- **Drawdowns**: Managed through volatility-weighted positions and optional stop-loss.
- **Blending**: Combined ElasticNet and LGB predictions improved signal stability.

---

## Next Steps

- Add **macroeconomic covariates** (e.g., refinery throughput, EIA data).
- Incorporate **spatial features** (tile-based temperature deltas).
- Deploy a **real-time pipeline** for ongoing satellite data ingestion and trade signal generation.
- Extend to **other commodities** sensitive to weather anomalies (e.g., natural gas, corn futures).

---

## 📜 Citation

Project by Chanisda von der Luehe (Spring 2025, Oregon Quant Group)
