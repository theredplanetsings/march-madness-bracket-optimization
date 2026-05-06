## Predictive Weight Optimisation: A Machine Learning Approach to Maximising Expected Value in Men’s March Madness Brackets

Team members: Bryce Clement, Christian Rutherford, Devon Diaco

### Overview
This project models the 2026 NCAA Division I Men’s Basketball tournament (March Madness) as a 63‑game probability tree and builds a bracket optimiser aimed at maximising expected value under the ESPN Tournament Challenge scoring system, where later rounds are weighted more heavily.

### Objectives
- Predict game outcomes while optimising feature weights for bracket EV rather than single-game accuracy.
- Identify leverage opportunities where model probabilities exceed public sentiment.
- Simulate brackets to target high‑percentile finishes in large pools.

### Modelling approach
- Feature engineering: historical tournament data (2010–2026, excluding 2020), normalised efficiency metrics, strength of schedule, and a momentum feature with late‑season decay weighting.
- Baselines: logistic regression and random forest for calibrated win probabilities and non‑linear feature importance.
- Gradient boosting: XGBoost, LightGBM, and CatBoost to refine weights and minimise log loss.
- Expected value optimisation: incorporate public pick data to prioritise high‑EV upsets.
- Monte Carlo simulation: thousands of tournament runs to select brackets with the best 99th‑percentile likelihood.

### Results at a glance
- Best model: full-feature logistic regression with 83.0% accuracy and 0.410 log-loss (LOSO CV).
- Backtest (2010–2026): model bracket beats seed-chalk in 13/15 seasons with +428 mean ESPN points (52% relative gain).
- 2026 outcomes: model bracket scored 1,570 vs 1,000 for seed-chalk and selected Michigan as champion with 67.2% probability.

### Quickstart
Using the processed data in this repo:

```bash
python src/run_2026.py
```

Rebuild features and retrain models from raw data:

```bash
python src/build_features.py
python src/train_models.py
```

Optional (if you have KenPom credentials and want to refresh raw inputs):

```bash
python src/scrape_kenpom.py
python src/scrape_fourfactors.py
```

### Credentials
KenPom scraping requires credentials. Provide them via `KENPOM_EMAIL`/`KENPOM_PASSWORD` or produce a [credentials.json](credentials.json) (gitignored) with:

```json
{
  "kenpom_email": "you@example.com",
  "kenpom_password": "your_password"
}

```

### Repository layout
- [credentials.json](credentials.json) KenPom login (gitignored)
- [data/](data/) raw inputs and processed datasets
- [src/](src/) data prep, feature engineering, modelling, optimisation, and simulation scripts
- [models/](models/) trained model artefacts and tuning summaries
- [notebooks/](notebooks/) analysis notebooks and utilities
- [figures/](figures/) exported visuals
- [paper/](paper/) write-up materials

### Key outputs
- Processed datasets in [data/processed/](data/processed/)
- Trained models in [models/](models/)
- Backtest and optimisation results in [data/processed/](data/processed/)

### References
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://en.wikipedia.org/wiki/Gradient_boosting
- https://xgboost.readthedocs.io/en/release_3.2.0/
- https://lightgbm.readthedocs.io/en/stable/
- https://catboost.ai/
- https://kenpom.com/
- https://www.ncaa.com/stats/basketball-men/d1
- https://arxiv.org/abs/1412.0248

Changes to methods will be made as needed if issues arise.