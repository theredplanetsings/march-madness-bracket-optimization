"""Build notebooks/full_pipeline.ipynb programmatically.

Running this script overwrites the notebook with a clean, narrative-driven
walkthrough of the full project. After writing we execute the notebook
in-place so cell outputs are committed alongside the source.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "full_pipeline.ipynb"


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip("\n"))


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    cells.append(md(r"""
# Predictive Weight Optimisation for Men's March Madness Brackets

**Authors:** Bryce Clement, Christian Rutherford, Devon Diaco
**Course:** MAT-210, Davidson College, Spring 2026

This notebook walks through the full project end-to-end. Every result and
figure below is loaded from the on-disk artefacts produced by the scripts
in `src/`, so the notebook is cheap to re-run and stays consistent with
whatever the latest pipeline produced.

If any artefact below is missing, run the corresponding command:

| Artefact | Producer |
|---|---|
| `data/raw/kenpom_*.csv` | `python src/scrape_kenpom.py` |
| `data/raw/schedules/*.csv` | `python src/scrape_schedules.py` |
| `data/raw/fourfactors_*.csv` | `python src/scrape_fourfactors.py` |
| `data/processed/momentum_by_season.csv` | `python src/momentum.py` |
| `data/processed/matchups.csv` | `python src/build_features.py` |
| `models/*.joblib`, `models/cv_summary.json` | `python src/train_models.py` |
| `data/processed/round_importance_*.csv` | `python src/feature_importance.py` |
| `data/processed/backtest_results.csv` | `python src/backtest.py` |
| `data/processed/2026_*.csv` | `python src/run_2026.py` |
| `figures/*.png` | `python src/make_figures.py` |
"""))

    cells.append(code(r"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

ROOT = Path('..').resolve()
RAW = ROOT / 'data' / 'raw'
PROC = ROOT / 'data' / 'processed'
MODELS = ROOT / 'models'
FIG = ROOT / 'figures'

pd.options.display.float_format = '{:,.3f}'.format
plt.rcParams.update({
    'figure.dpi': 110, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})
"""))

    cells.append(md(r"""
## 1. Data layer

We pull three primary sources from KenPom (paid login) and one from
ESPN (Wayback Machine):

* **Season-end ratings** for every team-season 2010–2026 (offensive /
  defensive efficiency, tempo, luck, strength of schedule, seed).
* **Per-game schedules** for every tournament team-season — used to
  build the decay-weighted late-season momentum signal.
* **Four Factors** (eFG%, TO%, OR%, FTRate, both sides) — the proposal
  named *Effective Field Goal Percentage* explicitly so we wire eFG%
  into the feature set.
* **ESPN "who picked whom"** public-pick percentages, 2019 only, used
  as a synthetic seed×round prior for any other year (the only year
  that was reachable via the Wayback Machine).
"""))

    cells.append(code(r"""
import os
season_years = sorted({int(p.stem.split('_')[1]) for p in RAW.glob('kenpom_*.csv')})
n_schedules = len(list((RAW / 'schedules').glob('*.csv')))
n_fourfactors = len(list(RAW.glob('fourfactors_*.csv')))
espn = pd.read_csv(RAW / 'espn_picks_all.csv')

print(f'KenPom season-end files: {len(season_years)} seasons '
      f'({min(season_years)}-{max(season_years)})')
print(f'KenPom per-game schedules:   {n_schedules} team-seasons')
print(f'KenPom Four Factors files:   {n_fourfactors} seasons')
print(f'ESPN public picks:           {len(espn)} rows, year(s) {sorted(espn.Year.unique())}')
"""))

    cells.append(md(r"""
### 1.1 Matchup feature matrix

`build_features.py` joins all of the above into a single matchup matrix.
Each tournament game appears as **two rows** (mirrored Team A / Team B
framing) so the dataset is balanced. Each row is a 11-vector of *Team A
minus Team B* feature deltas: 10 KenPom statistics plus seed differential.
"""))

    cells.append(code(r"""
matchups = pd.read_csv(PROC / 'matchups.csv')
print(f'Matchup rows (with mirrors): {len(matchups)}')
print(f'Label balance: {matchups.label.mean():.3f}')
print(f'Seasons covered: {matchups.Season.min()}-{matchups.Season.max()}')
print(f'Round breakdown:')
print(matchups.Round.value_counts().to_frame('rows'))
"""))

    cells.append(code(r"""
delta_cols = [c for c in matchups.columns if c.startswith('d_')]
print('Feature deltas (model input):')
print(delta_cols)
matchups[['Season', 'Round', 'TeamA', 'TeamB'] + delta_cols].head(4)
"""))

    cells.append(md(r"""
## 2. Decay-weighted momentum feature

The proposal asked for "a decay weight to late-season team performance
& conference tournament results." We build this by walking each
team-season's per-game KenPom schedule and computing for each game

$$
\text{eff\_em}_g = \frac{\text{PF}_g - \text{PA}_g}{\text{poss}_g}\cdot 100
                   + \text{AdjEM}_{\text{opp}(g)}
                   + \text{loc\_adj}_g
$$

(i.e. "what AdjEM did this team play at in this game, adjusted for the
opponent and home-court advantage"). We exclude NCAA tournament games
to avoid label leakage, then take a weighted average where the weights
are an exponential decay with `tau = 30 days` anchored on Selection
Sunday, multiplied by `1.5×` for conference-tournament games. Subtracting
the team's season-long AdjEM gives `MomentumDelta` — a trajectory
signal orthogonal to the season-end rating already in `d_AdjEM`.
"""))

    cells.append(code(r"""
mom = pd.read_csv(PROC / 'momentum_by_season.csv')
print(f'Total team-seasons with decay momentum: {len(mom)}')
print(f"Mean MomentumDelta (across all 2010-2026 tournament teams): "
      f"{mom['MomentumDelta'].mean():+.2f} pts/100 possessions")
print()
print('2026 tournament teams - top 8 by trajectory:')
display(mom[mom.Season == 2026]
        .sort_values('MomentumDelta', ascending=False)
        .head(8)[['Team', 'SeasonAdjEM', 'MomentumDecay', 'MomentumDelta', 'NGames']])
print('2026 tournament teams - bottom 5 by trajectory:')
display(mom[mom.Season == 2026]
        .sort_values('MomentumDelta')
        .head(5)[['Team', 'SeasonAdjEM', 'MomentumDecay', 'MomentumDelta', 'NGames']])
"""))

    cells.append(code(r"""
display(Image(filename=str(FIG / 'momentum_2026.png')))
"""))

    cells.append(md(r"""
## 3. Win-probability model

We fit six models in a leave-one-season-out (LOSO) cross-validation
loop — at each fold we hold out one tournament season, train on the
other 15, predict the held-out season. Metrics are aggregated across
folds so they directly mimic real-world usage (predict the next
tournament from past tournaments only).

**Models compared:**
1. Seed-only Logistic Regression (interpretable baseline)
2. **Full Logistic Regression** — all 11 deltas (primary model)
3. XGBoost
4. Random Forest
5. LightGBM
6. CatBoost
"""))

    cells.append(code(r"""
import json
cv = json.loads((MODELS / 'cv_summary.json').read_text())
cv_df = pd.DataFrame(cv).T
cv_df = cv_df.sort_values('log_loss_mean')
display(cv_df[['log_loss_mean', 'log_loss_std', 'brier_mean', 'acc_mean']])
print(f'Best model by log-loss: {cv_df.index[0]}')
print(f'(log-loss is the proper scoring rule: it rewards calibrated probabilities,')
print(f' which is what bracket simulation downstream needs.)')
"""))

    cells.append(md(r"""
### 3.1 Model calibration

A reliability diagram bins the LOSO out-of-fold predictions by
predicted probability and plots the empirical win rate inside each bin.
A perfectly-calibrated model lies on the diagonal — predicted P=0.7
should imply that the team actually wins 70% of the time.
"""))

    cells.append(code(r"""
display(Image(filename=str(FIG / 'reliability.png')))
"""))

    cells.append(md(r"""
## 4. Round-specific feature importance

The proposal asked: "does 3-point shooting matter more in the Sweet 16
than in the Round of 64?" — a question about whether feature importance
*shifts* by round. We answer this two ways:

1. **Per-round logistic regression** (`round_importance_lr.csv`):
   train one LR per round, report standardised coefficients. Larger
   `|coef|` means the feature has more pull conditional on being in
   that round.
2. **Per-round permutation importance** (`round_importance_perm.csv`):
   for each (model, round, feature) we shuffle that one feature among
   the round's rows, score with the trained model, and measure the
   increase in log-loss. Averaged over 30 random shuffles for stability.
"""))

    cells.append(code(r"""
lr_imp = pd.read_csv(PROC / 'round_importance_lr.csv')
pivot = lr_imp.pivot(index='Feature', columns='Round', values='Coef')
pivot = pivot[[c for c in ['R64', 'R32', 'S16', 'E8', 'F4', 'NCG'] if c in pivot.columns]]
print('Standardised LR coefficients by round (sign matters: +ve favors Team A):')
display(pivot.round(2))
"""))

    cells.append(code(r"""
display(Image(filename=str(FIG / 'round_importance.png')))
"""))

    cells.append(md(r"""
**Reading the heatmap.** A few patterns jump out:

* `d_Seed` is dominant in R64 / R32 (chalk almost always wins early
  rounds) but its importance falls to ~0 in the Championship — by the
  Final Four, every remaining team is good enough that seed differential
  has stopped predicting much.
* `d_AdjO` (offensive efficiency) becomes the single most important
  feature in the Final Four and Championship. Defensive metrics matter
  most in the Elite 8.
* `d_Momentum` (decay-weighted late-season trajectory) carries the
  most signal in S16 — when teams that arrived hot keep winning.
* `d_OffEFG` and `d_DefEFG` overlap heavily with the AdjO/AdjD summary
  metrics and add comparatively little marginal log-loss reduction —
  but they do appear at the top in Final Four and Elite 8.
"""))

    cells.append(md(r"""
## 5. Monte Carlo bracket simulation

Given a per-matchup win probability function and a 64-team bracket, we
draw N = 10,000 simulated tournament outcomes by walking the bracket
node by node. Recording how often each team reaches each round gives
us a 64×7 reach-probability table. ESPN scoring (10 / 20 / 40 / 80 /
160 / 320 per correct pick R64→NCG) decomposes by linearity of
expectation:

$$
\mathbb{E}[\text{score}] = \sum_{\text{round } r} \text{pts}(r)
\cdot \mathbb{P}(\text{picked team reaches round } r)
$$

so the EV-maximising bracket is the closed-form *greedy* choice at each
game: pick the team with the higher reach probability for the *next*
round. No exponential bracket search is needed.
"""))

    cells.append(md(r"""
## 6. Historical backtest, 2010–2026

For each season we train a model on the *other* fifteen seasons and
score two strategies on that season's actual outcome:

* **`chalk_seed`** — pick the lower seed at every game (no model).
* **`chalk_model`** — pick the higher model-probability team at every
  game.
"""))

    cells.append(code(r"""
back = pd.read_csv(PROC / 'backtest_results.csv')
display(back.assign(diff=back.actual_model_score - back.actual_seed_score))
print()
print(f"Mean chalk-seed score:   {back.actual_seed_score.mean():,.1f}")
print(f"Mean chalk-model score:  {back.actual_model_score.mean():,.1f}")
print(f"Mean expected (model):   {back.expected_model_score.mean():,.1f}")
n_wins = int((back.actual_model_score > back.actual_seed_score).sum())
print(f"Years model > seed (actual): {n_wins} / {len(back)}")
"""))

    cells.append(code(r"""
display(Image(filename=str(FIG / 'backtest.png')))
"""))

    cells.append(md(r"""
## 7. 2026 tournament — pre-tournament prediction & retrospective score

Before the 2026 tournament began, the model assigned the championship
probabilities below. The actual tournament concluded with **Michigan
defeating Connecticut 69–63**.
"""))

    cells.append(code(r"""
reach = pd.read_csv(PROC / '2026_reach_probs_model.csv', index_col=0)
top = reach.sort_values('p_Champ', ascending=False).head(10)
display(top[['p_R32', 'p_S16', 'p_E8', 'p_F4', 'p_NCG', 'p_Champ']].round(3))
"""))

    cells.append(code(r"""
display(Image(filename=str(FIG / 'reach_2026.png')))
"""))

    cells.append(code(r"""
actual = pd.read_csv(PROC / '2026_actual_scores.csv')
print('2026 strategy scores against the actual outcome:')
display(actual)
"""))

    cells.append(md(r"""
### 7.1 2026 strategy distribution

We re-simulate 10,000 outcome trajectories under the same trained
model and score each strategy on every simulated outcome. This shows
the *distribution* of plausible scores, not just the realised one.
"""))

    cells.append(code(r"""
dist = pd.read_csv(PROC / '2026_strategy_distribution.csv')
display(dist.round(1))
"""))

    cells.append(code(r"""
display(Image(filename=str(FIG / 'strategy_2026.png')))
"""))

    cells.append(md(r"""
## 8. Public sentiment & leverage strategy (proposal-required EV piece)

The proposal asked for a leverage analysis using ESPN public-pick
percentages — picking high-probability teams the public is *under*-rating
to gain pool-relative EV. Wayback yielded ESPN data for **only 2019**;
we fit a seed×round prior from that single year and use it as a
synthetic public-pick distribution for any other year.
"""))

    cells.append(code(r"""
display(Image(filename=str(FIG / 'public_picks.png')))
"""))

    cells.append(md(r"""
## 9. Proposal coverage

Every concrete commitment in the original project proposal:

| Proposal commitment | Status |
|---|---|
| Phenomenon: 2026 NCAA tournament | done — predicted prospectively, scored retrospectively |
| 10–15 yrs historical data | done — 17 seasons (2010–2026) |
| AdjEM | done |
| Strength of schedule | done — `d_SOS-AdjEM`, `d_NCSOS-AdjEM` |
| Effective Field Goal % | done — `d_OffEFG`, `d_DefEFG` from KenPom Four Factors |
| Decay-weighted late-season momentum + conf tournaments | done — `d_Momentum` |
| Logistic regression baseline | done — `seed_only_LR`, `full_LR` |
| Random Forest non-linear importance | done — `rf` |
| Round-specific feature importance | done — `feature_importance.py` |
| XGBoost | done — `xgb` |
| LightGBM | done — `lgbm` |
| CatBoost | done — `catboost` |
| Log-loss minimisation | done |
| EV optimisation | done — `ev_optimize.py` (random search prototype) |
| Monte Carlo, 99th-percentile pool finish | done — 10,000 sims, P(top 1%) reported |
| Multi-year ESPN public-pick data | partial — only 2019 reachable, used as seed-prior |

Only the multi-year public-pick data is missing; the rest of the
proposal is fully delivered.
"""))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def main():
    nb = build()
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(NB_PATH))
    print(f"wrote {NB_PATH}")

    print("executing notebook in-place ...", flush=True)
    nb_loaded = nbf.read(str(NB_PATH), as_version=4)
    client = NotebookClient(nb_loaded, timeout=120, kernel_name="python3",
                            resources={"metadata": {"path": str(NB_PATH.parent)}})
    client.execute()
    nbf.write(nb_loaded, str(NB_PATH))
    print(f"executed and saved {NB_PATH}")


if __name__ == "__main__":
    main()
