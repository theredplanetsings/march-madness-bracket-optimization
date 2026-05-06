"""Generate figures for the paper."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = f"{ROOT}/data/processed"
FIG = f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def fig_backtest():
    df = pd.read_csv(f"{PROC}/backtest_results.csv")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w/2, df.actual_seed_score, w, label="Chalk-by-seed", color="#888")
    ax.bar(x + w/2, df.actual_model_score, w, label="Chalk-by-model (KenPom + LR)", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(df.season.astype(int), rotation=0)
    ax.set_ylabel("ESPN bracket score (10/20/40/80/160/320)")
    ax.set_title(f"Backtest: per-season actual ESPN score, n={len(df)} tournaments  "
                 f"(model wins {int((df.actual_model_score > df.actual_seed_score).sum())}/{len(df)})")
    ax.legend(loc="upper left", frameon=False)
    ax.axhline(df.actual_seed_score.mean(), color="#888", lw=0.7, ls="--", alpha=0.6)
    ax.axhline(df.actual_model_score.mean(), color="#1f77b4", lw=0.7, ls="--", alpha=0.6)
    ax.text(len(df) - 0.5, df.actual_seed_score.mean() + 30,
            f"mean {df.actual_seed_score.mean():.0f}", color="#555", ha="right", fontsize=8)
    ax.text(len(df) - 0.5, df.actual_model_score.mean() + 30,
            f"mean {df.actual_model_score.mean():.0f}", color="#1f77b4", ha="right", fontsize=8)
    plt.tight_layout()
    p = f"{FIG}/backtest.png"
    plt.savefig(p); plt.close()
    print(f"  saved {p}")


def fig_reliability():
    df = pd.read_csv(f"{PROC}/reliability_full_LR.csv")
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color="#bbb", lw=1, ls="--", label="Perfect calibration")
    ax.scatter(df.pred_mean, df.obs_mean, s=df["n"] / 4, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Predicted P(team A wins)")
    ax.set_ylabel("Empirical frequency (LOSO)")
    ax.set_title(f"Model calibration (n={int(df['n'].sum())} matchups)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    plt.tight_layout()
    p = f"{FIG}/reliability.png"
    plt.savefig(p); plt.close()
    print(f"  saved {p}")


def fig_2026_reach_heatmap():
    df = pd.read_csv(f"{PROC}/2026_reach_probs_model.csv", index_col=0)
    df = df.sort_values("p_Champ", ascending=False).head(16)
    cols = ["p_R32", "p_S16", "p_E8", "p_F4", "p_NCG", "p_Champ"]
    labels = ["R32", "S16", "E8", "F4", "NCG", "Champ"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(df[cols].values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df.index)
    for i in range(len(df)):
        for j, c in enumerate(cols):
            v = df[c].iloc[i]
            tc = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=tc, fontsize=8.5)
    ax.set_title("2026 model: P(team reaches round R) — top 16 by champion prob")
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    p = f"{FIG}/reach_2026.png"
    plt.savefig(p); plt.close()
    print(f"  saved {p}")


def fig_strategy_distribution():
    """Compare strategies on (a) actual 2026 score and (b) MC distribution."""
    actual = pd.read_csv(f"{PROC}/2026_actual_scores.csv")
    dist = pd.read_csv(f"{PROC}/2026_strategy_distribution.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: actual 2026 score
    ax = axes[0]
    colors = {"chalk_seed": "#888", "chalk_model": "#1f77b4", "ev_max": "#2ca02c"}
    bars = ax.bar(actual.strategy, actual.actual_score,
                  color=[colors.get(s, "#444") for s in actual.strategy])
    for b, v in zip(bars, actual.actual_score):
        ax.text(b.get_x() + b.get_width() / 2, v + 15, f"{int(v)}", ha="center", fontsize=9)
    ax.set_ylabel("ESPN points")
    ax.set_title("Actual 2026 ESPN score by strategy")
    ax.set_ylim(0, max(actual.actual_score) * 1.18)

    # Right: MC distribution mean ± std
    ax = axes[1]
    x = np.arange(len(dist))
    ax.errorbar(x, dist.mean_score, yerr=dist.std_score, fmt="o",
                color="#1f77b4", capsize=4, lw=1.5, markersize=8, label="Mean ± SD")
    for i, m in enumerate(dist.mean_score):
        ax.text(i, m + dist.std_score.iloc[i] + 30,
                f"p10–p90\n{int(dist.p10_score.iloc[i])}–{int(dist.p90_score.iloc[i])}",
                ha="center", fontsize=8, color="#444")
    ax.set_xticks(x); ax.set_xticklabels(dist.strategy)
    ax.set_ylabel("Score (over 10,000 simulated outcomes)")
    ax.set_title("Strategy score distribution (Monte Carlo, n=10,000)")
    ax.set_ylim(700, 2050)
    plt.tight_layout()
    p = f"{FIG}/strategy_2026.png"
    plt.savefig(p); plt.close()
    print(f"  saved {p}")


def fig_public_pick_table():
    df = pd.read_csv(f"{PROC}/public_pick_table.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(df.values * 100, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns)
    ax.set_yticks(range(len(df))); ax.set_yticklabels([str(s) for s in df.index])
    ax.set_xlabel("Round")
    ax.set_ylabel("Seed")
    for i in range(len(df)):
        for j in range(len(df.columns)):
            v = df.values[i, j] * 100
            tc = "white" if v > 60 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color=tc, fontsize=7.5)
    ax.set_title("Public pick rate by seed × round (% of brackets, ESPN 2019)")
    plt.colorbar(im, ax=ax, fraction=0.04, label="% of public brackets")
    plt.tight_layout()
    p = f"{FIG}/public_picks.png"
    plt.savefig(p); plt.close()
    print(f"  saved {p}")


def fig_round_importance():
    """Per-round permutation importance heatmap for the Full LR model.

    Answers the proposal's "does feature X matter more in S16 than R64?"
    question visually. Uses the full_LR rows from
    `data/processed/round_importance_perm.csv`.
    """
    df = pd.read_csv(f"{PROC}/round_importance_perm.csv")
    df = df[df.Model == "full_LR"]
    if len(df) == 0:
        print("  [skip] no full_LR rows in round_importance_perm.csv")
        return
    rounds = ["R64", "R32", "S16", "E8", "F4", "NCG"]
    pivot = df.pivot(index="Feature", columns="Round", values="MeanDeltaLogLoss")
    pivot = pivot.reindex(columns=[r for r in rounds if r in pivot.columns])
    pivot = pivot.sort_values("R64", ascending=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    vmax = max(0.05, float(pivot.values.max()))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            tc = "white" if v > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=tc, fontsize=8.5)
    ax.set_title("Per-round permutation importance (Full LR)\nΔ log-loss when feature is shuffled within round")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Δ log-loss")
    plt.tight_layout()
    p = f"{FIG}/round_importance.png"
    plt.savefig(p); plt.close()
    print(f"  saved {p}")


def fig_momentum_2026():
    """Bar chart of 2026 tournament teams ranked by their late-season
    momentum *relative to the field median*. Centering on the 2026
    tournament median (rather than zero) makes the bars honestly readable
    as "hotter than the typical tournament team" vs "cooler" — without
    the systematic positive bias that comes from tournament teams
    peaking late by selection nature.
    """
    mom = pd.read_csv(f"{PROC}/momentum_by_season.csv")
    sub = mom[mom.Season == 2026].copy()
    if len(sub) == 0:
        print("  [skip] no 2026 rows in momentum_by_season.csv")
        return
    field_median = float(sub["MomentumDelta"].median())
    sub["MomentumVsField"] = sub["MomentumDelta"] - field_median
    sub = sub.sort_values("MomentumVsField", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 9.5))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in sub["MomentumVsField"]]
    ax.barh(sub["Team"], sub["MomentumVsField"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Late-season form vs 2026 field median ({field_median:+.1f} pts/100)")
    ax.set_title("2026 tournament: late-season form relative to the field\n"
                 "decay-weighted eff. AdjEM − season AdjEM, then centered on field median")
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    p = f"{FIG}/momentum_2026.png"
    plt.savefig(p); plt.close()
    print(f"  saved {p}")


def main():
    fig_backtest()
    fig_reliability()
    fig_2026_reach_heatmap()
    fig_strategy_distribution()
    fig_public_pick_table()
    fig_round_importance()
    fig_momentum_2026()


if __name__ == "__main__":
    main()
