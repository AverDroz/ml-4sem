"""EDA visualizations for wine and CS:GO datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ─── Style ────────────────────────────────────────────────────────────────────

PALETTE = "muted"
FIG_DPI = 120

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)
plt.rcParams.update({"figure.dpi": FIG_DPI, "axes.spines.top": False,
                      "axes.spines.right": False})


def _save(fig: plt.Figure, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ─── Generic ──────────────────────────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame, title: str,
                       save_path: Path | None = None) -> None:
    """Histogram grid for all numeric columns."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    n = len(num_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=30, color=sns.color_palette(PALETTE)[0],
                     edgecolor="white", linewidth=0.5)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_path)


def plot_correlation_heatmap(df: pd.DataFrame, title: str,
                              save_path: Path | None = None) -> None:
    """Correlation heatmap."""
    corr = df.select_dtypes(include="number").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)


def plot_boxplots(df: pd.DataFrame, title: str,
                  save_path: Path | None = None) -> None:
    """Boxplot grid for all numeric columns."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    n = len(num_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor=sns.color_palette(PALETTE)[1], alpha=0.7),
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="o", markersize=2, alpha=0.4))
        axes[i].set_title(col, fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_path)


def plot_missing_values(df: pd.DataFrame, title: str,
                        save_path: Path | None = None) -> None:
    """Bar chart of missing value counts per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print(f"[{title}] No missing values.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(missing) * 0.9), 4))
    bars = ax.bar(missing.index, missing.values,
                  color=sns.color_palette(PALETTE)[3])
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, save_path)


# ─── Wine-specific ────────────────────────────────────────────────────────────

def plot_wine_target_distribution(df: pd.DataFrame,
                                   save_path: Path | None = None) -> None:
    counts = df["quality"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=sns.color_palette(PALETTE))
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Quality score")
    ax.set_ylabel("Count")
    ax.set_title("Wine — распределение целевой переменной (quality)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)


def plot_wine_feature_vs_quality(df: pd.DataFrame,
                                  save_path: Path | None = None) -> None:
    features = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, feat in zip(axes, features):
        df.boxplot(column=feat, by="quality", ax=ax,
                   patch_artist=False, grid=False)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("quality")
        ax.set_ylabel("")
        plt.sca(ax)
        plt.title(feat, fontsize=9)

    fig.suptitle("Wine — ключевые признаки vs quality",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)


def plot_wine_scatter_matrix(df: pd.DataFrame,
                              save_path: Path | None = None) -> None:
    cols = ["alcohol", "volatile acidity", "sulphates", "density", "quality"]
    fig = plt.figure(figsize=(10, 9))
    pd.plotting.scatter_matrix(df[cols], alpha=0.3, figsize=(10, 9),
                                diagonal="hist", color="steelblue")
    plt.suptitle("Wine — scatter matrix ключевых признаков",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_path)


# ─── CS:GO-specific ───────────────────────────────────────────────────────────

def plot_csgo_target_distribution(df: pd.DataFrame,
                                   save_path: Path | None = None) -> None:
    counts = df["bomb_planted"].value_counts()
    labels = ["Bomb NOT planted", "Bomb planted"]
    colors = sns.color_palette(PALETTE, 2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar
    axes[0].bar(labels, counts.values, color=colors)
    for bar, val in zip(axes[0].patches, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f"{val:,}", ha="center", fontsize=10)
    axes[0].set_title("Абсолютные значения")
    axes[0].set_ylabel("Count")

    # Pie
    axes[1].pie(counts.values, labels=labels, autopct="%1.1f%%",
                colors=colors, startangle=90)
    axes[1].set_title("Доля классов")

    fig.suptitle("CS:GO — распределение целевой переменной (bomb_planted)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)


def plot_csgo_map_distribution(df: pd.DataFrame,
                                save_path: Path | None = None) -> None:
    """Works on RAW (before encoding) df."""
    if df["map"].dtype == object or str(df["map"].dtype) == "string":
        map_counts = df["map"].value_counts()
    else:
        # Already encoded — skip
        print("map already encoded, skipping map distribution plot")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(map_counts.index, map_counts.values,
                   color=sns.color_palette(PALETTE, len(map_counts)))
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("CS:GO — количество раундов по картам",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Count")
    fig.tight_layout()
    _save(fig, save_path)


def plot_csgo_health_money(df_clean: pd.DataFrame,
                            save_path: Path | None = None) -> None:
    """CT vs T health and money comparison by bomb status."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    pairs = [
        ("ct_health", "t_health", "Здоровье CT vs T"),
        ("ct_money", "t_money", "Деньги CT vs T"),
        ("ct_players_alive", "t_players_alive", "Живые игроки CT vs T"),
        ("ct_armor", "t_armor", "Броня CT vs T"),
    ]

    for ax, (col_ct, col_t, title) in zip(axes.flatten(), pairs):
        data_0 = df_clean[df_clean["bomb_planted"] == 0]
        data_1 = df_clean[df_clean["bomb_planted"] == 1]

        ax.scatter(data_0[col_ct].sample(min(500, len(data_0)), random_state=42),
                   data_0[col_t].sample(min(500, len(data_0)), random_state=42),
                   alpha=0.3, s=10, label="Not planted", color=sns.color_palette(PALETTE)[0])
        ax.scatter(data_1[col_ct].sample(min(500, len(data_1)), random_state=42),
                   data_1[col_t].sample(min(500, len(data_1)), random_state=42),
                   alpha=0.4, s=10, label="Planted", color=sns.color_palette(PALETTE)[3])

        ax.set_xlabel(col_ct)
        ax.set_ylabel(col_t)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle("CS:GO — зависимости признаков от bomb_planted",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
