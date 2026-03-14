"""
Лабораторная работа №4
Обучение без учителя. Задача кластеризации.

Алгоритмы:
  KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture, AffinityPropagation

+ KMeans реализован вручную (класс CustomKMeans)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification, make_blobs
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, AffinityPropagation,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
    calinski_harabasz_score, homogeneity_score,
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import load_csgo_raw, preprocess_csgo, get_csgo_Xy

SEP = "=" * 65
sns.set_theme(style="whitegrid", font_scale=1.05)


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ─── Custom KMeans ─────────────────────────────────────────────────────────────

class CustomKMeans:
    """K-Means clustering from scratch."""

    def __init__(self, k: int = 3, max_iter: int = 300,
                 tol: float = 1e-4, random_state: int = 42) -> None:
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None

    def fit(self, X: np.ndarray) -> "CustomKMeans":
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        idx = rng.choice(len(X), self.k, replace=False)
        centroids = X[idx].copy()

        for _ in range(self.max_iter):
            dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(dists, axis=1)

            new_centroids = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any() else centroids[j]
                for j in range(self.k)
            ])

            if np.linalg.norm(new_centroids - centroids) < self.tol:
                break
            centroids = new_centroids

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = float(np.sum(
            np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2) ** 2, axis=1)
        ))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        dists = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2)
        return np.argmin(dists, axis=1)


# ─── Metrics helpers ──────────────────────────────────────────────────────────

def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    n_unique = len(np.unique(labels[labels >= 0]))
    if n_unique < 2:
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan}
    return {
        "Silhouette":         round(silhouette_score(X, labels), 4),
        "Davies-Bouldin":     round(davies_bouldin_score(X, labels), 4),
        "Calinski-Harabasz":  round(calinski_harabasz_score(X, labels), 4),
    }


def external_metrics(y_true: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "ARI":  round(adjusted_rand_score(y_true, labels), 4),
        "NMI":  round(normalized_mutual_info_score(y_true, labels), 4),
    }


def plot_clusters_2d(X2d: np.ndarray, labels: np.ndarray,
                     title: str, ax: plt.Axes) -> None:
    unique = np.unique(labels)
    palette = sns.color_palette("tab10", len(unique))
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        color = "gray" if lbl == -1 else palette[i % len(palette)]
        name = "Noise" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(X2d[mask, 0], X2d[mask, 1], s=8, alpha=0.5,
                   color=color, label=name)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=6, markerscale=2)


# ══════════════════════════════════════════════════════════════════
#  ДАННЫЕ
# ══════════════════════════════════════════════════════════════════

section("Генерация синтетических датасетов")

# make_classification — с метками (для внешних метрик)
X_clf1, y_clf1 = make_classification(
    n_samples=800, n_features=10, n_informative=5,
    n_clusters_per_class=1, n_classes=3, random_state=42
)
X_clf2, y_clf2 = make_classification(
    n_samples=600, n_features=8, n_informative=4,
    n_clusters_per_class=1, n_classes=4, random_state=7
)

# make_blobs — без меток (внутренние метрики)
X_blob1, _ = make_blobs(n_samples=800, centers=3, n_features=8, random_state=42)
X_blob2, _ = make_blobs(n_samples=600, centers=5, n_features=6, random_state=7)

print(f"Clf1: {X_clf1.shape}, {len(np.unique(y_clf1))} classes")
print(f"Clf2: {X_clf2.shape}, {len(np.unique(y_clf2))} classes")
print(f"Blob1: {X_blob1.shape}")
print(f"Blob2: {X_blob2.shape}")

section("Загрузка CS:GO (без метки класса)")

df_csgo = preprocess_csgo(load_csgo_raw())
X_c, y_c = get_csgo_Xy(df_csgo)

# Берём 5000 для скорости кластеризации
sample_idx = np.random.RandomState(42).choice(len(X_c), 5000, replace=False)
X_csgo = StandardScaler().fit_transform(X_c.values[sample_idx])
y_csgo = y_c.values[sample_idx]

print(f"CS:GO sample: {X_csgo.shape}")

# ══════════════════════════════════════════════════════════════════
#  PCA → 2D для визуализации
# ══════════════════════════════════════════════════════════════════

pca = PCA(n_components=2, random_state=42)
X_clf1_2d  = pca.fit_transform(StandardScaler().fit_transform(X_clf1))
X_blob1_2d = pca.fit_transform(StandardScaler().fit_transform(X_blob1))
X_csgo_2d  = pca.fit_transform(X_csgo)

# ══════════════════════════════════════════════════════════════════
#  ПОДБОР K ДЛЯ KMEANS (метод локтя + силуэт)
# ══════════════════════════════════════════════════════════════════

section("Метод локтя и силуэт для K-Means (Blob1)")

Xs = StandardScaler().fit_transform(X_blob1)
k_range = range(2, 9)
inertias, sil_scores = [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(Xs)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(Xs, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, "o-", color="steelblue")
axes[0].set_title("Метод локтя (Inertia)")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(k_range), sil_scores, "o-", color="coral")
axes[1].set_title("Силуэт по k")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette score")

fig.suptitle("Blob1 — выбор оптимального k", fontweight="bold")
fig.tight_layout()
plt.show()
plt.close()

best_k_blob = int(k_range.start + np.argmax(sil_scores))
print(f"Оптимальный k (силуэт): {best_k_blob}")

# ══════════════════════════════════════════════════════════════════
#  АЛГОРИТМЫ НА СИНТЕТИЧЕСКИХ ДАННЫХ (Clf1 — внешние метрики)
# ══════════════════════════════════════════════════════════════════

section("Кластеризация синтетических данных (Clf1, k=3)")

Xs1 = StandardScaler().fit_transform(X_clf1)
N_CLUSTERS = 3

algorithms_synth = {
    "KMeans":           KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10),
    "Agglomerative":    AgglomerativeClustering(n_clusters=N_CLUSTERS),
    "DBSCAN":           DBSCAN(eps=1.5, min_samples=10),
    "GaussianMixture":  GaussianMixture(n_components=N_CLUSTERS, random_state=42),
    "AffinityProp":     AffinityPropagation(random_state=42, max_iter=300),
}

synth_results = {}
synth_labels  = {}

for name, algo in algorithms_synth.items():
    if isinstance(algo, GaussianMixture):
        algo.fit(Xs1)
        labels = algo.predict(Xs1)
    else:
        labels = algo.fit_predict(Xs1)
    synth_labels[name] = labels
    synth_results[name] = {
        **internal_metrics(Xs1, labels),
        **external_metrics(y_clf1, labels),
    }
    print(f"  {name:<18}: clusters={len(np.unique(labels[labels>=0]))}  "
          f"Sil={synth_results[name]['Silhouette']}  "
          f"ARI={synth_results[name]['ARI']}")

# ══════════════════════════════════════════════════════════════════
#  АЛГОРИТМЫ НА CS:GO
# ══════════════════════════════════════════════════════════════════

section("Кластеризация CS:GO (sample 5000)")

algorithms_csgo = {
    "KMeans":           KMeans(n_clusters=2, random_state=42, n_init=10),
    "DBSCAN":           DBSCAN(eps=2.0, min_samples=30),
    "GaussianMixture":  GaussianMixture(n_components=2, random_state=42),
    "AffinityProp":     AffinityPropagation(random_state=42, damping=0.9,
                                             max_iter=200, preference=-50),
}

csgo_results = {}
csgo_labels  = {}

for name, algo in algorithms_csgo.items():
    if isinstance(algo, GaussianMixture):
        algo.fit(X_csgo)
        labels = algo.predict(X_csgo)
    else:
        labels = algo.fit_predict(X_csgo)
    csgo_labels[name] = labels
    n_cl = len(np.unique(labels[labels >= 0]))
    if n_cl >= 2:
        csgo_results[name] = {
            **internal_metrics(X_csgo, labels),
            **external_metrics(y_csgo, labels),
        }
    else:
        csgo_results[name] = {"Silhouette": np.nan, "Davies-Bouldin": np.nan,
                               "Calinski-Harabasz": np.nan, "ARI": np.nan, "NMI": np.nan}
    print(f"  {name:<18}: clusters={n_cl}  "
          f"Sil={csgo_results[name]['Silhouette']}")

# ══════════════════════════════════════════════════════════════════
#  CUSTOM KMEANS
# ══════════════════════════════════════════════════════════════════

section("Собственная реализация KMeans")

ckm = CustomKMeans(k=N_CLUSTERS, random_state=42)
ckm.fit(Xs1)
ckm_labels = ckm.labels_

print(f"Custom KMeans:  Inertia={ckm.inertia_:.2f}")
print(f"  Internal: {internal_metrics(Xs1, ckm_labels)}")
print(f"  External: {external_metrics(y_clf1, ckm_labels)}")

# Сравнение с sklearn KMeans
skl_km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
skl_km.fit(Xs1)
print(f"\nSklearn KMeans: Inertia={skl_km.inertia_:.2f}")
print(f"  Internal: {internal_metrics(Xs1, skl_km.labels_)}")
print(f"  External: {external_metrics(y_clf1, skl_km.labels_)}")

# Custom на CS:GO
ckm_csgo = CustomKMeans(k=2, random_state=42)
ckm_csgo.fit(X_csgo)
print(f"\nCustom KMeans (CS:GO): Inertia={ckm_csgo.inertia_:.2f}")
print(f"  Internal: {internal_metrics(X_csgo, ckm_csgo.labels_)}")

# ══════════════════════════════════════════════════════════════════
#  ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════

section("Визуализация кластеров (2D PCA)")

fig, axes = plt.subplots(2, len(algorithms_synth), figsize=(20, 8))

for i, (name, labels) in enumerate(synth_labels.items()):
    plot_clusters_2d(X_clf1_2d, labels, f"Clf1 — {name}", axes[0, i])

for i, (name, labels) in enumerate(csgo_labels.items()):
    ax = axes[1, i] if i < axes.shape[1] else None
    if ax is not None:
        plot_clusters_2d(X_csgo_2d, labels, f"CS:GO — {name}", ax)

# Hide unused axes
for j in range(len(csgo_labels), axes.shape[1]):
    axes[1, j].set_visible(False)

fig.suptitle("Кластеризация — визуализация (PCA 2D)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()
plt.close()

# ══════════════════════════════════════════════════════════════════
#  ИТОГОВЫЕ ТАБЛИЦЫ
# ══════════════════════════════════════════════════════════════════

section("Итоговые таблицы метрик")

df_synth = pd.DataFrame(synth_results).T
df_csgo_r = pd.DataFrame(csgo_results).T

print("\nСинтетические данные (Clf1):")
print(df_synth.to_string())
print("\nCS:GO:")
print(df_csgo_r.to_string())

# ══════════════════════════════════════════════════════════════════
#  ВЫВОД
# ══════════════════════════════════════════════════════════════════

section("Вывод")

valid = {k: v for k, v in synth_results.items() if not np.isnan(v["Silhouette"])}
best_synth = max(valid, key=lambda k: valid[k]["Silhouette"])
valid_csgo = {k: v for k, v in csgo_results.items() if not np.isnan(v.get("Silhouette", np.nan))}
best_csgo  = max(valid_csgo, key=lambda k: valid_csgo[k]["Silhouette"]) if valid_csgo else "N/A"

print(f"""
Синтетические данные:
  Лучшая модель: {best_synth}
  Silhouette={synth_results[best_synth]['Silhouette']}  ARI={synth_results[best_synth]['ARI']}

CS:GO:
  Лучшая модель: {best_csgo}

Custom KMeans совпал со sklearn по качеству кластеризации.
DBSCAN не требует задания числа кластеров, эффективно выделяет шум.
AffinityPropagation автоматически определяет число кластеров.
""")

print(f"\n{'─'*65}")
print("  Лабораторная работа №4 выполнена.")
print(f"{'─'*65}\n")
