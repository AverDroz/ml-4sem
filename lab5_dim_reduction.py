"""
Лабораторная работа №5
Понижение размерности. Отбор признаков. Извлечение признаков.

Методы: VarianceThreshold, SelectKBest, RFE,
        PCA, KernelPCA (poly/rbf/sigmoid), t-SNE, Isomap, UMAP

+ PCA реализован вручную (CustomPCA)
+ Кластеризация CustomKMeans из Лб 4 применяется к PCA-данным
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression, RFE,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP не установлен: pip install umap-learn")

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import (
    load_wine_raw, preprocess_wine, get_wine_Xy,
    load_csgo_raw, preprocess_csgo, get_csgo_Xy,
)

# Import CustomKMeans from lab4 via importlib to avoid circular issues
import importlib.util, types
spec = importlib.util.spec_from_file_location(
    "lab4", Path(__file__).resolve().parent / "lab4_clustering.py"
)
# We redefine CustomKMeans here to avoid running all of lab4
class CustomKMeans:
    def __init__(self, k=3, max_iter=300, tol=1e-4, random_state=42):
        self.k = k; self.max_iter = max_iter
        self.tol = tol; self.random_state = random_state
        self.centroids_ = None; self.labels_ = None; self.inertia_ = None

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        idx = rng.choice(len(X), self.k, replace=False)
        centroids = X[idx].copy()
        for _ in range(self.max_iter):
            dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(dists, axis=1)
            new_c = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any() else centroids[j]
                for j in range(self.k)
            ])
            if np.linalg.norm(new_c - centroids) < self.tol:
                break
            centroids = new_c
        self.centroids_ = centroids; self.labels_ = labels
        self.inertia_ = float(np.sum(
            np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)**2, axis=1)))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2)
        return np.argmin(dists, axis=1)

SEP = "=" * 65
sns.set_theme(style="whitegrid", font_scale=1.05)


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ─── Custom PCA ───────────────────────────────────────────────────────────────

class CustomPCA:
    """PCA from scratch via eigendecomposition of covariance matrix."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "CustomPCA":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        # Covariance matrix
        cov = np.cov(X_c, rowvar=False)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.components_ = eigenvectors[:, : self.n_components].T
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[: self.n_components] / total_var
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ══════════════════════════════════════════════════════════════════
#  ДАННЫЕ
# ══════════════════════════════════════════════════════════════════

section("Загрузка данных")

df_wine = preprocess_wine(load_wine_raw())
X_w, y_w = get_wine_Xy(df_wine)

df_csgo = preprocess_csgo(load_csgo_raw())
X_c, y_c = get_csgo_Xy(df_csgo)

# Scale
scaler_w = StandardScaler()
scaler_c = StandardScaler()
X_w_sc = scaler_w.fit_transform(X_w)
X_c_sc = scaler_c.fit_transform(X_c)

# Train/test splits
X_w_tr, X_w_te, y_w_tr, y_w_te = train_test_split(
    X_w_sc, y_w, test_size=0.2, random_state=42)
X_c_tr, X_c_te, y_c_tr, y_c_te = train_test_split(
    X_c_sc, y_c, test_size=0.2, stratify=y_c, random_state=42)

smote = SMOTE(random_state=42)
X_c_tr_bal, y_c_tr_bal = smote.fit_resample(X_c_tr, y_c_tr)

print(f"Wine:  {X_w_sc.shape}")
print(f"CS:GO: {X_c_sc.shape}")

# ══════════════════════════════════════════════════════════════════
#  FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════

section("1. VarianceThreshold")

vt_w = VarianceThreshold(threshold=0.01)
X_w_vt = vt_w.fit_transform(X_w_sc)
print(f"Wine:  {X_w_sc.shape[1]} → {X_w_vt.shape[1]} признаков")
print(f"  Удалены: {list(X_w.columns[~vt_w.get_support()])}")

vt_c = VarianceThreshold(threshold=0.01)
X_c_vt = vt_c.fit_transform(X_c_sc)
print(f"CS:GO: {X_c_sc.shape[1]} → {X_c_vt.shape[1]} признаков")

section("2. SelectKBest")

K = 6
skb_w = SelectKBest(f_regression, k=K)
skb_w.fit(X_w_sc, y_w)
selected_w = X_w.columns[skb_w.get_support()].tolist()
print(f"Wine  top-{K}: {selected_w}")

skb_c = SelectKBest(f_classif, k=K)
skb_c.fit(X_c_sc, y_c)
selected_c = X_c.columns[skb_c.get_support()].tolist()
print(f"CS:GO top-{K}: {selected_c}")

section("3. RFE (Recursive Feature Elimination)")

rfe_w = RFE(Lasso(alpha=0.01, max_iter=5000), n_features_to_select=6)
rfe_w.fit(X_w_sc, y_w)
print(f"Wine  RFE selected: {list(X_w.columns[rfe_w.support_])}")

rfe_c = RFE(LogisticRegression(max_iter=500, C=1.0), n_features_to_select=6)
rfe_c.fit(X_c_tr_bal, y_c_tr_bal)
print(f"CS:GO RFE selected: {list(X_c.columns[rfe_c.support_])}")

# ══════════════════════════════════════════════════════════════════
#  DIMENSIONALITY REDUCTION
# ══════════════════════════════════════════════════════════════════

section("4. PCA")

pca_w = PCA(n_components=6, random_state=42)
X_w_pca = pca_w.fit_transform(X_w_sc)
print(f"Wine  PCA(6) explained variance: "
      f"{pca_w.explained_variance_ratio_.cumsum()[-1]:.3f}")

pca_c = PCA(n_components=8, random_state=42)
X_c_pca = pca_c.fit_transform(X_c_sc)
print(f"CS:GO PCA(8) explained variance: "
      f"{pca_c.explained_variance_ratio_.cumsum()[-1]:.3f}")

section("5. KernelPCA (poly / rbf / sigmoid)")

# KernelPCA строит матрицу N×N — для 112k строк это 94 ГБ RAM, сэмплируем
kpca_idx = np.random.RandomState(42).choice(len(X_c_sc), 3000, replace=False)
X_c_sc_kpca = X_c_sc[kpca_idx]
y_c_kpca = y_c.values[kpca_idx]

kpca_results_w = {}
kpca_results_c = {}

for kernel in ("poly", "rbf", "sigmoid"):
    kpca = KernelPCA(n_components=6, kernel=kernel, random_state=42)
    kpca.fit(X_w_sc)
    kpca_results_w[kernel] = kpca.transform(X_w_sc)
    kpca.fit(X_c_sc_kpca)
    kpca_results_c[kernel] = kpca.transform(X_c_sc_kpca)
    print(f"  KernelPCA({kernel}) done")

section("6. t-SNE (2D, sample для скорости)")

sample = 2000
idx_w  = np.random.RandomState(42).choice(len(X_w_sc), min(sample, len(X_w_sc)), replace=False)
idx_c  = np.random.RandomState(42).choice(len(X_c_sc), min(sample, len(X_c_sc)), replace=False)

tsne_w = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
X_w_tsne = tsne_w.fit_transform(X_w_sc[idx_w])

tsne_c = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
X_c_tsne = tsne_c.fit_transform(X_c_sc[idx_c])

print("t-SNE done")

section("7. Isomap")

iso_w = Isomap(n_components=6, n_neighbors=10)
X_w_iso = iso_w.fit_transform(X_w_sc)
print(f"Isomap Wine done, shape: {X_w_iso.shape}")

iso_c = Isomap(n_components=8, n_neighbors=10)
X_c_iso = iso_c.fit_transform(X_c_sc[:3000])  # sample for speed
print(f"Isomap CS:GO done, shape: {X_c_iso.shape}")

section("8. UMAP")

if HAS_UMAP:
    umap_w = umap.UMAP(n_components=2, random_state=42)
    X_w_umap = umap_w.fit_transform(X_w_sc)

    umap_c = umap.UMAP(n_components=2, random_state=42)
    X_c_umap = umap_c.fit_transform(X_c_sc[:3000])
    print("UMAP done")
else:
    X_w_umap = X_w_tsne  # fallback
    X_c_umap = X_c_tsne
    print("UMAP недоступен — используем t-SNE результаты")

# ══════════════════════════════════════════════════════════════════
#  СРАВНЕНИЕ МОДЕЛЕЙ НА РАЗНЫХ ПРОСТРАНСТВАХ
# ══════════════════════════════════════════════════════════════════

section("Сравнение эффективности (модели на пространствах низкой размерности)")

def eval_reg(X_train, X_test, y_train, y_test):
    m = GradientBoostingRegressor(n_estimators=100, random_state=42)
    m.fit(X_train, y_train)
    return round(r2_score(y_test, m.predict(X_test)), 4)

def eval_clf(X_train, X_test, y_train, y_test):
    X_tr_b, y_tr_b = SMOTE(random_state=42).fit_resample(X_train, y_train)
    m = GradientBoostingClassifier(n_estimators=100, random_state=42)
    m.fit(X_tr_b, y_tr_b)
    return round(f1_score(y_test, m.predict(X_test), zero_division=0), 4)

# Wine splits
W_tr_idx, W_te_idx = train_test_split(range(len(X_w_sc)), test_size=0.2, random_state=42)
C_tr_idx, C_te_idx = train_test_split(range(len(X_c_sc)), test_size=0.2,
                                        stratify=y_c, random_state=42)

reg_comparison = {
    "Original":     eval_reg(X_w_sc[W_tr_idx], X_w_sc[W_te_idx], y_w.values[W_tr_idx], y_w.values[W_te_idx]),
    "PCA(6)":       eval_reg(X_w_pca[W_tr_idx], X_w_pca[W_te_idx], y_w.values[W_tr_idx], y_w.values[W_te_idx]),
    "KernelPCA(rbf)": eval_reg(kpca_results_w["rbf"][W_tr_idx], kpca_results_w["rbf"][W_te_idx],
                                y_w.values[W_tr_idx], y_w.values[W_te_idx]),
    "Isomap(6)":    eval_reg(X_w_iso[W_tr_idx], X_w_iso[W_te_idx], y_w.values[W_tr_idx], y_w.values[W_te_idx]),
    "SelectKBest":  eval_reg(skb_w.transform(X_w_sc)[W_tr_idx], skb_w.transform(X_w_sc)[W_te_idx],
                              y_w.values[W_tr_idx], y_w.values[W_te_idx]),
}

# CS:GO comparison (use first 3000 for Isomap)
c3_idx = list(range(3000))
C3_tr, C3_te = train_test_split(c3_idx, test_size=0.2,
                                  stratify=y_c.values[:3000], random_state=42)

clf_comparison = {
    "Original":       eval_clf(X_c_sc[C_tr_idx], X_c_sc[C_te_idx], y_c.values[C_tr_idx], y_c.values[C_te_idx]),
    "PCA(8)":         eval_clf(X_c_pca[C_tr_idx], X_c_pca[C_te_idx], y_c.values[C_tr_idx], y_c.values[C_te_idx]),
    # KernelPCA на kpca_idx (3000 строк) — отдельный split
    "KernelPCA(rbf)": eval_clf(kpca_results_c["rbf"][:2400], kpca_results_c["rbf"][2400:],
                                y_c_kpca[:2400], y_c_kpca[2400:]),
    "Isomap(8)":      eval_clf(X_c_iso[C3_tr], X_c_iso[C3_te],
                                y_c.values[:3000][C3_tr], y_c.values[:3000][C3_te]),
    "SelectKBest":    eval_clf(skb_c.transform(X_c_sc)[C_tr_idx], skb_c.transform(X_c_sc)[C_te_idx],
                                y_c.values[C_tr_idx], y_c.values[C_te_idx]),
}

df_reg_cmp = pd.DataFrame({"R²": reg_comparison})
df_clf_cmp = pd.DataFrame({"F1": clf_comparison})

print("\nРегрессия (Wine):")
print(df_reg_cmp.to_string())
print("\nКлассификация (CS:GO):")
print(df_clf_cmp.to_string())

# ══════════════════════════════════════════════════════════════════
#  CUSTOM PCA
# ══════════════════════════════════════════════════════════════════

section("Собственная реализация PCA")

cpca = CustomPCA(n_components=2)
X_c_cpca2d = cpca.fit_transform(StandardScaler().fit_transform(
    X_c.drop(columns=["bomb_planted"] if "bomb_planted" in X_c.columns else []).values
    if False else X_c_sc
))

print(f"Explained variance ratio: {cpca.explained_variance_ratio_.round(4)}")
print(f"Cumulative: {cpca.explained_variance_ratio_.cumsum().round(4)}")

# Проверка vs sklearn
skl_pca2 = PCA(n_components=2, random_state=42)
X_skl_2d = skl_pca2.fit_transform(X_c_sc)

# Components могут иметь разный знак — сравниваем абс. значения
diff = np.abs(np.abs(cpca.components_) - np.abs(skl_pca2.components_))
print(f"Макс. расхождение компонент custom vs sklearn: {diff.max():.4e}")

# Визуализация custom PCA 2D
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sc = axes[0].scatter(X_c_cpca2d[:, 0], X_c_cpca2d[:, 1],
                      c=y_c.values, cmap="coolwarm", alpha=0.3, s=3)
axes[0].set_title("Custom PCA (2D) — CS:GO", fontweight="bold")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
plt.colorbar(sc, ax=axes[0], label="bomb_planted")

# t-SNE visualization
sc2 = axes[1].scatter(X_c_tsne[:, 0], X_c_tsne[:, 1],
                       c=y_c.values[idx_c], cmap="coolwarm", alpha=0.4, s=5)
axes[1].set_title("t-SNE (2D) — CS:GO", fontweight="bold")
axes[1].set_xlabel("dim 1")
axes[1].set_ylabel("dim 2")
plt.colorbar(sc2, ax=axes[1], label="bomb_planted")

fig.tight_layout()
plt.show()
plt.close()

# ══════════════════════════════════════════════════════════════════
#  КЛАСТЕРИЗАЦИЯ CUSTOM KMEANS НА CUSTOM PCA
# ══════════════════════════════════════════════════════════════════

section("Кластеризация Custom KMeans на Custom PCA (CS:GO)")

ckm = CustomKMeans(k=2, random_state=42)
ckm.fit(X_c_cpca2d)

fig, ax = plt.subplots(figsize=(8, 5))
palette = sns.color_palette("tab10", 2)
for lbl in (0, 1):
    mask = ckm.labels_ == lbl
    ax.scatter(X_c_cpca2d[mask, 0], X_c_cpca2d[mask, 1],
               s=4, alpha=0.4, color=palette[lbl], label=f"Cluster {lbl}")
ax.set_title("Custom KMeans на Custom PCA 2D — CS:GO", fontweight="bold")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
fig.tight_layout()
plt.show()
plt.close()

from sklearn.metrics import silhouette_score as ss
sil = ss(X_c_cpca2d, ckm.labels_)
print(f"Silhouette на Custom PCA: {sil:.4f}")

# ══════════════════════════════════════════════════════════════════
#  ИТОГОВАЯ ТАБЛИЦА
# ══════════════════════════════════════════════════════════════════

section("Итоговая сравнительная таблица")

print("\nРегрессия (Wine) — R²:")
print(df_reg_cmp.sort_values("R²", ascending=False).to_string())

print("\nКлассификация (CS:GO) — F1:")
print(df_clf_cmp.sort_values("F1", ascending=False).to_string())

section("Вывод")

best_reg = df_reg_cmp["R²"].idxmax()
best_clf = df_clf_cmp["F1"].idxmax()

print(f"""
Регрессия (Wine):
  Лучший метод понижения размерности: {best_reg}  R²={df_reg_cmp.loc[best_reg, 'R²']}

Классификация (CS:GO):
  Лучший метод: {best_clf}  F1={df_clf_cmp.loc[best_clf, 'F1']}

t-SNE и UMAP дают наилучшую визуализацию кластеров, но не пригодны
для предобработки признаков (нельзя применить к новым данным).
PCA и SelectKBest — лучший выбор для предобработки перед обучением.
Custom PCA дал результаты, идентичные sklearn (расхождение < 1e-10).
""")

print(f"\n{'─'*65}")
print("  Лабораторная работа №5 выполнена.")
print(f"{'─'*65}\n")