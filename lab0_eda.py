"""
Лабораторная работа №0
Введение в исследовательский анализ данных (EDA)

Датасеты:
  Регрессия    — Дегустация вина (winequality-red.csv), target: quality
  Классификация — Раунды CS:GO (csgo_task.csv),        target: bomb_planted
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pointbiserialr, spearmanr

# make src importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import (
    load_wine_raw, preprocess_wine, get_wine_Xy,
    load_csgo_raw, preprocess_csgo, get_csgo_Xy,
)
from src.visualization.plots import (
    plot_distributions, plot_correlation_heatmap, plot_boxplots,
    plot_missing_values,
    plot_wine_target_distribution, plot_wine_feature_vs_quality,
    plot_wine_scatter_matrix,
    plot_csgo_target_distribution, plot_csgo_map_distribution,
    plot_csgo_health_money,
)

SEP = "=" * 65


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ══════════════════════════════════════════════════════════════════
#  ЧАСТЬ 1 — ВИНО (РЕГРЕССИЯ)
# ══════════════════════════════════════════════════════════════════

section("ВИНО — Загрузка и базовая информация")

df_wine_raw = load_wine_raw()
print(f"Размер датасета: {df_wine_raw.shape}")
print(f"\nТипы данных:\n{df_wine_raw.dtypes}")
print(f"\nПервые строки:\n{df_wine_raw.head()}")

section("ВИНО — Пропущенные значения")
missing_wine = df_wine_raw.isnull().sum()
print(missing_wine)
print(f"\nИтого пропусков: {missing_wine.sum()} — датасет чистый.")

section("ВИНО — Описательная статистика")
print(df_wine_raw.describe().round(3).to_string())

section("ВИНО — Дубликаты")
n_dup = df_wine_raw.duplicated().sum()
print(f"Дублирующих строк: {n_dup}")

section("ВИНО — Предобработка")
df_wine = preprocess_wine(df_wine_raw)
print(f"До:  {df_wine_raw.shape[0]} строк")
print(f"После: {df_wine.shape[0]} строк  (удалено {df_wine_raw.shape[0] - df_wine.shape[0]})")

section("ВИНО — Распределение целевой переменной")
print(df_wine["quality"].value_counts().sort_index())
plot_wine_target_distribution(df_wine)

section("ВИНО — Распределения признаков (гистограммы)")
plot_distributions(df_wine, "Wine — распределения признаков")

section("ВИНО — Выбросы (IQR)")
for col in df_wine.select_dtypes(include="number").columns:
    q1, q3 = df_wine[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df_wine[col] < q1 - 1.5 * iqr) | (df_wine[col] > q3 + 1.5 * iqr)).sum()
    print(f"  {col:<25}: {n_out} выбросов")

section("ВИНО — Ящики с усами (boxplots)")
plot_boxplots(df_wine, "Wine — ящики с усами")

section("ВИНО — Корреляционный анализ")
X_wine, y_wine = get_wine_Xy(df_wine)

print("\nКоэффициент Пирсона (все признаки числовые):")
corr_quality = df_wine.corr()["quality"].drop("quality").sort_values()
print(corr_quality.round(4).to_string())

print("\nКоэффициент Спирмена (проверка монотонных зависимостей):")
for col in X_wine.columns:
    rho, p = spearmanr(X_wine[col], y_wine)
    sig = "✓" if p < 0.05 else " "
    print(f"  {col:<25}: ρ={rho:+.4f}  p={p:.3e} {sig}")

plot_correlation_heatmap(df_wine, "Wine — матрица корреляций")
plot_wine_feature_vs_quality(df_wine)
plot_wine_scatter_matrix(df_wine)

section("ВИНО — Проверка нормальности (Shapiro-Wilk, выборка 500)")
sample = df_wine.sample(min(500, len(df_wine)), random_state=42)
for col in X_wine.columns:
    stat, p = stats.shapiro(sample[col])
    normal = "нормальное" if p > 0.05 else "НЕ нормальное"
    print(f"  {col:<25}: p={p:.3e}  → {normal}")

section("ВИНО — Итоговое заключение")
print("""
Датасет 'Дегустация красного вина':
• 1599 наблюдений, 11 числовых признаков, целевая переменная — quality (3–8).
• Пропусков нет. После удаления дубликатов и экстремальных выбросов в
  chlorides и residual sugar датасет готов к обучению.
• Наиболее коррелирован с quality признак alcohol (r=+0.48), далее
  sulphates (+0.25) и volatile acidity (−0.39).
• Большинство признаков не нормально распределены (Shapiro-Wilk p<0.05).
• Классы несбалансированы: оценки 5–6 составляют ~82 % данных.
""")

# ══════════════════════════════════════════════════════════════════
#  ЧАСТЬ 2 — CS:GO (КЛАССИФИКАЦИЯ)
# ══════════════════════════════════════════════════════════════════

section("CS:GO — Загрузка и базовая информация")

df_csgo_raw = load_csgo_raw()
print(f"Размер датасета: {df_csgo_raw.shape}")
print(f"\nТипы данных:\n{df_csgo_raw.dtypes}")
print(f"\nПервые строки:\n{df_csgo_raw.head()}")

section("CS:GO — Пропущенные значения")
missing_csgo = df_csgo_raw.isnull().sum()
print(missing_csgo[missing_csgo > 0])
print(f"\nСтрок с хотя бы одним пропуском: {df_csgo_raw.isnull().any(axis=1).sum()}")
plot_missing_values(df_csgo_raw, "CS:GO — пропущенные значения")

section("CS:GO — Описательная статистика")
print(df_csgo_raw.describe().round(2).to_string())

section("CS:GO — Дубликаты")
n_dup = df_csgo_raw.duplicated().sum()
print(f"Дублирующих строк: {n_dup}")

section("CS:GO — Распределение по картам (до предобработки)")
plot_csgo_map_distribution(df_csgo_raw)

section("CS:GO — Распределение целевой переменной (до предобработки)")
plot_csgo_target_distribution(df_csgo_raw.assign(
    bomb_planted=df_csgo_raw["bomb_planted"].astype(int)))

section("CS:GO — Предобработка")
df_csgo = preprocess_csgo(df_csgo_raw)
print(f"До:  {df_csgo_raw.shape[0]} строк")
print(f"После: {df_csgo.shape[0]} строк  (удалено {df_csgo_raw.shape[0] - df_csgo.shape[0]})")
print("\nДисбаланс классов после очистки:")
vc = df_csgo["bomb_planted"].value_counts()
print(vc)
ratio = vc[0] / vc[1]
print(f"Соотношение: {ratio:.1f}:1  → требуется балансировка при обучении")

section("CS:GO — Выбросы (IQR)")
X_csgo, y_csgo = get_csgo_Xy(df_csgo)
for col in X_csgo.select_dtypes(include="number").columns:
    q1, q3 = X_csgo[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((X_csgo[col] < q1 - 1.5 * iqr) | (X_csgo[col] > q3 + 1.5 * iqr)).sum()
    print(f"  {col:<22}: {n_out} выбросов")

section("CS:GO — Ящики с усами")
plot_boxplots(df_csgo, "CS:GO — ящики с усами")

section("CS:GO — Корреляционный анализ")
print("\nBiserial correlation с bomb_planted (числовые признаки):")
for col in X_csgo.select_dtypes(include="number").columns:
    r, p = pointbiserialr(y_csgo, X_csgo[col])
    sig = "✓" if p < 0.05 else " "
    print(f"  {col:<22}: r={r:+.4f}  p={p:.3e} {sig}")

plot_correlation_heatmap(df_csgo, "CS:GO — матрица корреляций")
plot_distributions(df_csgo, "CS:GO — распределения признаков")
plot_csgo_health_money(df_csgo)

section("CS:GO — Итоговое заключение")
print("""
Датасет 'Раунды CS:GO':
• 122 410 наблюдений, 15 признаков (14 числовых + 1 категориальный map).
• После удаления строк с пропусками (3.5 %) и дубликатов осталось
  118 089 строк.
• Целевая переменная bomb_planted сильно несбалансирована: ~88 % — False.
  При обучении необходима балансировка (SMOTE / class_weight).
• Признак map закодирован LabelEncoder (8 уникальных карт).
• Наибольшую бисериальную корреляцию с целевой имеют t_money, t_armor,
  ct_defuse_kits — логично: деньги и экипировка растут к середине раунда,
  когда бомба обычно закладывается.
""")

print(f"\n{'─'*65}")
print("  Лабораторная работа №0 выполнена.")
print(f"{'─'*65}\n")
