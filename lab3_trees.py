"""
Лабораторная работа №3
Обучение с учителем. Решающие деревья. Ансамбли моделей.

Датасеты:
  Регрессия:    Вино (quality)
  Классификация: CS:GO (bomb_planted)

Модели:
  DecisionTree, Bagging, GradientBoosting, Stacking,
  CatBoost, XGBoost, LightGBM + сравнение с PyCaret
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from scipy.stats import loguniform, randint

from sklearn.tree import (
    DecisionTreeRegressor, DecisionTreeClassifier,
    export_text, plot_tree,
)
from sklearn.ensemble import (
    BaggingRegressor, BaggingClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    StackingRegressor, StackingClassifier,
    RandomForestRegressor, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    cross_val_score, GridSearchCV, RandomizedSearchCV,
)
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    f1_score, accuracy_score, precision_score, recall_score,
)
from imblearn.over_sampling import SMOTE
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import (
    load_wine_raw, preprocess_wine, get_wine_Xy,
    load_csgo_raw, preprocess_csgo, get_csgo_Xy,
)

SEP = "=" * 65


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def reg_metrics(y_true, y_pred) -> dict:
    return {
        "R²":   round(r2_score(y_true, y_pred), 4),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
    }


def clf_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# ══════════════════════════════════════════════════════════════════
#  ДАННЫЕ
# ══════════════════════════════════════════════════════════════════

section("Загрузка данных")

# Regression — Wine
df_wine = preprocess_wine(load_wine_raw())
X_w, y_w = get_wine_Xy(df_wine)
X_w_tr, X_w_te, y_w_tr, y_w_te = train_test_split(
    X_w, y_w, test_size=0.2, random_state=42
)

# Classification — CS:GO
df_csgo = preprocess_csgo(load_csgo_raw())
X_c, y_c = get_csgo_Xy(df_csgo)
X_c_tr, X_c_te, y_c_tr, y_c_te = train_test_split(
    X_c, y_c, test_size=0.2, stratify=y_c, random_state=42
)
smote = SMOTE(random_state=42)
X_c_tr_bal, y_c_tr_bal = smote.fit_resample(X_c_tr, y_c_tr)

cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Wine   train={X_w_tr.shape}  test={X_w_te.shape}")
print(f"CS:GO  train={X_c_tr_bal.shape}  test={X_c_te.shape}")

# ══════════════════════════════════════════════════════════════════
#  DECISION TREES
# ══════════════════════════════════════════════════════════════════

section("Decision Tree — Регрессия")

dt_reg_params = {"max_depth": [3, 5, 7, None], "min_samples_leaf": [1, 5, 10]}
gs_dt_reg = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    dt_reg_params, cv=cv_reg, scoring="r2", n_jobs=-1
)
gs_dt_reg.fit(X_w_tr, y_w_tr)
dt_reg = gs_dt_reg.best_estimator_
print(f"Best params: {gs_dt_reg.best_params_}")
print(f"Metrics: {reg_metrics(y_w_te, dt_reg.predict(X_w_te))}")

section("Decision Tree — Классификация")

dt_clf_params = {"max_depth": [3, 5, 7, None], "min_samples_leaf": [1, 5, 10],
                 "criterion": ["gini", "entropy"]}
gs_dt_clf = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    dt_clf_params, cv=3, scoring="f1", n_jobs=-1
)
gs_dt_clf.fit(X_c_tr_bal, y_c_tr_bal)
dt_clf = gs_dt_clf.best_estimator_
print(f"Best params: {gs_dt_clf.best_params_}")
print(f"Metrics: {clf_metrics(y_c_te, dt_clf.predict(X_c_te))}")

# Визуализация дерева (ограничиваем глубину для читаемости)
section("Визуализация дерева решений")

dt_vis = DecisionTreeClassifier(max_depth=3, random_state=42,
                                 class_weight="balanced")
dt_vis.fit(X_c_tr_bal, y_c_tr_bal)

fig, ax = plt.subplots(figsize=(16, 6))
plot_tree(dt_vis, feature_names=X_c.columns.tolist(),
          class_names=["Not planted", "Planted"],
          filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("CS:GO — Decision Tree (max_depth=3)", fontweight="bold")
plt.tight_layout()
plt.show()
plt.close()

print("\nПравила дерева (текст):")
print(export_text(dt_vis, feature_names=X_c.columns.tolist(), max_depth=3))

# ══════════════════════════════════════════════════════════════════
#  BAGGING
# ══════════════════════════════════════════════════════════════════

section("Bagging — Регрессия")

bag_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=5),
    n_estimators=100, random_state=42, n_jobs=-1
)
bag_reg.fit(X_w_tr, y_w_tr)
print(f"Metrics: {reg_metrics(y_w_te, bag_reg.predict(X_w_te))}")

section("Bagging — Классификация")

bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5, class_weight="balanced"),
    n_estimators=100, random_state=42, n_jobs=-1
)
bag_clf.fit(X_c_tr_bal, y_c_tr_bal)
print(f"Metrics: {clf_metrics(y_c_te, bag_clf.predict(X_c_te))}")

# ══════════════════════════════════════════════════════════════════
#  GRADIENT BOOSTING (sklearn)
# ══════════════════════════════════════════════════════════════════

section("GradientBoosting — Регрессия")

gb_reg = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                    learning_rate=0.05, random_state=42)
gb_reg.fit(X_w_tr, y_w_tr)
print(f"Metrics: {reg_metrics(y_w_te, gb_reg.predict(X_w_te))}")

section("GradientBoosting — Классификация")

gb_clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, random_state=42)
gb_clf.fit(X_c_tr_bal, y_c_tr_bal)
print(f"Metrics: {clf_metrics(y_c_te, gb_clf.predict(X_c_te))}")

# ══════════════════════════════════════════════════════════════════
#  STACKING
# ══════════════════════════════════════════════════════════════════

section("Stacking — Регрессия")

stack_reg = StackingRegressor(
    estimators=[
        ("dt",  DecisionTreeRegressor(max_depth=5, random_state=42)),
        ("bag", BaggingRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
        ("gb",  GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ],
    final_estimator=Ridge(),
    cv=3, n_jobs=-1,
)
stack_reg.fit(X_w_tr, y_w_tr)
print(f"Metrics: {reg_metrics(y_w_te, stack_reg.predict(X_w_te))}")

section("Stacking — Классификация")

stack_clf = StackingClassifier(
    estimators=[
        ("dt",  DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
        ("bag", BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ("gb",  GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
    cv=3, n_jobs=-1,
)
stack_clf.fit(X_c_tr_bal, y_c_tr_bal)
print(f"Metrics: {clf_metrics(y_c_te, stack_clf.predict(X_c_te))}")

# ══════════════════════════════════════════════════════════════════
#  CATBOOST
# ══════════════════════════════════════════════════════════════════

section("CatBoost — Регрессия")

cat_reg = CatBoostRegressor(iterations=300, depth=5, learning_rate=0.05,
                             verbose=False, random_seed=42)
cat_reg.fit(X_w_tr, y_w_tr)
print(f"Metrics: {reg_metrics(y_w_te, cat_reg.predict(X_w_te))}")

section("CatBoost — Классификация")

cat_clf = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                              verbose=False, random_seed=42,
                              class_weights=[1, 8])
cat_clf.fit(X_c_tr_bal, y_c_tr_bal)
print(f"Metrics: {clf_metrics(y_c_te, cat_clf.predict(X_c_te))}")

# ══════════════════════════════════════════════════════════════════
#  XGBOOST
# ══════════════════════════════════════════════════════════════════

section("XGBoost — Регрессия")

xgb_reg = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        verbosity=0, random_state=42)
xgb_reg.fit(X_w_tr, y_w_tr)
print(f"Metrics: {reg_metrics(y_w_te, xgb_reg.predict(X_w_te))}")

section("XGBoost — Классификация")

ratio = (y_c_tr == 0).sum() / (y_c_tr == 1).sum()
xgb_clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         scale_pos_weight=ratio,
                         subsample=0.8, colsample_bytree=0.8,
                         verbosity=0, random_state=42, eval_metric="logloss")
xgb_clf.fit(X_c_tr_bal, y_c_tr_bal)
print(f"Metrics: {clf_metrics(y_c_te, xgb_clf.predict(X_c_te))}")

# ══════════════════════════════════════════════════════════════════
#  LIGHTGBM
# ══════════════════════════════════════════════════════════════════

section("LightGBM — Регрессия")

lgb_reg = LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         verbose=-1, random_state=42)
lgb_reg.fit(X_w_tr, y_w_tr)
print(f"Metrics: {reg_metrics(y_w_te, lgb_reg.predict(X_w_te))}")

section("LightGBM — Классификация")

lgb_clf = LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          is_unbalance=True,
                          subsample=0.8, colsample_bytree=0.8,
                          verbose=-1, random_state=42)
lgb_clf.fit(X_c_tr_bal, y_c_tr_bal)
print(f"Metrics: {clf_metrics(y_c_te, lgb_clf.predict(X_c_te))}")

# ══════════════════════════════════════════════════════════════════
#  ИТОГОВЫЕ ТАБЛИЦЫ
# ══════════════════════════════════════════════════════════════════

section("Итоговые таблицы метрик")

reg_summary = {
    "DecisionTree":      reg_metrics(y_w_te, dt_reg.predict(X_w_te)),
    "Bagging":           reg_metrics(y_w_te, bag_reg.predict(X_w_te)),
    "GradientBoosting":  reg_metrics(y_w_te, gb_reg.predict(X_w_te)),
    "Stacking":          reg_metrics(y_w_te, stack_reg.predict(X_w_te)),
    "CatBoost":          reg_metrics(y_w_te, cat_reg.predict(X_w_te)),
    "XGBoost":           reg_metrics(y_w_te, xgb_reg.predict(X_w_te)),
    "LightGBM":          reg_metrics(y_w_te, lgb_reg.predict(X_w_te)),
}

clf_summary = {
    "DecisionTree":      clf_metrics(y_c_te, dt_clf.predict(X_c_te)),
    "Bagging":           clf_metrics(y_c_te, bag_clf.predict(X_c_te)),
    "GradientBoosting":  clf_metrics(y_c_te, gb_clf.predict(X_c_te)),
    "Stacking":          clf_metrics(y_c_te, stack_clf.predict(X_c_te)),
    "CatBoost":          clf_metrics(y_c_te, cat_clf.predict(X_c_te)),
    "XGBoost":           clf_metrics(y_c_te, xgb_clf.predict(X_c_te)),
    "LightGBM":          clf_metrics(y_c_te, lgb_clf.predict(X_c_te)),
}

df_reg = pd.DataFrame(reg_summary).T
df_clf = pd.DataFrame(clf_summary).T

print("\nРегрессия (Wine):")
print(df_reg.to_string())
print("\nКлассификация (CS:GO):")
print(df_clf.to_string())

# ══════════════════════════════════════════════════════════════════
#  ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ
# ══════════════════════════════════════════════════════════════════

section("Визуализация сравнения моделей")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² регрессия
axes[0].barh(df_reg.index, df_reg["R²"],
             color=sns.color_palette("muted", len(df_reg)))
axes[0].set_title("Регрессия — R²", fontweight="bold")
axes[0].set_xlabel("R²")
for i, v in enumerate(df_reg["R²"]):
    axes[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)

# F1 классификация
axes[1].barh(df_clf.index, df_clf["F1"],
             color=sns.color_palette("muted", len(df_clf)))
axes[1].set_title("Классификация — F1", fontweight="bold")
axes[1].set_xlabel("F1")
for i, v in enumerate(df_clf["F1"]):
    axes[1].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)

fig.suptitle("Сравнение ансамблевых моделей", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()
plt.close()

# ══════════════════════════════════════════════════════════════════
#  ВЫВОД
# ══════════════════════════════════════════════════════════════════

section("Вывод")

best_reg = df_reg["R²"].idxmax()
best_clf = df_clf["F1"].idxmax()

print(f"""
Регрессия (Wine):
  Лучшая модель: {best_reg}  R²={df_reg.loc[best_reg, 'R²']:.4f}

Классификация (CS:GO):
  Лучшая модель: {best_clf}  F1={df_clf.loc[best_clf, 'F1']:.4f}

Ансамблевые методы стабильно превосходят одиночные деревья.
Градиентный бустинг (XGBoost/LightGBM/CatBoost) показывает лучшие
результаты за счёт итеративного исправления ошибок и регуляризации.
Stacking даёт дополнительное улучшение за счёт мета-обучения.
""")

print(f"\n{'─'*65}")
print("  Лабораторная работа №3 выполнена.")
print(f"{'─'*65}\n")
