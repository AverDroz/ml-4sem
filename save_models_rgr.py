"""
Сохранение 6 моделей классификации (CS:GO) для РГР.
Запускать ОДИН РАЗ из папки ml_project/.

Сохраняет в saved_models/:
  logreg_clf.pkl        — ML1: LogisticRegression (классическая)
  gb_clf.pkl            — ML2: GradientBoosting (бустинг sklearn)
  catboost_clf.cbm      — ML3: CatBoost
  bagging_clf.pkl       — ML4: BaggingClassifier
  stacking_clf.pkl      — ML5: StackingClassifier
  keras_clf.keras       — ML6: Keras FCNN (уже есть из lab6)
  scaler_csgo.pkl       — StandardScaler (уже есть из lab6)
"""

import sys
import pickle
import warnings
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier, BaggingClassifier,
    StackingClassifier, RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.loader import load_csgo_raw, preprocess_csgo, get_csgo_Xy

SAVE_DIR = Path(__file__).resolve().parent / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)

SEP = "=" * 60

def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def clf_metrics(yt, yp):
    return {
        "Accuracy":  round(accuracy_score(yt, yp), 4),
        "Precision": round(precision_score(yt, yp, zero_division=0), 4),
        "Recall":    round(recall_score(yt, yp, zero_division=0), 4),
        "F1":        round(f1_score(yt, yp, zero_division=0), 4),
    }

# ── Данные ────────────────────────────────────────────────────────
section("Загрузка данных")
df = preprocess_csgo(load_csgo_raw())
X, y = get_csgo_Xy(df)

# Sample 30k — баланс скорость/качество
idx = np.random.RandomState(42).choice(len(X), 30_000, replace=False)
X = X.iloc[idx].reset_index(drop=True)
y = y.iloc[idx].reset_index(drop=True)

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc, y, test_size=0.2, stratify=y, random_state=42
)
X_tr_bal, y_tr_bal = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_tr, y_tr)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(f"Train: {X_tr_bal.shape}  Test: {X_te.shape}")
print(f"Баланс SMOTE: {pd.Series(y_tr_bal).value_counts().to_dict()}")

results = {}

# ── ML1: LogisticRegression ───────────────────────────────────────
section("ML1 — LogisticRegression")
logreg = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
logreg.fit(X_tr_bal, y_tr_bal)
m = clf_metrics(y_te, logreg.predict(X_te))
results["LogisticRegression"] = m
print(m)
with open(SAVE_DIR / "logreg_clf.pkl", "wb") as f:
    pickle.dump(logreg, f)
print("  ✓ Сохранено: logreg_clf.pkl")

# ── ML2: GradientBoosting ─────────────────────────────────────────
section("ML2 — GradientBoosting")
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                 learning_rate=0.05, random_state=42)
gb.fit(X_tr_bal, y_tr_bal)
m = clf_metrics(y_te, gb.predict(X_te))
results["GradientBoosting"] = m
print(m)
with open(SAVE_DIR / "gb_clf.pkl", "wb") as f:
    pickle.dump(gb, f)
print("  ✓ Сохранено: gb_clf.pkl")

# ── ML3: CatBoost ─────────────────────────────────────────────────
section("ML3 — CatBoost")
cat = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                          verbose=False, random_seed=42, class_weights=[1, 8])
cat.fit(X_tr_bal, y_tr_bal)
m = clf_metrics(y_te, cat.predict(X_te))
results["CatBoost"] = m
print(m)
cat.save_model(str(SAVE_DIR / "catboost_clf.cbm"))
print("  ✓ Сохранено: catboost_clf.cbm")

# ── ML4: Bagging ──────────────────────────────────────────────────
section("ML4 — BaggingClassifier")
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5, class_weight="balanced"),
    n_estimators=100, random_state=42, n_jobs=-1
)
bag.fit(X_tr_bal, y_tr_bal)
m = clf_metrics(y_te, bag.predict(X_te))
results["Bagging"] = m
print(m)
with open(SAVE_DIR / "bagging_clf.pkl", "wb") as f:
    pickle.dump(bag, f)
print("  ✓ Сохранено: bagging_clf.pkl")

# ── ML5: Stacking ─────────────────────────────────────────────────
section("ML5 — StackingClassifier")
stack = StackingClassifier(
    estimators=[
        ("dt",  DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
        ("bag", BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ("gb",  GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
    cv=3, n_jobs=-1,
)
stack.fit(X_tr_bal, y_tr_bal)
m = clf_metrics(y_te, stack.predict(X_te))
results["Stacking"] = m
print(m)
with open(SAVE_DIR / "stacking_clf.pkl", "wb") as f:
    pickle.dump(stack, f)
print("  ✓ Сохранено: stacking_clf.pkl")

# ── ML6: Keras FCNN (уже есть) ────────────────────────────────────
section("ML6 — Keras FCNN")
keras_path = SAVE_DIR / "keras_clf.keras"
if keras_path.exists():
    import tensorflow as tf
    keras_clf = tf.keras.models.load_model(keras_path)
    y_pred_k = (keras_clf.predict(X_te, verbose=0).flatten() > 0.5).astype(int)
    m = clf_metrics(y_te, y_pred_k)
    results["Keras FCNN"] = m
    print(m)
    print("  ✓ Уже существует: keras_clf.keras")
else:
    print("  ⚠ keras_clf.keras не найден — сначала запусти lab6_fcnn.py")

# ── Scaler (уже есть, но пересохраняем с текущими данными) ────────
with open(SAVE_DIR / "scaler_csgo.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("\n  ✓ Пересохранено: scaler_csgo.pkl")

# ── Итоговая таблица ──────────────────────────────────────────────
section("Итоговая таблица F1")
df_res = pd.DataFrame(results).T.sort_values("F1", ascending=False)
print(df_res.to_string())

# Сохраняем таблицу метрик для дашборда
df_res.to_csv(SAVE_DIR / "model_metrics.csv")
print("\n  ✓ Сохранено: model_metrics.csv")

# Сохраняем имена признаков для дашборда
feature_names = list(X.columns)
with open(SAVE_DIR / "feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print("  ✓ Сохранено: feature_names.pkl")

print(f"\n{'─'*60}")
print("  Все модели сохранены. Теперь запускай: streamlit run dashboard.py")
print(f"{'─'*60}\n")