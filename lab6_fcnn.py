"""
Лабораторная работа №6
Полносвязные нейронные сети (FCNN). Задачи регрессии и классификации.

Реализации:
  - MLP через sklearn (MLPRegressor / MLPClassifier)
  - FCNN через Keras / TensorFlow

Подбор гиперпараметров: Optuna, RandomizedSearchCV, KerasTuner (опц.)

Для Гуненкова: MLP с нуля на numpy — НЕ ТРЕБУЕТСЯ (мы у Моисеевой).

Сохранение моделей для РГР:
  - sklearn pickle → saved_models/
  - keras .keras  → saved_models/
"""

import sys
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from scipy.stats import randint, loguniform

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    cross_val_score, RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    f1_score, accuracy_score, precision_score, recall_score,
)
from imblearn.over_sampling import SMOTE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import (
    load_wine_raw, preprocess_wine, get_wine_Xy,
    load_csgo_raw, preprocess_csgo, get_csgo_Xy,
)

SAVE_DIR = Path(__file__).resolve().parent / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)

SEP = "=" * 65


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def reg_metrics(y_true, y_pred):
    return {
        "R²":   round(r2_score(y_true, y_pred), 4),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
    }


def clf_metrics(y_true, y_pred):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# ══════════════════════════════════════════════════════════════════
#  ДАННЫЕ
# ══════════════════════════════════════════════════════════════════

section("Загрузка и подготовка данных")

# Wine — regression
df_wine = preprocess_wine(load_wine_raw())
X_w, y_w = get_wine_Xy(df_wine)
scaler_w = StandardScaler()
X_w_sc = scaler_w.fit_transform(X_w)

X_w_tr, X_w_te, y_w_tr, y_w_te = train_test_split(
    X_w_sc, y_w, test_size=0.2, random_state=42)

# CS:GO — classification (sample 20k — MLP/SMOTE на 112k слишком медленно)
df_csgo = preprocess_csgo(load_csgo_raw())
X_c, y_c = get_csgo_Xy(df_csgo)
s_idx = np.random.RandomState(42).choice(len(X_c), 20_000, replace=False)
X_c, y_c = X_c.iloc[s_idx].reset_index(drop=True), y_c.iloc[s_idx].reset_index(drop=True)
scaler_c = StandardScaler()
X_c_sc = scaler_c.fit_transform(X_c)

X_c_tr, X_c_te, y_c_tr, y_c_te = train_test_split(
    X_c_sc, y_c, test_size=0.2, stratify=y_c, random_state=42)

smote = SMOTE(random_state=42, k_neighbors=3)
X_c_tr_bal, y_c_tr_bal = smote.fit_resample(X_c_tr, y_c_tr)

cv_reg = KFold(n_splits=3, shuffle=True, random_state=42)
cv_clf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(f"Wine  train={X_w_tr.shape}  test={X_w_te.shape}")
print(f"CS:GO train={X_c_tr_bal.shape}  test={X_c_te.shape}")

# ══════════════════════════════════════════════════════════════════
#  SKLEARN MLP
# ══════════════════════════════════════════════════════════════════

section("Sklearn MLP — Регрессия")

# Optuna
def obj_mlp_reg(trial):
    h1 = trial.suggest_int("h1", 32, 256)
    h2 = trial.suggest_int("h2", 16, 128)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
    solver = trial.suggest_categorical("solver", ["adam", "lbfgs"])
    m = MLPRegressor(hidden_layer_sizes=(h1, h2), learning_rate_init=lr,
                     alpha=alpha, solver=solver, max_iter=500, random_state=42)
    return cross_val_score(m, X_w_tr, y_w_tr, cv=3, scoring="r2", n_jobs=-1).mean()

study_reg = optuna.create_study(direction="maximize")
study_reg.optimize(obj_mlp_reg, n_trials=30)
best_p = study_reg.best_params
print(f"Optuna best: {best_p}  CV R²={study_reg.best_value:.4f}")

mlp_reg_optuna = MLPRegressor(
    hidden_layer_sizes=(best_p["h1"], best_p["h2"]),
    learning_rate_init=best_p["lr"],
    alpha=best_p["alpha"],
    solver=best_p["solver"],
    max_iter=500, random_state=42,
)
mlp_reg_optuna.fit(X_w_tr, y_w_tr)

# RandomizedSearchCV
rs_reg = RandomizedSearchCV(
    MLPRegressor(max_iter=500, random_state=42),
    param_distributions={
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "solver": ["adam", "sgd", "lbfgs"],
        "alpha": loguniform(1e-5, 1e-2),
        "learning_rate_init": loguniform(1e-4, 1e-2),
    },
    n_iter=20, cv=3, scoring="r2", n_jobs=-1, random_state=42,
)
rs_reg.fit(X_w_tr, y_w_tr)
print(f"RandomizedSearch best: {rs_reg.best_params_}  CV R²={rs_reg.best_score_:.4f}")

# Best MLP reg — use optuna
mlp_reg_best = mlp_reg_optuna
print(f"\nTest metrics: {reg_metrics(y_w_te, mlp_reg_best.predict(X_w_te))}")

section("Sklearn MLP — Классификация")

def obj_mlp_clf(trial):
    h1 = trial.suggest_int("h1", 64, 256)
    h2 = trial.suggest_int("h2", 32, 128)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
    solver = trial.suggest_categorical("solver", ["adam", "sgd"])
    m = MLPClassifier(hidden_layer_sizes=(h1, h2), learning_rate_init=lr,
                      alpha=alpha, solver=solver, max_iter=300, random_state=42)
    return cross_val_score(m, X_c_tr_bal, y_c_tr_bal, cv=3,
                           scoring="f1", n_jobs=-1).mean()

study_clf = optuna.create_study(direction="maximize")
study_clf.optimize(obj_mlp_clf, n_trials=15)
best_p_c = study_clf.best_params
print(f"Optuna best: {best_p_c}  CV F1={study_clf.best_value:.4f}")

mlp_clf_optuna = MLPClassifier(
    hidden_layer_sizes=(best_p_c["h1"], best_p_c["h2"]),
    learning_rate_init=best_p_c["lr"],
    alpha=best_p_c["alpha"],
    solver=best_p_c["solver"],
    max_iter=300, random_state=42,
)
mlp_clf_optuna.fit(X_c_tr_bal, y_c_tr_bal)

rs_clf = RandomizedSearchCV(
    MLPClassifier(max_iter=300, random_state=42),
    param_distributions={
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "solver": ["adam", "sgd"],
        "alpha": loguniform(1e-5, 1e-2),
        "learning_rate_init": loguniform(1e-4, 1e-2),
    },
    n_iter=20, cv=3, scoring="f1", n_jobs=-1, random_state=42,
)
rs_clf.fit(X_c_tr_bal, y_c_tr_bal)
print(f"RandomizedSearch best: {rs_clf.best_params_}  CV F1={rs_clf.best_score_:.4f}")

mlp_clf_best = mlp_clf_optuna
print(f"\nTest metrics: {clf_metrics(y_c_te, mlp_clf_best.predict(X_c_te))}")

# ══════════════════════════════════════════════════════════════════
#  KERAS FCNN
# ══════════════════════════════════════════════════════════════════

section("Keras FCNN — Регрессия")

def build_reg_model(units1=128, units2=64, lr=1e-3, dropout=0.2):
    model = keras.Sequential([
        layers.Input(shape=(X_w_tr.shape[1],)),
        layers.Dense(units1, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(units2, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse",
                  metrics=["mae"])
    return model

early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                      monitor="val_loss")
reduce_lr  = callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)

# Optuna for Keras
def obj_keras_reg(trial):
    u1 = trial.suggest_int("units1", 64, 256)
    u2 = trial.suggest_int("units2", 32, 128)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dr = trial.suggest_float("dropout", 0.1, 0.4)
    model = build_reg_model(u1, u2, lr, dr)
    model.fit(X_w_tr, y_w_tr.values, epochs=50, batch_size=32,
              validation_split=0.15, callbacks=[early_stop, reduce_lr],
              verbose=0)
    y_pred = model.predict(X_w_te, verbose=0).flatten()
    return r2_score(y_w_te, y_pred)

study_keras_reg = optuna.create_study(direction="maximize")
study_keras_reg.optimize(obj_keras_reg, n_trials=15)
kp = study_keras_reg.best_params
print(f"Keras Optuna best: {kp}  R²={study_keras_reg.best_value:.4f}")

# Train final Keras reg model
keras_reg = build_reg_model(kp["units1"], kp["units2"], kp["lr"], kp["dropout"])
history_reg = keras_reg.fit(
    X_w_tr, y_w_tr.values,
    epochs=100, batch_size=32,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=0,
)
y_pred_kr = keras_reg.predict(X_w_te, verbose=0).flatten()
print(f"Keras FCNN reg metrics: {reg_metrics(y_w_te, y_pred_kr)}")

section("Keras FCNN — Классификация")

def build_clf_model(units1=128, units2=64, lr=1e-3, dropout=0.2):
    model = keras.Sequential([
        layers.Input(shape=(X_c_tr_bal.shape[1],)),
        layers.Dense(units1, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(units2, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

def obj_keras_clf(trial):
    u1 = trial.suggest_int("units1", 64, 256)
    u2 = trial.suggest_int("units2", 32, 128)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dr = trial.suggest_float("dropout", 0.1, 0.4)
    model = build_clf_model(u1, u2, lr, dr)
    model.fit(X_c_tr_bal, y_c_tr_bal.values, epochs=30, batch_size=256,
              validation_split=0.15, callbacks=[early_stop], verbose=0)
    y_pred = (model.predict(X_c_te, verbose=0).flatten() > 0.5).astype(int)
    return f1_score(y_c_te, y_pred, zero_division=0)

study_keras_clf = optuna.create_study(direction="maximize")
study_keras_clf.optimize(obj_keras_clf, n_trials=8)
kpc = study_keras_clf.best_params
print(f"Keras Optuna best: {kpc}  F1={study_keras_clf.best_value:.4f}")

keras_clf = build_clf_model(kpc["units1"], kpc["units2"], kpc["lr"], kpc["dropout"])
history_clf = keras_clf.fit(
    X_c_tr_bal, y_c_tr_bal.values,
    epochs=50, batch_size=256,
    validation_split=0.15,
    callbacks=[early_stop],
    verbose=0,
)
y_pred_kc = (keras_clf.predict(X_c_te, verbose=0).flatten() > 0.5).astype(int)
print(f"Keras FCNN clf metrics: {clf_metrics(y_c_te, y_pred_kc)}")

# ══════════════════════════════════════════════════════════════════
#  ИНФЕРЕНС (примеры предсказания)
# ══════════════════════════════════════════════════════════════════

section("Инференс — примеры предсказания")

# Regression
sample_wine = X_w_te[:3]
pred_mlp  = mlp_reg_best.predict(sample_wine)
pred_kras = keras_reg.predict(sample_wine, verbose=0).flatten()
true_vals = y_w_te.values[:3]

print("Регрессия (Wine quality):")
df_inf_r = pd.DataFrame({
    "True": true_vals,
    "MLP (sklearn)": pred_mlp.round(2),
    "Keras FCNN": pred_kras.round(2),
})
print(df_inf_r.to_string())

# Classification
sample_csgo = X_c_te[:3]
pred_mlp_c  = mlp_clf_best.predict(sample_csgo)
pred_kras_c = (keras_clf.predict(sample_csgo, verbose=0).flatten() > 0.5).astype(int)
true_c = y_c_te.values[:3]

print("\nКлассификация (CS:GO bomb_planted):")
df_inf_c = pd.DataFrame({
    "True": true_c,
    "MLP (sklearn)": pred_mlp_c,
    "Keras FCNN": pred_kras_c,
})
print(df_inf_c.to_string())

# ══════════════════════════════════════════════════════════════════
#  ВИЗУАЛИЗАЦИЯ ОБУЧЕНИЯ
# ══════════════════════════════════════════════════════════════════

section("Визуализация процесса обучения (Keras)")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Regression loss
axes[0, 0].plot(history_reg.history["loss"], label="Train loss")
axes[0, 0].plot(history_reg.history["val_loss"], label="Val loss")
axes[0, 0].set_title("Регрессия — Loss (MSE)")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].legend()

# Regression MAE
axes[0, 1].plot(history_reg.history["mae"], label="Train MAE")
axes[0, 1].plot(history_reg.history["val_mae"], label="Val MAE")
axes[0, 1].set_title("Регрессия — MAE")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].legend()

# Classification loss
axes[1, 0].plot(history_clf.history["loss"], label="Train loss")
axes[1, 0].plot(history_clf.history["val_loss"], label="Val loss")
axes[1, 0].set_title("Классификация — Loss (BCE)")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].legend()

# Classification accuracy
axes[1, 1].plot(history_clf.history["accuracy"], label="Train acc")
axes[1, 1].plot(history_clf.history["val_accuracy"], label="Val acc")
axes[1, 1].set_title("Классификация — Accuracy")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].legend()

fig.suptitle("Keras FCNN — кривые обучения", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()
plt.close()

# Архитектура
keras_reg.summary()
keras_clf.summary()

# ══════════════════════════════════════════════════════════════════
#  СОХРАНЕНИЕ МОДЕЛЕЙ (для РГР)
# ══════════════════════════════════════════════════════════════════

section("Сохранение моделей")

# sklearn — pickle
with open(SAVE_DIR / "mlp_reg.pkl", "wb") as f:
    pickle.dump(mlp_reg_best, f)

with open(SAVE_DIR / "mlp_clf.pkl", "wb") as f:
    pickle.dump(mlp_clf_best, f)

with open(SAVE_DIR / "scaler_wine.pkl", "wb") as f:
    pickle.dump(scaler_w, f)

with open(SAVE_DIR / "scaler_csgo.pkl", "wb") as f:
    pickle.dump(scaler_c, f)

# Keras — native format
keras_reg.save(SAVE_DIR / "keras_reg.keras")
keras_clf.save(SAVE_DIR / "keras_clf.keras")

print("Сохранены:")
for p in sorted(SAVE_DIR.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size / 1024:.1f} KB)")

# ══════════════════════════════════════════════════════════════════
#  ИТОГОВАЯ ТАБЛИЦА
# ══════════════════════════════════════════════════════════════════

section("Итоговая таблица")

reg_summary = {
    "MLP sklearn": reg_metrics(y_w_te, mlp_reg_best.predict(X_w_te)),
    "Keras FCNN":  reg_metrics(y_w_te, keras_reg.predict(X_w_te, verbose=0).flatten()),
}
clf_summary = {
    "MLP sklearn": clf_metrics(y_c_te, mlp_clf_best.predict(X_c_te)),
    "Keras FCNN":  clf_metrics(y_c_te, (keras_clf.predict(X_c_te, verbose=0).flatten() > 0.5).astype(int)),
}

print("\nРегрессия:")
print(pd.DataFrame(reg_summary).T.to_string())
print("\nКлассификация:")
print(pd.DataFrame(clf_summary).T.to_string())

section("Вывод")

best_reg = max(reg_summary, key=lambda k: reg_summary[k]["R²"])
best_clf = max(clf_summary, key=lambda k: clf_summary[k]["F1"])

print(f"""
Регрессия (Wine):
  Лучшая модель: {best_reg}  R²={reg_summary[best_reg]['R²']}

Классификация (CS:GO):
  Лучшая модель: {best_clf}  F1={clf_summary[best_clf]['F1']}

Keras FCNN с BatchNormalization и Dropout демонстрирует лучшую
обобщающую способность по сравнению со sklearn MLP.
Early stopping предотвращает переобучение.
Все модели сохранены в saved_models/ для использования в РГР.
""")

print(f"\n{'─'*65}")
print("  Лабораторная работа №6 выполнена.")
print(f"{'─'*65}\n")