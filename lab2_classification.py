"""Лабораторная работа №2 — Классификация (CS:GO, bomb_planted)"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.loader import load_csgo_raw, preprocess_csgo, get_csgo_Xy

# ── Custom metrics ────────────────────────────────────────────────
def cm_custom(yt, yp):
    cm = np.zeros((2,2), int)
    for t, p in zip(yt, yp): cm[t,p] += 1
    return cm

def metrics_custom(yt, yp):
    tp=np.sum((yp==1)&(yt==1)); fp=np.sum((yp==1)&(yt==0)); fn=np.sum((yp==0)&(yt==1))
    acc = np.mean(yt==yp)
    pre = tp/(tp+fp) if tp+fp else 0
    rec = tp/(tp+fn) if tp+fn else 0
    f1  = 2*pre*rec/(pre+rec) if pre+rec else 0
    return {"Accuracy":round(acc,4),"Precision":round(pre,4),"Recall":round(rec,4),"F1":round(f1,4)}

# ── Custom kNN ────────────────────────────────────────────────────
class CustomKNN:
    DIST = {
        "euclidean": lambda X,x: np.sqrt(((X-x)**2).sum(axis=1)),
        "manhattan": lambda X,x: np.abs(X-x).sum(axis=1),
        "cosine":    lambda X,x: 1-(X@x)/(np.linalg.norm(X,axis=1)*np.linalg.norm(x)+1e-10),
    }
    def __init__(self, k=5, metric="euclidean"): self.k=k; self.metric=metric
    def fit(self, X, y): self._X=np.array(X,float); self._y=np.array(y,int); return self
    def predict(self, X):
        fn = self.DIST[self.metric]
        preds = []
        for x in np.array(X, float):
            idx = np.argsort(fn(self._X, x))[:self.k]
            labels = self._y[idx]
            preds.append(np.bincount(labels).argmax())
        return np.array(preds)

# ── Data ──────────────────────────────────────────────────────────
df = preprocess_csgo(load_csgo_raw())
X, y = get_csgo_Xy(df)

# Sample 20k — SVM на 118k строк требует слишком много RAM
idx = np.random.RandomState(42).choice(len(X), 20_000, replace=False)
X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr_b, y_tr_b = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_tr, y_tr)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(f"Train: {X_tr_b.shape}  Test: {X_te.shape}")
print(f"Баланс после SMOTE: {pd.Series(y_tr_b).value_counts().to_dict()}")

# ── Models ────────────────────────────────────────────────────────
pipe = lambda clf: Pipeline([("sc", StandardScaler()), ("m", clf)])

models = {
    "LogReg":      pipe(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    "kNN-eucl":    pipe(KNeighborsClassifier(n_neighbors=7, metric="euclidean")),
    "kNN-manh":    pipe(KNeighborsClassifier(n_neighbors=7, metric="manhattan")),
    "kNN-cos":     pipe(KNeighborsClassifier(n_neighbors=7, metric="cosine")),
    "NaiveBayes":  pipe(GaussianNB()),
    "SVM-linear":  pipe(SVC(kernel="linear",  probability=True, class_weight="balanced", max_iter=2000, random_state=42)),
    "SVM-poly":    pipe(SVC(kernel="poly",    probability=True, class_weight="balanced", max_iter=2000, random_state=42)),
    "SVM-rbf":     pipe(SVC(kernel="rbf",     probability=True, class_weight="balanced", random_state=42)),
    "SVM-sigmoid": pipe(SVC(kernel="sigmoid", probability=True, class_weight="balanced", max_iter=2000, random_state=42)),
}

# ── Hyperparameter tuning ─────────────────────────────────────────
print("\n── GridSearchCV ──")
gs = GridSearchCV(models["LogReg"], {"m__C":[0.01,0.1,1,10]}, cv=cv, scoring="f1", n_jobs=-1)
gs.fit(X_tr_b, y_tr_b)
models["LogReg"] = gs.best_estimator_
print(f"LogReg: {gs.best_params_}  F1={gs.best_score_:.4f}")

gs_knn = GridSearchCV(models["kNN-eucl"], {"m__n_neighbors":[3,5,7,11,15]}, cv=cv, scoring="f1", n_jobs=-1)
gs_knn.fit(X_tr_b, y_tr_b)
best_k = gs_knn.best_params_["m__n_neighbors"]
for key, metric in [("kNN-eucl","euclidean"),("kNN-manh","manhattan"),("kNN-cos","cosine")]:
    models[key] = pipe(KNeighborsClassifier(n_neighbors=best_k, metric=metric))
print(f"kNN: best k={best_k}  F1={gs_knn.best_score_:.4f}")

print("\n── Optuna (LogReg) ──")
study = optuna.create_study(direction="maximize")
study.optimize(lambda t: cross_val_score(
    pipe(LogisticRegression(C=t.suggest_float("C",1e-3,100,log=True),
                             max_iter=1000, class_weight="balanced", random_state=42)),
    X_tr_b, y_tr_b, cv=cv, scoring="f1", n_jobs=-1).mean(), n_trials=20)
models["LogReg"] = pipe(LogisticRegression(C=study.best_params["C"], max_iter=1000,
                                            class_weight="balanced", random_state=42))
print(f"Optuna: C={study.best_params['C']:.4f}  F1={study.best_value:.4f}")

# ── Train all ─────────────────────────────────────────────────────
print("\n── Обучение моделей ──")
for name, m in models.items():
    m.fit(X_tr_b, y_tr_b); print(f"  ✓ {name}")

# ── Custom kNN ────────────────────────────────────────────────────
print("\n── Custom kNN ──")
sc = StandardScaler()
Xtr_sc = sc.fit_transform(X_tr_b); Xte_sc = sc.transform(X_te)
s_idx = np.random.RandomState(42).choice(len(Xtr_sc), 3000, replace=False)
for metric in ("euclidean","manhattan","cosine"):
    ck = CustomKNN(k=best_k, metric=metric).fit(Xtr_sc[s_idx], y_tr_b.values[s_idx])
    print(f"  ({metric}): {metrics_custom(y_te.values, ck.predict(Xte_sc))}")

# ── Evaluate ──────────────────────────────────────────────────────
print("\n── Sklearn метрики ──")
results = {}
for name, m in models.items():
    yp = m.predict(X_te)
    has_proba = hasattr(m.named_steps["m"], "predict_proba")
    yprob = m.predict_proba(X_te)[:,1] if has_proba else None
    results[name] = {
        "Accuracy":  round(accuracy_score(y_te, yp), 4),
        "Precision": round(precision_score(y_te, yp, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, yp, zero_division=0), 4),
        "F1":        round(f1_score(y_te, yp, zero_division=0), 4),
        "ROC AUC":   round(roc_auc_score(y_te, yprob), 4) if yprob is not None else None,
    }
df_res = pd.DataFrame(results).T
print(df_res.to_string())

print("\n── Custom метрики ──")
df_custom = pd.DataFrame({n: metrics_custom(y_te.values, m.predict(X_te))
                           for n, m in models.items()}).T
print(df_custom.to_string())
diff = np.abs(df_res[["Accuracy","Precision","Recall","F1"]].astype(float).values
              - df_custom[["Accuracy","Precision","Recall","F1"]].values)
print(f"Макс. расхождение sklearn vs custom: {diff.max():.2e} ✓")

# ── Plots ─────────────────────────────────────────────────────────
# kNN: влияние k
f1_k = [cross_val_score(pipe(KNeighborsClassifier(n_neighbors=k)),
                         X_tr_b, y_tr_b, cv=3, scoring="f1", n_jobs=-1).mean()
         for k in range(1, 20, 2)]
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(list(range(1,20,2)), f1_k, "o-", color="steelblue")
ax.set_xlabel("k"); ax.set_ylabel("CV F1"); ax.set_title("kNN — влияние k на F1", fontweight="bold")
plt.tight_layout(); plt.show(); plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(12,4))
for ax, name in zip(axes, ["LogReg","kNN-eucl","SVM-rbf"]):
    ConfusionMatrixDisplay(confusion_matrix(y_te, models[name].predict(X_te)),
                           display_labels=["Not planted","Planted"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=9)
fig.suptitle("Матрицы ошибок", fontweight="bold"); plt.tight_layout(); plt.show(); plt.close()

# ROC
fig, ax = plt.subplots(figsize=(8,6))
for name, m in models.items():
    if hasattr(m.named_steps["m"], "predict_proba"):
        prob = m.predict_proba(X_te)[:,1]
        fpr, tpr, _ = roc_curve(y_te, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, prob):.3f})")
ax.plot([0,1],[0,1],"k--"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC-кривые", fontweight="bold"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show(); plt.close()

# ── Ошибки 1/2 рода ──────────────────────────────────────────────
best = df_res["F1"].idxmax()
cm = cm_custom(y_te.values, models[best].predict(X_te))
print(f"\nЛучшая модель: {best}  F1={df_res.loc[best,'F1']}")
print(f"FP={cm[0,1]} — ошибка 1 рода (бомба не заложена, модель говорит заложена)")
print(f"FN={cm[1,0]} — ошибка 2 рода (бомба заложена, модель говорит нет) ← критичнее")
print(f"\n{'─'*55}\n  Лабораторная работа №2 выполнена.\n{'─'*55}\n")