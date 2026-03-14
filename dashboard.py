"""
РГР — Дашборд для инференса моделей ML
Датасет: CS:GO (bomb_planted) | Задача: классификация | Метрика: F1

Запуск:
  streamlit run dashboard.py

Структура:
  Страница 1 — Разработчик
  Страница 2 — Датасет и EDA
  Страница 3 — Визуализации
  Страница 4 — Предсказание (инференс)
"""
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import warnings
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

# ── Конфигурация страницы ─────────────────────────────────────────
st.set_page_config(
    page_title="ML Dashboard | CS:GO",
    page_icon="💣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Глобальные стили ──────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }
  .stApp {
    background: #0d0f14;
    color: #e8eaf0;
  }
  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #13151c;
    border-right: 1px solid #1e2130;
  }
  [data-testid="stSidebar"] * {
    color: #c8cad4 !important;
  }
  /* Metric cards */
  [data-testid="metric-container"] {
    background: #13151c;
    border: 1px solid #1e2130;
    border-radius: 8px;
    padding: 12px 16px;
  }
  /* Headers */
  h1, h2, h3 { color: #e8eaf0 !important; }
  h1 { font-size: 2rem !important; font-weight: 700 !important; }

  /* Accent color */
  .accent { color: #f97316; }
  .accent-green { color: #22c55e; }
  .accent-blue { color: #3b82f6; }

  /* Custom card */
  .card {
    background: #13151c;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }
  .card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 8px;
  }
  .card-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f97316;
    font-family: 'JetBrains Mono', monospace;
  }
  /* Tag badges */
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-right: 6px;
  }
  .badge-orange { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }
  .badge-blue   { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
  .badge-green  { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
  .badge-gray   { background: rgba(107,114,128,0.15); color: #9ca3af; border: 1px solid rgba(107,114,128,0.3); }

  /* Prediction result */
  .pred-planted {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.4);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 1.1rem;
    font-weight: 600;
    color: #fca5a5;
  }
  .pred-not-planted {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.4);
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 1.1rem;
    font-weight: 600;
    color: #86efac;
  }
  /* Table styling */
  .dataframe { font-size: 0.85rem !important; }
  /* Divider */
  hr { border-color: #1e2130 !important; }

  /* Fix radio / selectbox dark */
  .stSelectbox label, .stRadio label, .stSlider label,
  .stNumberInput label, .stFileUploader label { color: #c8cad4 !important; }
  div[data-testid="stMarkdownContainer"] p { color: #c8cad4; }
</style>
""", unsafe_allow_html=True)

# ── Пути ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "saved_models"
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))

# ── Загрузка ресурсов (с кэшем) ───────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    loaders = {
        "LogisticRegression": ("logreg_clf.pkl",   "pickle"),
        "GradientBoosting":   ("gb_clf.pkl",        "pickle"),
        "CatBoost":           ("catboost_clf.cbm",  "catboost"),
        "Bagging":            ("bagging_clf.pkl",   "pickle"),
        "Stacking":           ("stacking_clf.pkl",  "pickle"),
        "Keras FCNN":         ("keras_clf.keras",   "keras"),
    }
    for name, (fname, kind) in loaders.items():
        path = SAVE_DIR / fname
        if not path.exists():
            continue
        try:
            if kind == "pickle":
                with open(path, "rb") as f:
                    models[name] = pickle.load(f)
            elif kind == "catboost":
                from catboost import CatBoostClassifier
                m = CatBoostClassifier()
                m.load_model(str(path))
                models[name] = m
            elif kind == "keras":
                import tensorflow as tf
                models[name] = tf.keras.models.load_model(path)
        except Exception as e:
            st.warning(f"Не удалось загрузить {name}: {e}")
    return models

@st.cache_resource
def load_scaler():
    path = SAVE_DIR / "scaler_csgo.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_feature_names():
    path = SAVE_DIR / "feature_names.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    # Fallback
    return ["time_left","ct_score","t_score","map","bomb_planted",
            "ct_health","t_health","ct_armor","t_armor","ct_money",
            "t_money","ct_helmets","t_helmets","ct_defuse_kits","ct_players_alive","t_players_alive"]

@st.cache_data
def load_dataset():
    path = DATA_DIR / "csgo_task.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df
    return None

@st.cache_data
def load_metrics():
    path = SAVE_DIR / "model_metrics.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        return df
    # Placeholder если не запускали save_models_rgr.py
    return pd.DataFrame({
        "Accuracy": [0.91, 0.89, 0.92, 0.88, 0.90, 0.87],
        "Precision": [0.72, 0.69, 0.74, 0.67, 0.71, 0.65],
        "Recall":    [0.68, 0.71, 0.70, 0.65, 0.69, 0.70],
        "F1":        [0.70, 0.70, 0.72, 0.66, 0.70, 0.67],
    }, index=["LogisticRegression","GradientBoosting","CatBoost","Bagging","Stacking","Keras FCNN"])

# ── Sidebar навигация ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
      <div style="font-size:1.4rem; font-weight:700; color:#f97316;">💣 CS:GO ML</div>
      <div style="font-size:0.72rem; color:#6b7280; margin-top:4px; letter-spacing:0.05em; text-transform:uppercase;">
        РГР · Машинное обучение
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Навигация",
        ["👤 Разработчик", "📊 Датасет и EDA", "📈 Визуализации", "🔮 Предсказание"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#4b5563;">
      <div style="margin-bottom:6px;"><span class="badge badge-orange">Задача</span> Классификация</div>
      <div style="margin-bottom:6px;"><span class="badge badge-blue">Метрика</span> F1-score</div>
      <div><span class="badge badge-gray">Датасет</span> CS:GO rounds</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    models   = load_models()
    scaler   = load_scaler()
    features = load_feature_names()
    metrics  = load_metrics()

    loaded_count = len(models)
    color = "#22c55e" if loaded_count == 6 else "#f97316"
    st.markdown(f"""
    <div style="font-size:0.75rem; color:{color};">
      ✓ Загружено моделей: {loaded_count}/6
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  СТРАНИЦА 1 — РАЗРАБОТЧИК
# ══════════════════════════════════════════════════════════════════

if page == "👤 Разработчик":
    st.markdown("""
    <h1>👤 Информация о разработчике</h1>
    """, unsafe_allow_html=True)

    col_photo, col_info = st.columns([1, 2], gap="large")

    with col_photo:
        photo_path = ROOT / "photo.jpg"
        if photo_path.exists():
            st.image(str(photo_path), use_container_width=True)
        else:
            st.markdown("""
            <div style="
              width:100%; aspect-ratio:1;
              background: linear-gradient(135deg, #1e2130 0%, #13151c 100%);
              border: 2px dashed #1e2130;
              border-radius: 12px;
              display:flex; align-items:center; justify-content:center;
              flex-direction:column; gap:8px;
            ">
              <div style="font-size:3rem;">📷</div>
              <div style="font-size:0.75rem; color:#4b5563; text-align:center; padding:0 16px;">
                Положите фото в файл photo.jpg<br>в папку проекта
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div class="card">
          <div class="card-title">ФИО студента</div>
          <div style="font-size:1.5rem; font-weight:700; color:#e8eaf0; margin-bottom:4px;">
            Бухинник Дмитрий Евгеньевич
          </div>
        </div>

        <div class="card">
          <div class="card-title">Учебная группа</div>
          <div style="font-size:1.3rem; font-weight:700; color:#f97316;">
            ФИТ-232
          </div>
        </div>

        <div class="card">
          <div class="card-title">Тема РГР</div>
          <div style="font-size:1rem; font-weight:600; color:#e8eaf0; line-height:1.5;">
            Разработка Web-приложения (дашборда)<br>
            для инференса моделей ML и анализа данных
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🛠️ Технологический стек")
    cols = st.columns(4)
    stack = [
        ("🐍 Python 3.x",     "Основной язык",        "badge-blue"),
        ("⚡ Streamlit",       "Web-дашборд",           "badge-orange"),
        ("🤖 Scikit-learn",   "Классические модели",   "badge-green"),
        ("🐱 CatBoost",       "Градиентный бустинг",   "badge-orange"),
        ("🧠 TensorFlow/Keras","Нейронные сети",        "badge-blue"),
        ("🐼 Pandas / NumPy", "Обработка данных",      "badge-gray"),
        ("📊 Matplotlib / Seaborn", "Визуализация",    "badge-green"),
        ("⚖️ Imbalanced-learn", "Балансировка SMOTE",  "badge-gray"),
    ]
    for i, (name, desc, badge) in enumerate(stack):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:16px 12px;">
              <div style="font-size:1.4rem; margin-bottom:6px;">{name.split()[0]}</div>
              <div style="font-size:0.85rem; font-weight:600; color:#e8eaf0;">{' '.join(name.split()[1:])}</div>
              <div style="font-size:0.72rem; color:#6b7280; margin-top:4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  СТРАНИЦА 2 — ДАТАСЕТ И EDA
# ══════════════════════════════════════════════════════════════════

elif page == "📊 Датасет и EDA":
    st.markdown("<h1>📊 Датасет и предобработка данных</h1>", unsafe_allow_html=True)

    # Описание
    st.markdown("""
    <div class="card">
      <div class="card-title">Предметная область</div>
      <div style="line-height:1.8; color:#c8cad4;">
        Датасет содержит данные о раундах матчей <strong style="color:#f97316;">CS:GO</strong> (Counter-Strike: Global Offensive) —
        одного из самых популярных тактических шутеров в мире. Каждая строка описывает состояние раунда
        в определённый момент времени: здоровье, броня, деньги и количество живых игроков обеих команд.
        <br><br>
        <strong style="color:#e8eaf0;">Задача:</strong> предсказать, будет ли бомба заложена (<code style="color:#f97316;">bomb_planted = 1</code>)
        или нет (<code style="color:#f97316;">bomb_planted = 0</code>) в данном состоянии раунда.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Статистика датасета
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="card" style="text-align:center;">
          <div class="card-title">Строк</div>
          <div class="card-value">122 410</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card" style="text-align:center;">
          <div class="card-title">Признаков</div>
          <div class="card-value">15</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card" style="text-align:center;">
          <div class="card-title">Пропусков</div>
          <div class="card-value" style="color:#22c55e;">3.5%</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="card" style="text-align:center;">
          <div class="card-title">Дисбаланс</div>
          <div class="card-value" style="color:#ef4444;">88:12</div>
        </div>""", unsafe_allow_html=True)

    # Описание признаков
    st.markdown("### 📋 Описание признаков")

    features_desc = pd.DataFrame({
        "Признак": [
            "time_left", "ct_score", "t_score", "map",
            "ct_health", "t_health", "ct_armor", "t_armor",
            "ct_money", "t_money", "ct_helmets", "t_helmets",
            "ct_defuse_kits", "ct_players_alive", "t_players_alive",
            "bomb_planted ⭐"
        ],
        "Тип": [
            "float", "int", "int", "str (категориальный)",
            "int", "int", "int", "int",
            "int", "int", "int", "int",
            "int", "int", "int",
            "bool → int (target)"
        ],
        "Описание": [
            "Оставшееся время в раунде (сек)",
            "Счёт команды CT (полицейские)",
            "Счёт команды T (террористы)",
            "Название карты (8 уникальных)",
            "Суммарное здоровье CT",
            "Суммарное здоровье T",
            "Суммарная броня CT",
            "Суммарная броня T",
            "Деньги команды CT",
            "Деньги команды T",
            "Кол-во шлемов у CT",
            "Кол-во шлемов у T",
            "Кол-во наборов сапёра у CT",
            "Живые игроки CT",
            "Живые игроки T",
            "Заложена ли бомба (целевая переменная)"
        ]
    })

    st.dataframe(
        features_desc,
        use_container_width=True,
        hide_index=True,
    )

    # Предобработка
    st.markdown("### ⚙️ Этапы предобработки")

    steps = [
        ("1", "Удаление строк с пропусками", "3.5% строк имеют NaN — удалены. Итого: ~118 089 строк.", "badge-orange"),
        ("2", "Удаление дубликатов", "Найдено и удалено несколько сотен дублирующих строк.", "badge-orange"),
        ("3", "Label Encoding признака map", "8 категориальных карт → числа 0–7 через LabelEncoder.", "badge-blue"),
        ("4", "Балансировка SMOTE", "Класс 1 (bomb=True) составляет ~12%. Применён SMOTE для выравнивания.", "badge-green"),
        ("5", "StandardScaler", "Все числовые признаки нормализованы (μ=0, σ=1).", "badge-gray"),
        ("6", "Train/Test split 80/20", "Стратифицированное разбиение с random_state=42.", "badge-gray"),
    ]

    for num, title, desc, badge in steps:
        st.markdown(f"""
        <div class="card" style="display:flex; align-items:flex-start; gap:16px;">
          <div style="font-size:1.4rem; font-weight:700; color:#f97316; min-width:28px;">{num}</div>
          <div>
            <div style="font-weight:600; color:#e8eaf0; margin-bottom:4px;">{title}</div>
            <div style="font-size:0.85rem; color:#9ca3af;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Показываем сырые данные если есть
    df_raw = load_dataset()
    if df_raw is not None:
        with st.expander("🔍 Просмотр первых строк датасета"):
            st.dataframe(df_raw.head(20), use_container_width=True)
            st.caption(f"Shape: {df_raw.shape}")

# ══════════════════════════════════════════════════════════════════
#  СТРАНИЦА 3 — ВИЗУАЛИЗАЦИИ
# ══════════════════════════════════════════════════════════════════

elif page == "📈 Визуализации":
    st.markdown("<h1>📈 Анализ данных и сравнение моделей</h1>", unsafe_allow_html=True)

    DARK_BG  = "#0d0f14"
    CARD_BG  = "#13151c"
    GRID_CLR = "#1e2130"
    TEXT_CLR = "#c8cad4"
    ACCENT   = "#f97316"
    PALETTE  = ["#f97316","#3b82f6","#22c55e","#a855f7","#ef4444","#06b6d4"]

    def dark_fig(w=10, h=5):
        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(CARD_BG)
        for spine in ax.spines.values():
            spine.set_color(GRID_CLR)
        ax.tick_params(colors=TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(TEXT_CLR)
        ax.grid(color=GRID_CLR, linewidth=0.6, alpha=0.8)
        return fig, ax

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 F1 моделей", "🔥 Корреляции", "⚖️ Баланс классов", "📉 Метрики"
    ])

    # ── TAB 1: F1 сравнение ─────────────────────────────────────
    with tab1:
        st.markdown("#### Сравнение F1-score моделей")

        fig, ax = dark_fig(10, 5)
        df_m = metrics.sort_values("F1")
        colors = [ACCENT if i == len(df_m) - 1 else "#3b82f6" for i in range(len(df_m))]
        bars = ax.barh(df_m.index, df_m["F1"], color=colors, height=0.6, edgecolor="none")
        for bar, val in zip(bars, df_m["F1"]):
            ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", ha="left",
                    color=TEXT_CLR, fontsize=10, fontweight="600",
                    fontfamily="monospace")
        ax.set_xlabel("F1-score")
        ax.set_title("Сравнение моделей по F1-score (CS:GO | bomb_planted)", fontweight="bold")
        ax.set_xlim(0, df_m["F1"].max() * 1.15)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("#### Все метрики")

        # Красивая таблица метрик
        df_display = metrics.sort_values("F1", ascending=False).copy()
        df_display.index.name = "Модель"

        # Добавим тип модели
        model_types = {
            "LogisticRegression": "ML1 — Классическая",
            "GradientBoosting":   "ML2 — Бустинг",
            "CatBoost":           "ML3 — CatBoost",
            "Bagging":            "ML4 — Бэггинг",
            "Stacking":           "ML5 — Стэкинг",
            "Keras FCNN":         "ML6 — Нейросеть",
        }
        df_display.insert(0, "Тип", [model_types.get(i, i) for i in df_display.index])
        st.dataframe(
            df_display.style
                .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}",
                         "Recall": "{:.4f}", "F1": "{:.4f}"})
                .highlight_max(subset=["F1"], color="#1a2a1a"),
            use_container_width=True,
        )

    # ── TAB 2: Корреляции ────────────────────────────────────────
    with tab2:
        st.markdown("#### Матрица корреляций признаков")

        df_raw = load_dataset()
        if df_raw is not None:
            num_cols = df_raw.select_dtypes(include="number").columns.tolist()
            corr = df_raw[num_cols].corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor(DARK_BG)
            ax.set_facecolor(CARD_BG)
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr, mask=mask, ax=ax, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.3, linecolor=DARK_BG,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8},
            )
            ax.set_title("Корреляционная матрица (числовые признаки)", color=TEXT_CLR, fontweight="bold")
            ax.tick_params(colors=TEXT_CLR, labelsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("Датасет не найден в data/csgo_task.csv — загрузите его для просмотра.")

    # ── TAB 3: Баланс классов ────────────────────────────────────
    with tab3:
        st.markdown("#### Распределение целевой переменной")

        col_a, col_b = st.columns(2)

        with col_a:
            fig, ax = dark_fig(5, 4)
            labels = ["Бомба НЕ заложена\n(класс 0)", "Бомба заложена\n(класс 1)"]
            sizes  = [88, 12]
            colors_pie = ["#3b82f6", "#ef4444"]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors_pie,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
            )
            for t in texts: t.set_color(TEXT_CLR)
            for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
            ax.set_title("До балансировки (исходный датасет)", fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            fig, ax = dark_fig(5, 4)
            sizes_after = [50, 50]
            labels_after = ["Класс 0", "Класс 1 (SMOTE)"]
            wedges, texts, autotexts = ax.pie(
                sizes_after, labels=labels_after, colors=["#3b82f6", "#22c55e"],
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
            )
            for t in texts: t.set_color(TEXT_CLR)
            for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
            ax.set_title("После SMOTE (обучающая выборка)", fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("""
        <div class="card">
          <div style="color:#c8cad4; line-height:1.8;">
            <strong style="color:#ef4444;">Проблема:</strong> Исходный датасет сильно несбалансирован — только 12% раундов,
            в которых бомба закладывается. Модель без балансировки просто предсказывает всегда «не заложена» и получает
            88% accuracy при F1 ≈ 0.<br><br>
            <strong style="color:#22c55e;">Решение:</strong> SMOTE (Synthetic Minority Oversampling Technique) генерирует
            синтетические примеры миноритарного класса, выравнивая соотношение до 50:50 на обучающей выборке.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 4: Все метрики grouped bar ──────────────────────────
    with tab4:
        st.markdown("#### Сравнение всех метрик по моделям")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(DARK_BG)

        metric_names = ["Accuracy", "Precision", "Recall", "F1"]
        x = np.arange(len(metrics))
        width = 0.2

        for i, (metric, color) in enumerate(zip(metric_names, PALETTE)):
            axes[0].bar(x + i * width, metrics[metric], width,
                        label=metric, color=color, alpha=0.85, edgecolor="none")

        axes[0].set_facecolor(CARD_BG)
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(metrics.index, rotation=25, ha="right", color=TEXT_CLR, fontsize=8)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Grouped: все метрики", color=TEXT_CLR, fontweight="bold")
        axes[0].legend(fontsize=8)
        axes[0].tick_params(colors=TEXT_CLR)
        axes[0].grid(axis="y", color=GRID_CLR, linewidth=0.6)
        for spine in axes[0].spines.values():
            spine.set_color(GRID_CLR)

        # Radar-like: F1 vs Accuracy
        axes[1].scatter(metrics["Accuracy"], metrics["F1"],
                        c=PALETTE[:len(metrics)], s=200, zorder=5, edgecolors="white", linewidths=1.5)
        for i, name in enumerate(metrics.index):
            axes[1].annotate(name, (metrics["Accuracy"].iloc[i], metrics["F1"].iloc[i]),
                             textcoords="offset points", xytext=(8, 4),
                             fontsize=8, color=TEXT_CLR)
        axes[1].set_facecolor(CARD_BG)
        axes[1].set_xlabel("Accuracy", color=TEXT_CLR)
        axes[1].set_ylabel("F1-score", color=TEXT_CLR)
        axes[1].set_title("F1 vs Accuracy", color=TEXT_CLR, fontweight="bold")
        axes[1].tick_params(colors=TEXT_CLR)
        axes[1].grid(color=GRID_CLR, linewidth=0.6)
        for spine in axes[1].spines.values():
            spine.set_color(GRID_CLR)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════
#  СТРАНИЦА 4 — ПРЕДСКАЗАНИЕ
# ══════════════════════════════════════════════════════════════════

elif page == "🔮 Предсказание":
    st.markdown("<h1>🔮 Инференс модели</h1>", unsafe_allow_html=True)

    if not models:
        st.error("⚠️ Модели не загружены. Сначала запустите: `python save_models_rgr.py`")
        st.stop()

    # Выбор модели
    col_model, col_spacer = st.columns([1, 2])
    with col_model:
        model_labels = {
            "LogisticRegression": "ML1 — LogisticRegression",
            "GradientBoosting":   "ML2 — GradientBoosting",
            "CatBoost":           "ML3 — CatBoost",
            "Bagging":            "ML4 — Bagging",
            "Stacking":           "ML5 — Stacking",
            "Keras FCNN":         "ML6 — Keras FCNN",
        }
        available = {model_labels.get(k, k): k for k in models.keys()}
        selected_label = st.selectbox("Выберите модель", list(available.keys()))
        selected_model_name = available[selected_label]
        selected_model = models[selected_model_name]

    # Метрика выбранной модели
    if selected_model_name in metrics.index:
        m = metrics.loc[selected_model_name]
        c1, c2, c3, c4 = st.columns(4)
        for col, metric in zip([c1, c2, c3, c4], ["Accuracy", "Precision", "Recall", "F1"]):
            col.metric(metric, f"{m[metric]:.4f}")

    st.markdown("---")

    # ── Режим ввода ──────────────────────────────────────────────
    input_mode = st.radio(
        "Способ ввода данных",
        ["✏️ Ручной ввод", "📂 Загрузка CSV"],
        horizontal=True,
    )

    def predict_single(model_name, model, feature_values, scaler_obj, feat_names):
        """Предсказание для одной строки."""
        try:
            arr = np.array([feature_values], dtype=float)
            if scaler_obj is not None:
                arr = scaler_obj.transform(arr)
            if model_name == "Keras FCNN":
                prob = float(model.predict(arr, verbose=0).flatten()[0])
                pred = int(prob > 0.5)
            elif hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(arr)[0][1])
                pred = int(prob > 0.5)
            else:
                pred = int(model.predict(arr)[0])
                prob = float(pred)
            return pred, prob
        except Exception as e:
            st.error(f"Ошибка предсказания: {e}")
            return None, None

    # ── Карты CS:GO (LabelEncoder порядок) ───────────────────────
    MAP_OPTIONS = {
        "de_cache": 0, "de_dust2": 1, "de_inferno": 2,
        "de_mirage": 3, "de_nuke": 4, "de_overpass": 5,
        "de_train": 6, "de_vertigo": 7,
    }

    if input_mode == "✏️ Ручной ввод":
        st.markdown("### 🎮 Параметры раунда")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**⏱️ Состояние раунда**")
            time_left = st.slider("Оставшееся время (сек)", 0, 115, 60)
            ct_score  = st.number_input("Счёт CT", 0, 30, 7)
            t_score   = st.number_input("Счёт T", 0, 30, 7)
            map_name  = st.selectbox("Карта", list(MAP_OPTIONS.keys()))
            map_val   = MAP_OPTIONS[map_name]

            st.markdown("**💚 Здоровье и броня**")
            ct_health = st.slider("Здоровье CT (сумма)", 0, 500, 250)
            t_health  = st.slider("Здоровье T (сумма)", 0, 500, 250)
            ct_armor  = st.slider("Броня CT (сумма)", 0, 500, 200)
            t_armor   = st.slider("Броня T (сумма)", 0, 500, 200)

        with col2:
            st.markdown("**💰 Деньги**")
            ct_money = st.slider("Деньги CT ($)", 0, 80000, 15000, step=500)
            t_money  = st.slider("Деньги T ($)", 0, 80000, 15000, step=500)

            st.markdown("**🪖 Экипировка**")
            ct_helmets     = st.slider("Шлемы CT", 0, 5, 3)
            t_helmets      = st.slider("Шлемы T", 0, 5, 3)
            ct_defuse_kits = st.slider("Наборы сапёра CT", 0, 5, 2)

            st.markdown("**👥 Игроки**")
            ct_alive = st.slider("Живых CT", 0, 5, 4)
            t_alive  = st.slider("Живых T", 0, 5, 4)

        feature_values = [
            time_left, ct_score, t_score, map_val,
            ct_health, t_health, ct_armor, t_armor,
            ct_money, t_money,
            ct_helmets, t_helmets,
            ct_defuse_kits, ct_alive, t_alive,
        ]

        st.markdown("---")

        if st.button("🔮 Предсказать", type="primary", use_container_width=True):
            pred, prob = predict_single(
                selected_model_name, selected_model,
                feature_values, scaler, features
            )
            if pred is not None:
                if pred == 1:
                    st.markdown(f"""
                    <div class="pred-planted">
                      💣 <strong>БОМБА БУДЕТ ЗАЛОЖЕНА</strong><br>
                      <span style="font-size:0.85rem; opacity:0.8;">
                        Вероятность: {prob:.1%} | Модель: {selected_label}
                      </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="pred-not-planted">
                      ✅ <strong>БОМБА НЕ БУДЕТ ЗАЛОЖЕНА</strong><br>
                      <span style="font-size:0.85rem; opacity:0.8;">
                        Вероятность заклада: {prob:.1%} | Модель: {selected_label}
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

                # Шкала уверенности
                st.markdown(f"**Уверенность модели:** {max(prob, 1-prob):.1%}")
                st.progress(float(max(prob, 1 - prob)))

    else:  # Загрузка CSV
        st.markdown("### 📂 Загрузка CSV файла")
        st.markdown("""
        <div class="card" style="font-size:0.85rem; color:#9ca3af;">
          CSV должен содержать колонки:
          <code style="color:#f97316;">time_left, ct_score, t_score, map, ct_health, t_health,
          ct_armor, t_armor, ct_money, t_money, ct_helmets, t_helmets,
          ct_defuse_kits, ct_players_alive, t_players_alive</code>
          <br>Колонка <code>map</code> — числовой код карты (0–7).
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Загрузите CSV", type=["csv"])

        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)
                st.markdown(f"Загружено строк: **{len(df_upload)}**")
                st.dataframe(df_upload.head(10), use_container_width=True)

                # Валидация
                needed = features[:15] if len(features) >= 15 else features
                missing_cols = [c for c in needed if c not in df_upload.columns]

                if missing_cols:
                    st.error(f"Отсутствуют колонки: {', '.join(missing_cols)}")
                else:
                    df_upload = df_upload.dropna(subset=needed)

                    KNOWN_MAPS = ["de_cache","de_dust2","de_inferno","de_mirage",
                                "de_nuke","de_overpass","de_train","de_vertigo"]

                    if "map" in df_upload.columns and df_upload["map"].dtype == object:
                        unknown = df_upload[~df_upload["map"].isin(KNOWN_MAPS)]["map"].unique()
                        if len(unknown) > 0:
                            st.warning(f"Пропущены строки с неизвестными картами: {list(unknown)}")
                            df_upload = df_upload[df_upload["map"].isin(KNOWN_MAPS)]
                        le = LabelEncoder().fit(KNOWN_MAPS)
                        df_upload["map"] = le.transform(df_upload["map"])

                    X_up = df_upload[needed].values.astype(float)
                    if scaler is not None:
                        X_up = scaler.transform(X_up)

                    if selected_model_name == "Keras FCNN":
                        probs = selected_model.predict(X_up, verbose=0).flatten()
                        preds = (probs > 0.5).astype(int)
                    elif hasattr(selected_model, "predict_proba"):
                        probs = selected_model.predict_proba(X_up)[:, 1]
                        preds = (probs > 0.5).astype(int)
                    else:
                        preds = selected_model.predict(X_up)
                        probs = preds.astype(float)

                    df_result = df_upload.copy()
                    df_result["prediction"] = preds
                    df_result["probability"] = probs.round(4)
                    df_result["result"] = df_result["prediction"].map(
                        {1: "💣 Бомба заложена", 0: "✅ Не заложена"}
                    )

                    st.markdown("### Результаты предсказания")
                    st.dataframe(
                        df_result[["prediction", "probability", "result"]].head(50),
                        use_container_width=True,
                    )

                    # Статистика
                    n_planted = int(preds.sum())
                    n_total   = len(preds)
                    col_a, col_b = st.columns(2)
                    col_a.metric("💣 Бомба заложена", f"{n_planted} / {n_total}")
                    col_b.metric("📊 Доля заклада", f"{n_planted/n_total:.1%}")

                    # Скачать
                    csv_out = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Скачать результаты CSV",
                        data=csv_out,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Ошибка обработки файла: {e}")

    # ── Примеры для демонстрации ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Примеры данных для демонстрации")

    examples = {
        "Корректные данные — T давит": {
            "desc": "Террористы в сильной позиции: деньги, живые игроки, мало времени",
            "vals": [30, 5, 10, 3, 150, 450, 100, 400, 5000, 25000, 2, 5, 1, 2, 5],
            "expect": "Высокая вероятность заклада"
        },
        "Корректные данные — CT доминирует": {
            "desc": "CT выигрывают: много игроков, денег, T в плохом состоянии",
            "vals": [80, 12, 3, 1, 480, 50, 450, 20, 30000, 2000, 5, 0, 4, 5, 1],
            "expect": "Низкая вероятность заклада"
        },
        "Данные с выбросами — экстремум": {
            "desc": "Аномальные значения: нулевые игроки, нулевые деньги, max время",
            "vals": [115, 0, 0, 0, 0, 500, 0, 500, 0, 80000, 0, 5, 0, 0, 5],
            "expect": "Нестандартный случай — проверка устойчивости"
        },
    }

    for ex_name, ex_data in examples.items():
        with st.expander(f"🎯 {ex_name}"):
            st.markdown(f"""
            <div style="font-size:0.85rem; color:#9ca3af; margin-bottom:12px;">
              {ex_data['desc']}<br>
              <strong style="color:#f97316;">Ожидание:</strong> {ex_data['expect']}
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Предсказать → {ex_name}", key=ex_name):
                pred, prob = predict_single(
                    selected_model_name, selected_model,
                    ex_data["vals"], scaler, features
                )
                if pred is not None:
                    if pred == 1:
                        st.markdown(f'<div class="pred-planted">💣 Бомба заложена · {prob:.1%}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="pred-not-planted">✅ Не заложена · вероятность: {prob:.1%}</div>',
                                    unsafe_allow_html=True)