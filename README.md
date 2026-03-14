# ML Project — ОмГТУ, вариант К3/Р5

## Датасеты
- **Регрессия:** Дегустация красного вина (`winequality-red.csv`) — предсказываем `quality`
- **Классификация:** Раунды CS:GO (`csgo_task.csv`) — предсказываем `bomb_planted`

## Структура

```
ml_project/
├── data/                    # CSV датасеты
├── src/
│   ├── data/loader.py       # загрузка и предобработка
│   └── visualization/plots.py  # все графики EDA
├── saved_models/            # сохранённые модели (после lab6)
├── lab0_eda.py
├── lab1_regression.py
├── lab2_classification.py
├── lab3_trees.py
├── lab4_clustering.py
├── lab5_dim_reduction.py
├── lab6_fcnn.py
├── lab7_cnn.py             # требует датасеты (см. ниже)
└── requirements.txt
```

## Установка

```bash
pip install -r requirements.txt
```

## Запуск лабораторных

```bash
python lab0_eda.py
python lab1_regression.py
python lab2_classification.py
python lab3_trees.py
python lab4_clustering.py
python lab5_dim_reduction.py  # требует: pip install umap-learn
python lab6_fcnn.py
python lab7_cnn.py            # требует датасеты изображений
```

## Лабораторная 7 — датасеты изображений

**Часть 1 (кошки/собаки):** скачать по ссылке из задания, распаковать в `data/cats_dogs/`

```
data/cats_dogs/
  train/
    cats/
    dogs/
  validation/
    cats/
    dogs/
```

**Часть 2 (Caltech-101):** скачать с https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip

Выбрать 3 класса, распаковать в `data/caltech101/train/` и `data/caltech101/validation/`

## Сохранённые модели (lab6)

После выполнения `lab6_fcnn.py` в `saved_models/` появятся:
- `mlp_reg.pkl` — sklearn MLP регрессия
- `mlp_clf.pkl` — sklearn MLP классификация
- `keras_reg.keras` — Keras FCNN регрессия
- `keras_clf.keras` — Keras FCNN классификация
- `scaler_wine.pkl` — StandardScaler для вина
- `scaler_csgo.pkl` — StandardScaler для CS:GO
