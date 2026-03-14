"""
Лабораторная работа №7
Сверточные нейронные сети (CNN). Классификация изображений.

Часть 1: Своя CNN — датасет кошки/собаки
Часть 2: Transfer Learning (Fine-Tuning) — Caltech-101 (3 класса)
         Модели: InceptionV3, VGG19
"""

import os
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, VGG19
from tensorflow.keras import callbacks

tf.get_logger().setLevel("ERROR")

SEP = "=" * 65


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ── Параметры ─────────────────────────────────────────────────────
IMG_SIZE   = (150, 150)
BATCH_SIZE = 32
EPOCHS_OWN = 20   # своя CNN
EPOCHS_FT  = 10   # fine-tuning

# ── Флаги запуска ─────────────────────────────────────────────────
SKIP_PART1 = False   # True = пропустить обучение Части 1 (использовать сохранённую модель)
MODEL_PATH = Path("saved_models/catdog_cnn.keras")

early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                      monitor="val_accuracy")
reduce_lr  = callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=0)

# ══════════════════════════════════════════════════════════════════
#  ЧАСТЬ 1 — СВОЯ CNN (КОШКИ / СОБАКИ)
# ══════════════════════════════════════════════════════════════════

section("Часть 1 — Собственная CNN: кошки и собаки")

CATS_DOGS_DIR = Path("data/cats_dogs")

if not CATS_DOGS_DIR.exists():
    print(f"""
  ⚠  Датасет не найден: {CATS_DOGS_DIR}
  
  Скачайте датасет по ссылке из задания и распакуйте так:
    data/
      cats_dogs/
        train/
          cats/   ← .jpg изображения кошек
          dogs/   ← .jpg изображения собак
        validation/
          cats/
          dogs/

  Либо используйте tf.keras.utils.get_file() для автоматической загрузки.
""")
    # Создаём заглушку для демонстрации кода без данных
    CATS_DOGS_AVAILABLE = False
else:
    CATS_DOGS_AVAILABLE = True

if CATS_DOGS_AVAILABLE:
    # ── Аугментация ───────────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,          # аугментация 1: поворот
        horizontal_flip=True,       # аугментация 2: отражение
        zoom_range=0.2,             # аугментация 3: масштаб
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        CATS_DOGS_DIR / "train",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary",
    )
    val_gen = val_datagen.flow_from_directory(
        CATS_DOGS_DIR / "validation",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    print(f"Классы: {train_gen.class_indices}")
    print(f"Train batches: {len(train_gen)}  Val batches: {len(val_gen)}")

    # ── Архитектура ───────────────────────────────────────────────
    section("Архитектура CNN")

    own_cnn = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),

        # Блок 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Блок 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Блок 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Классификатор
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="CatDog_CNN")

    own_cnn.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    own_cnn.summary()

    # ── Обучение ──────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SKIP_PART1 and MODEL_PATH.exists():
        section("Загрузка сохранённой CNN (SKIP_PART1=True)")
        own_cnn = keras.models.load_model(MODEL_PATH)
        print(f"  ✓ Модель загружена из {MODEL_PATH}")
        history_own = None
    else:
        section("Обучение собственной CNN")
        history_own = own_cnn.fit(
            train_gen,
            epochs=EPOCHS_OWN,
            validation_data=val_gen,
            callbacks=[early_stop, reduce_lr],
        )
        own_cnn.save(MODEL_PATH)
        print(f"  ✓ Модель сохранена в {MODEL_PATH}")

    # ── Оценка ────────────────────────────────────────────────────
    val_loss, val_acc = own_cnn.evaluate(val_gen, verbose=0)
    print(f"\nVal Accuracy: {val_acc:.4f}  Val Loss: {val_loss:.4f}")

    # ── Вопросы из задания ────────────────────────────────────────
    section("Ответы на вопросы из задания")
    print("""
1. Этапы предобработки данных:
   - rescale 1/255: нормализация пикселей [0,255] → [0,1]
   - rotation_range, horizontal_flip, zoom_range: аугментация —
     искусственное расширение датасета для борьбы с переобучением
   - target_size=(150,150): все изображения приводятся к единому размеру

2. Параметры модели:
   - Conv2D(32, (3,3)): 32 фильтра 3×3, выявляют локальные признаки (края, текстуры)
   - MaxPooling2D(2,2): уменьшает карту признаков вдвое, снижает вычисления
   - BatchNormalization: стабилизирует обучение, ускоряет сходимость
   - GlobalAveragePooling2D: заменяет Flatten, меньше параметров, меньше overfitting
   - Dropout(0.5): регуляризация — 50% нейронов случайно отключается при обучении

3. Слои и что происходит:
   Conv2D → извлечение пространственных признаков (края, углы, текстуры)
   BatchNorm → нормализация активаций внутри слоя
   MaxPooling → субдискретизация (уменьшение размерности карт признаков)
   GlobalAvgPooling → агрегация признаков всей карты в вектор
   Dense(256, relu) → нелинейная классификация в пространстве признаков
   Dropout → регуляризация
   Dense(1, sigmoid) → выходная вероятность (кот=0, собака=1)
""")

    # ── Визуализация обучения ─────────────────────────────────────
    if history_own is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history_own.history["loss"], label="Train")
        axes[0].plot(history_own.history["val_loss"], label="Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(history_own.history["accuracy"], label="Train")
        axes[1].plot(history_own.history["val_accuracy"], label="Val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        fig.suptitle("CNN — кошки/собаки: кривые обучения", fontweight="bold")
        fig.tight_layout()
        plt.show()
        plt.close()
    else:
        print("  (графики обучения недоступны — модель загружена из файла)")

else:
    print("  Пропускаем Часть 1 — датасет не загружен.")
    val_acc = None

# ══════════════════════════════════════════════════════════════════
#  ЧАСТЬ 2 — TRANSFER LEARNING (Caltech-101)
# ══════════════════════════════════════════════════════════════════

section("Часть 2 — Transfer Learning: InceptionV3 и VGG19 (Caltech-101)")

CALTECH_DIR = Path("data/caltech101")
CHOSEN_CLASSES = ["airplanes", "car_side", "Faces"]  # 3 любых класса

if not CALTECH_DIR.exists():
    print(f"""
  ⚠  Датасет Caltech-101 не найден: {CALTECH_DIR}
  
  Скачайте: https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip
  Распакуйте и создайте структуру:
    data/
      caltech101/
        train/
          airplanes/
          car_side/
          Faces/
        validation/
          airplanes/
          car_side/
          Faces/
  
  Выбранные классы: {CHOSEN_CLASSES}
""")
    CALTECH_AVAILABLE = False
else:
    CALTECH_AVAILABLE = True

if CALTECH_AVAILABLE:
    IMG_SIZE_TL = (224, 224)

    # ── Аугментация ───────────────────────────────────────────────
    tl_train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,          # аугментация 1
        horizontal_flip=True,       # аугментация 2
        zoom_range=0.3,             # аугментация 3
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode="nearest",
    )
    tl_val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    tl_train_gen = tl_train_datagen.flow_from_directory(
        CALTECH_DIR / "train",
        target_size=IMG_SIZE_TL, batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    tl_val_gen = tl_val_datagen.flow_from_directory(
        CALTECH_DIR / "validation",
        target_size=IMG_SIZE_TL, batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    N_CLASSES = len(tl_train_gen.class_indices)
    print(f"Классы: {tl_train_gen.class_indices}")

    # ── InceptionV3 Fine-Tuning ───────────────────────────────────
    section("InceptionV3 — Fine-Tuning")

    base_inception = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE_TL, 3),
    )

    # Заморозить все, разморозить последний блок
    base_inception.trainable = True
    for layer in base_inception.layers[:-20]:
        layer.trainable = False

    inp = keras.Input(shape=(*IMG_SIZE_TL, 3))
    x = base_inception(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(N_CLASSES, activation="softmax")(x)

    inception_model = keras.Model(inp, out, name="InceptionV3_FineTune")
    inception_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    inception_model.summary()

    history_inc = inception_model.fit(
        tl_train_gen, epochs=EPOCHS_FT,
        validation_data=tl_val_gen,
        callbacks=[early_stop, reduce_lr],
    )

    # ── VGG19 Fine-Tuning ─────────────────────────────────────────
    section("VGG19 — Fine-Tuning")

    base_vgg = VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE_TL, 3),
    )

    base_vgg.trainable = True
    for layer in base_vgg.layers[:-4]:
        layer.trainable = False

    inp2 = keras.Input(shape=(*IMG_SIZE_TL, 3))
    x2 = base_vgg(inp2, training=False)
    x2 = layers.GlobalAveragePooling2D()(x2)
    x2 = layers.Dense(256, activation="relu")(x2)
    x2 = layers.Dropout(0.4)(x2)
    out2 = layers.Dense(N_CLASSES, activation="softmax")(x2)

    vgg_model = keras.Model(inp2, out2, name="VGG19_FineTune")
    vgg_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    vgg_model.summary()

    history_vgg = vgg_model.fit(
        tl_train_gen, epochs=EPOCHS_FT,
        validation_data=tl_val_gen,
        callbacks=[early_stop, reduce_lr],
    )

    # ── Итоговая таблица ──────────────────────────────────────────
    section("Сравнение моделей Transfer Learning")

    inc_acc = max(history_inc.history["val_accuracy"])
    vgg_acc = max(history_vgg.history["val_accuracy"])

    df_tl = pd.DataFrame({
        "Val Accuracy (max)": [inc_acc, vgg_acc],
        "Unfrozen layers":    [20, 4],
        "Input size":         ["224×224", "224×224"],
        "Augmentations":      [3, 3],
    }, index=["InceptionV3", "VGG19"])

    print(df_tl.round(4).to_string())

    # ── Визуализация ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for hist, name, color in [
        (history_inc, "InceptionV3", "steelblue"),
        (history_vgg, "VGG19",       "coral"),
    ]:
        axes[0].plot(hist.history["val_loss"],     label=name, color=color)
        axes[1].plot(hist.history["val_accuracy"], label=name, color=color)

    axes[0].set_title("Val Loss")
    axes[0].legend()
    axes[1].set_title("Val Accuracy")
    axes[1].legend()

    fig.suptitle("Transfer Learning — Caltech-101 (3 класса)", fontweight="bold")
    fig.tight_layout()
    plt.show()
    plt.close()

else:
    print("  Пропускаем Часть 2 — датасет не загружен.")

# ══════════════════════════════════════════════════════════════════
#  ВЫВОД
# ══════════════════════════════════════════════════════════════════

section("Вывод")

print(f"""
Часть 1 — Собственная CNN (кошки/собаки):
  {'Val Accuracy: ' + str(round(val_acc, 4)) if val_acc else 'Данные не загружены.'}
  Применены 3 аугментации: поворот, отражение, масштабирование.
  Архитектура: 3 свёрточных блока + GlobalAvgPooling + классификатор.

Часть 2 — Transfer Learning (Caltech-101):
  {'InceptionV3 и VGG19 дообучены (Fine-Tuning).' if CALTECH_AVAILABLE else 'Данные не загружены.'}
  Fine-Tuning: размораживаем последние слои предобученной сети,
  добавляем новый классификатор для 3 целевых классов.
  Обе модели обучены на ImageNet — нижние слои уже умеют детектировать
  края, текстуры и формы, что ускоряет обучение.
""")

print(f"\n{'─'*65}")
print("  Лабораторная работа №7 выполнена.")
print(f"{'─'*65}\n")