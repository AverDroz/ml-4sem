"""
Подготовка датасетов для Лабораторной работы №7.

Запускать ОДИН РАЗ перед lab7_cnn.py.

Ожидаемая структура исходных файлов:
  data/
    cats_dogs_raw/
      cats/   ← 1000 .jpg
      dogs/   ← 1000 .jpg
    caltech_raw/
      scissors/   ← папка из архива Caltech-101
      schooner/
      saxophone/

Скрипт создаст:
  data/cats_dogs/train/{cats,dogs}/
  data/cats_dogs/validation/{cats,dogs}/
  data/caltech101/train/{scissors,schooner,saxophone}/
  data/caltech101/validation/{scissors,schooner,saxophone}/
"""

import shutil
import random
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent / "data"


def split_dataset(src_dir: Path, dst_dir: Path,
                  val_ratio: float = 0.2) -> None:
    """Split src_dir/class_name/ → dst_dir/train/ and dst_dir/validation/"""
    for class_dir in sorted(src_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        files = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        if not files:
            print(f"  ⚠ {class_dir.name}: нет изображений")
            continue

        random.shuffle(files)
        n_val = max(1, int(len(files) * val_ratio))
        val_files   = files[:n_val]
        train_files = files[n_val:]

        for split, split_files in [("train", train_files), ("validation", val_files)]:
            out = dst_dir / split / class_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out / f.name)

        print(f"  {class_dir.name}: {len(train_files)} train / {len(val_files)} val")


# ── Кошки / Собаки ───────────────────────────────────────────────
src_cd = BASE / "cats_dogs_raw"
dst_cd = BASE / "cats_dogs"

if src_cd.exists():
    print("Подготовка датасета кошки/собаки...")
    if dst_cd.exists():
        shutil.rmtree(dst_cd)
    split_dataset(src_cd, dst_cd, val_ratio=0.2)
    print(f"  ✓ Готово → {dst_cd}\n")
else:
    print(f"⚠ Не найдено: {src_cd}")
    print("  Положите папки cats/ и dogs/ в data/cats_dogs_raw/\n")

# ── Caltech-101 ───────────────────────────────────────────────────
src_cal = BASE / "caltech_raw"
dst_cal = BASE / "caltech101"

if src_cal.exists():
    print("Подготовка датасета Caltech-101...")
    if dst_cal.exists():
        shutil.rmtree(dst_cal)
    split_dataset(src_cal, dst_cal, val_ratio=0.2)
    print(f"  ✓ Готово → {dst_cal}\n")
else:
    print(f"⚠ Не найдено: {src_cal}")
    print("  Положите папки scissors/ schooner/ saxophone/ в data/caltech_raw/\n")

print("Готово. Теперь запускайте lab7_cnn.py")