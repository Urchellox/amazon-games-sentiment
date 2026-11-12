"""data_loader.py — загрузка и подготовка данных Amazon Reviews 2023 (Video Games)."""
import gzip
import json
import random
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

REVIEWS_RAW = RAW_DIR / "Video_Games.jsonl.gz"
META_RAW = RAW_DIR / "meta_Video_Games.jsonl.gz"
REVIEWS_PARQUET = PROCESSED_DIR / "reviews.parquet"
PRODUCTS_PARQUET = PROCESSED_DIR / "products.parquet"

REVIEW_FIELDS = ["parent_asin", "user_id", "rating", "title", "text",
                 "timestamp", "helpful_vote", "verified_purchase"]
META_FIELDS = ["parent_asin", "title", "main_category", "store",
               "average_rating", "rating_number", "price", "categories"]


def read_jsonl_chunks(path, fields, chunksize=100_000, sample_frac=1.0, seed=42):
    """Читает .jsonl.gz построчно и отдаёт DataFrame порциями по chunksize строк.

    sample_frac < 1.0 — случайно оставляем долю строк (решение принимается
    до разбора JSON, поэтому это быстро). Берём только нужные поля.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    rng = random.Random(seed)
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if sample_frac < 1.0 and rng.random() > sample_frac:
                continue
            record = json.loads(line)
            rows.append({field: record.get(field) for field in fields})
            if len(rows) >= chunksize:
                yield pd.DataFrame(rows)
                rows = []
    if rows:
        yield pd.DataFrame(rows)


def clean_reviews(df):
    """Очистка и оптимизация типов одного чанка отзывов."""
    df = df.rename(columns={"title": "review_title"})
    df["review_title"] = df["review_title"].fillna("")
    df["text"] = df["text"].fillna("")
    df = df[df["text"].str.strip() != ""]                       # пустые тексты не нужны
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating", "parent_asin", "user_id"])
    df["rating"] = df["rating"].astype("int8")                  # 1..5
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype("int32")
    df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(bool)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def clean_products(df):
    """Очистка таблицы товаров."""
    df = df.rename(columns={"title": "product_title"})
    df["product_title"] = df["product_title"].fillna("")
    df["store"] = df["store"].fillna("")
    df["main_category"] = df["main_category"].fillna("Unknown").astype("category")
    df["categories"] = df["categories"].apply(
        lambda c: " > ".join(c) if isinstance(c, list) else "")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float32")
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce").astype("float32")
    df["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce").fillna(0).astype("int32")
    return df.drop_duplicates(subset="parent_asin").reset_index(drop=True)


def build_reviews(sample_frac=0.16, chunksize=100_000):
    """Собирает таблицу отзывов из сырого файла по чанкам."""
    parts, total = [], 0
    for i, chunk in enumerate(read_jsonl_chunks(REVIEWS_RAW, REVIEW_FIELDS,
                                                chunksize, sample_frac), start=1):
        part = clean_reviews(chunk)
        parts.append(part)
        total += len(part)
        print(f"  чанк {i}: накоплено {total:,} отзывов")
    df = pd.concat(parts, ignore_index=True)
    df["parent_asin"] = df["parent_asin"].astype("category")    # много повторов -> экономия памяти
    return df


def build_products():
    """Собирает таблицу товаров (файл небольшой, берём целиком)."""
    parts = [chunk for chunk in read_jsonl_chunks(META_RAW, META_FIELDS, 200_000)]
    return clean_products(pd.concat(parts, ignore_index=True))


def save_parquet(df, path):
    """Сохраняет DataFrame в Parquet, создавая папку при необходимости."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def compare_formats(df, sample_size=200_000):
    """Сравнивает CSV, JSONL и Parquet: размер, время записи/чтения, память."""
    out_dir = PROCESSED_DIR / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    formats = {
        "csv": ("reviews.csv",
                lambda d, p: d.to_csv(p, index=False), pd.read_csv),
        "jsonl": ("reviews.jsonl",
                  lambda d, p: d.to_json(p, orient="records", lines=True),
                  lambda p: pd.read_json(p, lines=True)),
        "parquet": ("reviews.parquet",
                    lambda d, p: d.to_parquet(p, index=False), pd.read_parquet),
    }
    results = {}
    for name, (filename, writer, reader) in formats.items():
        path = out_dir / filename
        t0 = time.perf_counter()
        writer(sample, path)
        write_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        loaded = reader(path)
        read_s = time.perf_counter() - t0
        results[name] = {
            "rows": len(loaded),
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "write_s": round(write_s, 2),
            "read_s": round(read_s, 2),
            "memory_mb": round(loaded.memory_usage(deep=True).sum() / 1e6, 1),
        }
        print(f"  {name:8s} {results[name]}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "format_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


if __name__ == "__main__":
    try:
        print("Читаем товары...")
        products = build_products()
        save_parquet(products, PRODUCTS_PARQUET)
        print(f"Товаров: {len(products):,}")

        print("Читаем отзывы (несколько минут)...")
        reviews = build_reviews()
        save_parquet(reviews, REVIEWS_PARQUET)
        print(f"Отзывов: {len(reviews):,}")
        print(reviews.dtypes)
        print(reviews.head(3).to_string())
        print(f"Память: {reviews.memory_usage(deep=True).sum() / 1e6:.0f} МБ")

        print("Сравнение форматов (CSV / JSONL / Parquet)...")
        compare_formats(reviews)
    except FileNotFoundError as e:
        print(f"Ошибка: {e}\nПоложите скачанные файлы в {RAW_DIR}")