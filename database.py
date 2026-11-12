"""database.py — SQLite: схема, загрузка данных, индексы."""
import sqlite3
import time

import pandas as pd

from data_loader import PROCESSED_DIR, PRODUCTS_PARQUET, REVIEWS_PARQUET
from eda import add_label, add_text_length

DB_PATH = PROCESSED_DIR / "reviews.db"

SCHEMA_SQL = """
CREATE TABLE products (
    parent_asin    TEXT PRIMARY KEY,
    product_title  TEXT,
    main_category  TEXT,
    store          TEXT,
    categories     TEXT,
    average_rating REAL,
    rating_number  INTEGER,
    price          REAL
);

CREATE TABLE reviews (
    review_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_asin       TEXT NOT NULL REFERENCES products(parent_asin),
    user_id           TEXT NOT NULL,
    rating            INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    label             TEXT NOT NULL CHECK (label IN ('negative', 'neutral', 'positive')),
    review_title      TEXT,
    text              TEXT NOT NULL,
    timestamp         TEXT NOT NULL,
    helpful_vote      INTEGER,
    verified_purchase INTEGER,
    n_words           INTEGER
);

CREATE TABLE predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id       INTEGER NOT NULL REFERENCES reviews(review_id),
    model           TEXT NOT NULL,
    predicted_label TEXT NOT NULL CHECK (predicted_label IN ('negative', 'neutral', 'positive')),
    score           REAL,
    UNIQUE (review_id, model)
);
"""

INDEX_SQL = """
CREATE INDEX idx_reviews_asin  ON reviews(parent_asin);
CREATE INDEX idx_reviews_user  ON reviews(user_id);
CREATE INDEX idx_reviews_label ON reviews(label);
CREATE INDEX idx_pred_review   ON predictions(review_id);
"""


def get_connection(db_path=DB_PATH):
    """Открывает соединение и включает проверку внешних ключей."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn):
    """Создаёт таблицы."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def load_products(conn, df):
    """Загружает товары."""
    df = df.copy()
    df["main_category"] = df["main_category"].astype(str)
    df.to_sql("products", conn, if_exists="append", index=False, chunksize=50_000)
    return len(df)


def prepare_reviews(df, valid_asins):
    """Добавляет метку и длину, оставляет отзывы с известным товаром, приводит типы."""
    df = add_text_length(add_label(df))
    df["parent_asin"] = df["parent_asin"].astype(str)
    df = df[df["parent_asin"].isin(valid_asins)].copy()   # целостность внешнего ключа
    df["label"] = df["label"].astype(str)
    df["verified_purchase"] = df["verified_purchase"].astype(int)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def load_reviews(conn, df, chunksize=50_000):
    """Загружает отзывы порциями."""
    total = 0
    for start in range(0, len(df), chunksize):
        part = df.iloc[start:start + chunksize]
        part.to_sql("reviews", conn, if_exists="append", index=False)
        total += len(part)
        print(f"  загружено {total:,} / {len(df):,}")
    conn.commit()


def create_indexes(conn):
    """Создаёт индексы (после загрузки это быстрее)."""
    conn.executescript(INDEX_SQL)


def table_counts(conn):
    """Число строк в каждой таблице."""
    counts = {}
    for table in ("products", "reviews", "predictions"):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def build_database(db_path=DB_PATH):
    """Создаёт базу с нуля из Parquet-файлов."""
    t0 = time.perf_counter()
    if db_path.exists():
        db_path.unlink()
    products = pd.read_parquet(PRODUCTS_PARQUET)
    reviews = pd.read_parquet(REVIEWS_PARQUET)
    conn = get_connection(db_path)
    try:
        create_schema(conn)
        load_products(conn, products)
        prepared = prepare_reviews(reviews, set(products["parent_asin"]))
        print(f"Отзывов без товара в meta (отброшено): {len(reviews) - len(prepared):,}")
        load_reviews(conn, prepared)
        print("Создаём индексы...")
        create_indexes(conn)
        print("Строк в таблицах:", table_counts(conn))
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        print("Нарушений внешних ключей:", len(violations))
    finally:
        conn.close()
    print(f"Готово: {db_path.stat().st_size / 1e6:.0f} МБ, {time.perf_counter() - t0:.0f} с")


if __name__ == "__main__":
    try:
        build_database()
    except FileNotFoundError:
        print("Не найдены Parquet-файлы. Сначала запустите: python data_loader.py")