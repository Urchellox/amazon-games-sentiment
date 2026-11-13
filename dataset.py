"""dataset.py — выборка и разбиение train/test для экспериментов с моделями."""
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import PROCESSED_DIR
from database import DB_PATH, get_connection

SPLIT_PATH = PROCESSED_DIR / "split.parquet"
SAMPLE_SIZE = 200_000
TEST_SIZE = 0.2
RANDOM_STATE = 42


def build_split(sample_size=SAMPLE_SIZE, test_size=TEST_SIZE):
    """Стратифицированная выборка из SQLite, разбиение train/test, сохранение в Parquet."""
    if not DB_PATH.exists():
        raise FileNotFoundError("Нет базы данных. Сначала создайте её (python database.py).")
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT review_id, text, label FROM reviews", conn)
    finally:
        conn.close()
    sample, _ = train_test_split(df, train_size=sample_size,
                                 stratify=df["label"], random_state=RANDOM_STATE)
    train, test = train_test_split(sample, test_size=test_size,
                                   stratify=sample["label"], random_state=RANDOM_STATE)
    data = pd.concat([train.assign(split="train"), test.assign(split="test")],
                     ignore_index=True)
    data.to_parquet(SPLIT_PATH, index=False)
    return data


def load_split(rebuild=False):
    """Возвращает (train, test); при отсутствии файла строит разбиение."""
    if rebuild or not SPLIT_PATH.exists():
        data = build_split()
    else:
        data = pd.read_parquet(SPLIT_PATH)
    train = data[data["split"] == "train"].reset_index(drop=True)
    test = data[data["split"] == "test"].reset_index(drop=True)
    return train, test