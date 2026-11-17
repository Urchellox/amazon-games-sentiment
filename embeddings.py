"""embeddings.py — метод 2: sentence-transformers (all-MiniLM-L6-v2) + Logistic Regression."""
import time

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data_loader import BASE_DIR
from dataset import RANDOM_STATE, load_split
from evaluation import PeakMemory, report_result
from preprocessing import remove_html

MODEL_NAME = "all-MiniLM-L6-v2"
MODELS_DIR = BASE_DIR / "models"
TRAIN_SUBSAMPLE = 30_000   # LR обучаем на подвыборке train (эмбеддинги дорогие)
TEST_SUBSAMPLE = None      # None = весь test, как у остальных моделей
BATCH_SIZE = 128
MAX_CHARS = 1000           # модель всё равно обрезает вход (~256 токенов)


def get_encoder():
    """Загружает sentence-transformers на GPU, если он доступен, иначе на CPU."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Нужен пакет sentence-transformers: pip install sentence-transformers")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    return SentenceTransformer(MODEL_NAME, device=device)


def encode(encoder, texts):
    """Тексты -> матрица эмбеддингов (n, 384); возвращает (матрица, секунды)."""
    cleaned = [remove_html(t)[:MAX_CHARS] for t in texts]
    t0 = time.perf_counter()
    vectors = encoder.encode(cleaned, batch_size=BATCH_SIZE, show_progress_bar=True,
                             convert_to_numpy=True)
    return vectors, time.perf_counter() - t0


def subsample(df, n):
    """Стратифицированная подвыборка n строк (или вся таблица, если n пусто/больше)."""
    if not n or n >= len(df):
        return df
    part, _ = train_test_split(df, train_size=n, stratify=df["label"],
                               random_state=RANDOM_STATE)
    return part.reset_index(drop=True)


def run_embeddings(train_size=TRAIN_SUBSAMPLE, test_size=TEST_SUBSAMPLE):
    """Эмбеддинги + LR: метрики, время, память, сохранение модели и предсказаний."""
    train, test = load_split()
    train, test = subsample(train, train_size), subsample(test, test_size)
    print(f"train: {len(train):,}, test: {len(test):,}")

    encoder = get_encoder()
    with PeakMemory() as mem:
        x_train, enc_train_s = encode(encoder, train["text"])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        t0 = time.perf_counter()
        clf.fit(x_train, train["label"])
        fit_s = time.perf_counter() - t0

        x_test, enc_test_s = encode(encoder, test["text"])
        t0 = time.perf_counter()
        preds = clf.predict(x_test)
        predict_s = enc_test_s + (time.perf_counter() - t0)
    train_s = enc_train_s + fit_s

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "minilm_lr.joblib")
    return report_result(
        "minilm_lr", test, preds, train_s, predict_s, mem.peak_mb, len(train),
        notes=f"{MODEL_NAME} + LR (balanced), обучение на подвыборке; "
              f"эмбеддинги train {enc_train_s:.0f} с, fit LR {fit_s:.1f} с")


if __name__ == "__main__":
    try:
        run_embeddings()
    except (FileNotFoundError, ImportError) as e:
        print(e)
