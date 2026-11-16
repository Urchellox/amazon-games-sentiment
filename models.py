"""models.py — baseline (majority, VADER) и классические модели (BoW/TF-IDF + LR)."""
import sys
import time

import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data_loader import BASE_DIR
from dataset import load_split
from evaluation import PeakMemory, report_result
from preprocessing import (build_stopwords, ensure_nltk_data, make_normalizers,
                           preprocess, remove_html)

MODELS_DIR = BASE_DIR / "models"

# (имя, тип представления, n-граммы, нормализация)
EXPERIMENTS = [
    ("bow_uni_none", "bow", (1, 1), "none"),
    ("tfidf_uni_none", "tfidf", (1, 1), "none"),
    ("tfidf_bi_none", "tfidf", (1, 2), "none"),
    ("tfidf_bi_snowball", "tfidf", (1, 2), "snowball"),
    ("tfidf_bi_lemma", "tfidf", (1, 2), "lemma"),
]


# ---------- baseline ----------

def vader_label(compound):
    """Стандартные пороги VADER: >= 0.05 positive, <= -0.05 negative, иначе neutral."""
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


def run_majority(train, test):
    """Baseline 1: всегда предсказываем самый частый класс обучающей выборки."""
    majority = train["label"].value_counts().idxmax()
    preds = [majority] * len(test)
    return report_result("majority", test, preds, 0.0, 0.0, 0.0, len(train),
                         notes=f"всегда '{majority}'")


def run_vader(train, test):
    """Baseline 2: лексикон VADER (без обучения)."""
    ensure_nltk_data()
    analyzer = SentimentIntensityAnalyzer()
    texts = [remove_html(t) for t in test["text"]]
    with PeakMemory() as mem:
        t0 = time.perf_counter()
        preds = [vader_label(analyzer.polarity_scores(t)["compound"]) for t in texts]
        predict_s = time.perf_counter() - t0
    return report_result("vader", test, preds, 0.0, predict_s, mem.peak_mb, len(train),
                         notes="лексикон, порог ±0.05")


def run_baselines():
    """Запускает оба baseline на общей выборке."""
    train, test = load_split()
    print(f"train: {len(train):,}, test: {len(test):,}")
    print("Доли классов в test:", test["label"].value_counts(normalize=True).round(3).to_dict())
    run_majority(train, test)
    run_vader(train, test)


# ---------- BoW / TF-IDF + LogisticRegression ----------

def make_vectorizer(kind, ngram_range):
    """BoW (счётчики) или TF-IDF. Токены уже готовы, делим только по пробелам."""
    common = dict(token_pattern=r"\S+", lowercase=False, min_df=3,
                  max_features=300_000, ngram_range=ngram_range)
    if kind == "bow":
        return CountVectorizer(**common)
    return TfidfVectorizer(sublinear_tf=True, **common)


def texts_to_docs(texts, stop, normalize):
    """Тексты -> строки из токенов, разделённых пробелом."""
    return [" ".join(preprocess(text, stop, normalize)) for text in texts]


def save_model(name, vectorizer, classifier, norm_name):
    """Сохраняет обученную модель для демо и повторного использования."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier,
                 "normalizer": norm_name}, MODELS_DIR / f"{name}.joblib")


def run_text_model(name, train, test, kind, ngram_range, norm_name):
    """Обучает и оценивает одну модель; время и память — от сырого текста до предсказания."""
    print(f"\nОбучаем {name}...")
    ensure_nltk_data()
    stop = build_stopwords()
    normalize = None if norm_name == "none" else make_normalizers()[norm_name]
    vectorizer = make_vectorizer(kind, ngram_range)
    classifier = LogisticRegression(max_iter=300, class_weight="balanced")

    with PeakMemory() as mem:
        t0 = time.perf_counter()
        x_train = vectorizer.fit_transform(texts_to_docs(train["text"], stop, normalize))
        classifier.fit(x_train, train["label"])
        train_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_test = vectorizer.transform(texts_to_docs(test["text"], stop, normalize))
        preds = classifier.predict(x_test)
        predict_s = time.perf_counter() - t0

    save_model(name, vectorizer, classifier, norm_name)
    notes = (f"{kind}, n-граммы {ngram_range}, нормализация: {norm_name}, "
             f"признаков: {x_train.shape[1]:,}")
    return report_result(name, test, preds, train_s, predict_s, mem.peak_mb,
                         len(train), notes=notes)


def run_text_models(only=None):
    """Запускает все эксперименты BoW/TF-IDF (или только перечисленные по имени)."""
    train, test = load_split()
    for name, kind, ngram_range, norm_name in EXPERIMENTS:
        if only and name not in only:
            continue
        run_text_model(name, train, test, kind, ngram_range, norm_name)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    try:
        if mode in ("baselines", "all"):
            run_baselines()
        if mode in ("text", "all"):
            run_text_models()
    except FileNotFoundError as e:
        print(e)