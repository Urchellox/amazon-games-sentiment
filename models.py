"""models.py — baseline (majority, VADER) и модели тональности."""
import time

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from dataset import load_split
from evaluation import PeakMemory, report_result
from preprocessing import ensure_nltk_data, remove_html


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


if __name__ == "__main__":
    try:
        run_baselines()
    except FileNotFoundError as e:
        print(e)