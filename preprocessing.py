"""preprocessing.py — очистка текста, токенизация, стоп-слова, stemming vs lemmatization."""
import html
import json
import re
import time
from functools import lru_cache

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

from data_loader import REPORTS_DIR, REVIEWS_PARQUET

NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet.zip",
    "omw-1.4": "corpora/omw-1.4.zip",
    "vader_lexicon": "sentiment/vader_lexicon.zip",
}
TAG_RE = re.compile(r"<[^>]+>")
POSSESSIVE_RE = re.compile(r"'s\b")
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
EXAMPLE_WORDS = ["playing", "played", "games", "gaming", "better", "was",
                 "running", "graphics", "amazing", "disappointed", "buggy",
                 "controller", "controllers", "loved", "worst", "unplayable"]


def ensure_nltk_data():
    """Скачивает ресурсы NLTK, если их ещё нет."""
    for name, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Скачиваем ресурс NLTK: {name}")
            nltk.download(name, quiet=True)


def build_stopwords():
    """Стоп-слова NLTK без отрицаний (not, no, don't...): для тональности они важны."""
    stop = set(stopwords.words("english"))
    keep = {w for w in stop if "n't" in w or w in {"no", "nor", "not"}}
    return stop - keep

def remove_html(text):
    """Убирает HTML-теги и сущности (регистр и пунктуацию не трогает)."""
    return html.unescape(TAG_RE.sub(" ", text))

def clean_text(text):
    """Убирает HTML, нормализует апострофы, переводит в нижний регистр."""
    text = remove_html(text).replace("’", "'").lower()
    return POSSESSIVE_RE.sub("", text)    # game's -> game


def tokenize(text):
    """Очистка + разбиение на токены."""
    return TOKEN_RE.findall(clean_text(text))


def remove_stopwords(tokens, stop):
    """Убирает стоп-слова и односимвольные токены."""
    return [t for t in tokens if t not in stop and len(t) > 1]


def make_normalizers():
    """Четыре варианта нормализации слов (с кэшем: слова повторяются очень часто)."""
    porter = PorterStemmer()
    snowball = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    def lemma(word):
        # Без POS-тегов лемматизатор считает слово существительным,
        # поэтому сначала пробуем как глагол, потом как существительное.
        return lemmatizer.lemmatize(lemmatizer.lemmatize(word, "v"), "n")

    funcs = {
        "none": lambda w: w,
        "porter": porter.stem,
        "snowball": snowball.stem,
        "lemma": lemma,
    }
    return {name: lru_cache(maxsize=None)(fn) for name, fn in funcs.items()}


def preprocess(text, stop, normalize=None):
    """Полный пайплайн: токены -> без стоп-слов -> (нормализация)."""
    tokens = remove_stopwords(tokenize(text), stop)
    if normalize is not None:
        tokens = [normalize(t) for t in tokens]
    return tokens


def compare_normalizers(texts, stop):
    """Сравнивает варианты нормализации: размер словаря и время."""
    raw = [tokenize(t) for t in texts]
    filtered = [remove_stopwords(tokens, stop) for tokens in raw]
    all_tokens = [tok for tokens in filtered for tok in tokens]

    methods = {}
    for name, fn in make_normalizers().items():
        t0 = time.perf_counter()
        normalized = [fn(tok) for tok in all_tokens]
        elapsed = time.perf_counter() - t0
        methods[name] = {"vocab_size": len(set(normalized)), "time_s": round(elapsed, 2)}

    base = methods["none"]["vocab_size"]
    for stats in methods.values():
        stats["vocab_reduction_pct"] = round(100 * (1 - stats["vocab_size"] / base), 1)

    return {
        "n_reviews": len(texts),
        "tokens_total": sum(len(t) for t in raw),
        "tokens_after_stopwords": len(all_tokens),
        "methods": methods,
    }


def show_examples(words):
    """Таблица: как каждый метод преобразует примеры слов."""
    normalizers = make_normalizers()
    rows = []
    for word in words:
        row = {"word": word}
        for name, fn in normalizers.items():
            if name != "none":
                row[name] = fn(word)
        rows.append(row)
    return pd.DataFrame(rows)


def run_comparison(sample_size=30_000):
    """Сравнение stemming и lemmatization на выборке отзывов, сохранение результатов."""
    ensure_nltk_data()
    texts = pd.read_parquet(REVIEWS_PARQUET, columns=["text"])["text"]
    texts = texts.sample(min(sample_size, len(texts)), random_state=42)
    stop = build_stopwords()

    results = compare_normalizers(texts, stop)
    examples = show_examples(EXAMPLE_WORDS)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "preprocessing_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    examples.to_csv(REPORTS_DIR / "normalization_examples.csv", index=False)

    print(f"Отзывов: {results['n_reviews']:,}")
    print(f"Токенов всего: {results['tokens_total']:,}")
    print(f"После удаления стоп-слов: {results['tokens_after_stopwords']:,}")
    print(pd.DataFrame(results["methods"]).T.to_string())
    print()
    print(examples.to_string(index=False))
    return results, examples


if __name__ == "__main__":
    try:
        run_comparison()
    except FileNotFoundError:
        print("Не найден reviews.parquet. Сначала запустите: python data_loader.py")