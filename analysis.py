"""analysis.py — частотный анализ, n-граммы и характерные слова по классам."""
import math
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import REPORTS_DIR, REVIEWS_PARQUET
from eda import FIGURES_DIR, LABEL_COLORS, LABEL_ORDER, add_label
from preprocessing import build_stopwords, ensure_nltk_data, preprocess


def load_balanced_sample(n_per_class=30_000):
    """Выборка с одинаковым числом отзывов каждого класса."""
    df = pd.read_parquet(REVIEWS_PARQUET, columns=["rating", "text"])
    df = add_label(df)
    return (df.groupby("label", observed=True)
              .sample(n=n_per_class, random_state=42)
              .reset_index(drop=True))


def tokenize_corpus(df, stop):
    """Словарь: класс -> список токенизированных отзывов."""
    corpus = {}
    for label in LABEL_ORDER:
        texts = df.loc[df["label"] == label, "text"]
        corpus[label] = [preprocess(text, stop) for text in texts]
    return corpus


def count_ngrams(token_lists, n):
    """Считает частоты n-грамм (n=1 — отдельные слова)."""
    counter = Counter()
    for tokens in token_lists:
        if n == 1:
            counter.update(tokens)
        else:
            counter.update(zip(*(tokens[i:] for i in range(n))))
    return counter


def top_ngrams(corpus, n, k=15):
    """Таблица топ-k n-грамм для каждого класса."""
    rows = []
    for label in LABEL_ORDER:
        counter = count_ngrams(corpus[label], n)
        for rank, (gram, count) in enumerate(counter.most_common(k), start=1):
            text = " ".join(gram) if isinstance(gram, tuple) else gram
            rows.append({"label": label, "n": n, "rank": rank,
                         "ngram": text, "count": count})
    return pd.DataFrame(rows)


def distinctive_words(corpus, k=15, min_count=100):
    """Слова, характерные для класса: log((доля в классе) / (доля в остальных))."""
    counters = {label: count_ngrams(corpus[label], 1) for label in LABEL_ORDER}
    totals = {label: sum(c.values()) for label, c in counters.items()}
    rows = []
    for label in LABEL_ORDER:
        others = [other for other in LABEL_ORDER if other != label]
        rest = Counter()
        for other in others:
            rest.update(counters[other])
        n_rest = sum(totals[other] for other in others)

        scored = []
        for word, count in counters[label].items():
            if count + rest[word] < min_count:      # редкие слова даёт шум
                continue
            score = (math.log((count + 1) / totals[label])
                     - math.log((rest[word] + 1) / n_rest))
            scored.append((score, word, count))
        scored.sort(reverse=True)
        for rank, (score, word, count) in enumerate(scored[:k], start=1):
            rows.append({"label": label, "rank": rank, "word": word,
                         "count": count, "log_ratio": round(score, 2)})
    return pd.DataFrame(rows)


def plot_class_bars(table, label_col, value_col, title, filename):
    """Три горизонтальных barplot (по одному на класс)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, label, color in zip(axes, LABEL_ORDER, LABEL_COLORS):
        part = table[table["label"] == label].sort_values("rank", ascending=False)
        ax.barh(part[label_col], part[value_col], color=color)
        ax.set_title(label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def run_analysis(n_per_class=30_000):
    """Полный частотный анализ: таблицы, графики, сохранение в reports/."""
    ensure_nltk_data()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stop = build_stopwords()

    print(f"Выборка: по {n_per_class:,} отзывов каждого класса")
    corpus = tokenize_corpus(load_balanced_sample(n_per_class), stop)

    tables = []
    for n, name in ((1, "униграммы"), (2, "биграммы"), (3, "триграммы")):
        table = top_ngrams(corpus, n)
        tables.append(table)
        plot_class_bars(table, "ngram", "count", f"Топ-15: {name} по классам",
                        f"{4 + n:02d}_ngrams_{n}.png")
        print(f"\n--- {name} ---")
        print(table.pivot(index="rank", columns="label", values="ngram")[LABEL_ORDER].to_string())

    distinct = distinctive_words(corpus)
    plot_class_bars(distinct, "word", "log_ratio", "Характерные слова классов",
                    "08_distinctive_words.png")
    print("\n--- характерные слова ---")
    print(distinct.pivot(index="rank", columns="label", values="word")[LABEL_ORDER].to_string())

    pd.concat(tables).to_csv(REPORTS_DIR / "top_ngrams.csv", index=False)
    distinct.to_csv(REPORTS_DIR / "distinctive_words.csv", index=False)
    print(f"\nГрафики: {FIGURES_DIR}")


if __name__ == "__main__":
    try:
        run_analysis()
    except FileNotFoundError:
        print("Не найден reviews.parquet. Сначала запустите: python data_loader.py")