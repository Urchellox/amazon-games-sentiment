"""eda.py — метки тональности и разведочный анализ данных (EDA)."""
import json

import matplotlib
matplotlib.use("Agg")  # рисуем сразу в файлы, без окон
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import REPORTS_DIR, REVIEWS_PARQUET

FIGURES_DIR = REPORTS_DIR / "figures"
LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_COLORS = ["#e45756", "#9d9d9d", "#54a24b"]
RATING_TO_LABEL = {1: "negative", 2: "negative", 3: "neutral",
                   4: "positive", 5: "positive"}


def add_label(df):
    """Добавляет колонку label (negative/neutral/positive) по рейтингу."""
    df["label"] = pd.Categorical(df["rating"].map(RATING_TO_LABEL),
                                 categories=LABEL_ORDER)
    return df


def add_text_length(df):
    """Добавляет число слов в отзыве."""
    df["n_words"] = df["text"].str.count(r"\S+").astype("int32")
    return df


def class_distribution(df):
    """Количество, доли классов и коэффициент дисбаланса (max / min)."""
    counts = df["label"].value_counts().reindex(LABEL_ORDER)
    return {
        "counts": {k: int(v) for k, v in counts.items()},
        "shares_pct": {k: round(float(v) / counts.sum() * 100, 1)
                       for k, v in counts.items()},
        "imbalance_ratio": round(float(counts.max() / counts.min()), 1),
    }


def length_stats(df):
    """Средняя и медианная длина (в словах) по классам и общая статистика."""
    by_label = df.groupby("label", observed=True)["n_words"].agg(["mean", "median"]).round(1)
    overall = df["n_words"].describe(percentiles=[0.5, 0.9, 0.99]).round(1)
    return by_label, overall


def plot_ratings_and_classes(df):
    """Рисунок 1: распределение звёзд и классов."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df["rating"].value_counts().sort_index().plot.bar(ax=axes[0], color="#4c78a8")
    axes[0].set_title("Распределение оценок")
    axes[0].set_xlabel("Звёзды")
    axes[0].set_ylabel("Число отзывов")
    axes[0].bar_label(axes[0].containers[0], fmt="{:,.0f}")
    df["label"].value_counts().reindex(LABEL_ORDER).plot.bar(ax=axes[1], color=LABEL_COLORS)
    axes[1].set_title("Распределение классов тональности")
    axes[1].set_xlabel("Класс")
    axes[1].set_ylabel("Число отзывов")
    axes[1].bar_label(axes[1].containers[0], fmt="{:,.0f}")
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_ratings_classes.png", dpi=150)
    plt.close(fig)


def plot_text_length(df):
    """Рисунок 2: распределение длины отзывов по классам (до 99-го перцентиля)."""
    limit = int(df["n_words"].quantile(0.99))
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in zip(LABEL_ORDER, LABEL_COLORS):
        words = df.loc[df["label"] == label, "n_words"]
        ax.hist(words[words <= limit], bins=50, alpha=0.5, density=True,
                label=label, color=color)
    ax.set_title(f"Длина отзывов по классам (до {limit} слов)")
    ax.set_xlabel("Число слов")
    ax.set_ylabel("Плотность")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_text_length.png", dpi=150)
    plt.close(fig)


def plot_reviews_per_year(df):
    """Рисунок 3: число отзывов по годам."""
    per_year = df["timestamp"].dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(per_year.index.astype(str), per_year.values, color="#4c78a8")
    ax.set_title("Число отзывов по годам")
    ax.set_xlabel("Год")
    ax.set_ylabel("Число отзывов")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_reviews_per_year.png", dpi=150)
    plt.close(fig)


def run_eda():
    """Загружает данные, считает статистику, рисует графики, сохраняет JSON."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(REVIEWS_PARQUET,
                         columns=["parent_asin", "user_id", "rating", "text",
                                  "timestamp", "verified_purchase"])
    df = add_text_length(add_label(df))
    by_label, overall = length_stats(df)
    summary = {
        "n_reviews": int(len(df)),
        "n_products": int(df["parent_asin"].nunique()),
        "n_users": int(df["user_id"].nunique()),
        "period": [str(df["timestamp"].min().date()), str(df["timestamp"].max().date())],
        "verified_share_pct": round(float(df["verified_purchase"].mean() * 100), 1),
        "html_br_share_pct": round(float(df["text"].str.contains("<br />", regex=False).mean() * 100), 1),
        "classes": class_distribution(df),
        "words_by_label": by_label.to_dict("index"),
        "words_overall": overall.to_dict(),
    }
    plot_ratings_and_classes(df)
    plot_text_length(df)
    plot_reviews_per_year(df)
    with open(REPORTS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return df, summary


if __name__ == "__main__":
    try:
        run_eda()
    except FileNotFoundError:
        print("Не найден reviews.parquet. Сначала запустите: python data_loader.py")