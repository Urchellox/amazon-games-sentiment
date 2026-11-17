"""error_analysis.py — confusion matrix и разбор ошибок выбранной модели."""
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix

from data_loader import REPORTS_DIR
from database import DB_PATH, get_connection
from eda import FIGURES_DIR, LABEL_ORDER
from evaluation import PREDICTIONS_DIR, RESULTS_PATH
from preprocessing import ensure_nltk_data, remove_html

N_SAMPLES = 30          # сколько ошибок разбираем вручную
PER_PAIR = 5            # не больше стольких ошибок на пару (истина -> предсказание)
VADER_GAP = 0.5         # порог расхождения «звёзды против текста» по VADER

NEGATION_RE = re.compile(r"\b(?:not|no|never|nothing|cannot)\b|n't", re.I)
CONTRAST_RE = re.compile(r"\b(?:but|however|although|though|except|unfortunately|despite)\b", re.I)
SARCASM_RE = re.compile(r"yeah,? right|oh,? great|thanks a lot|great job|real nice|so much for"
                        r"|what a joke|brilliant\b|just great|good luck", re.I)


def list_models():
    """Имена моделей, для которых сохранены предсказания."""
    if not PREDICTIONS_DIR.exists():
        return []
    return sorted(p.stem for p in PREDICTIONS_DIR.glob("*.parquet"))


def best_model():
    """Лучшая по macro-F1 модель среди тех, у кого есть предсказания."""
    available = list_models()
    if not available or not RESULTS_PATH.exists():
        raise FileNotFoundError("Нет предсказаний моделей. Сначала обучите модели (пункты 8-9).")
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    scored = [(results[name]["f1_macro"], name) for name in available if name in results]
    if not scored:
        raise FileNotFoundError("Нет предсказаний моделей. Сначала обучите модели (пункты 8-9).")
    return max(scored)[1]


def load_predictions(name):
    """Предсказания модели: review_id, y_true, y_pred."""
    path = PREDICTIONS_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Нет файла предсказаний: {path}")
    return pd.read_parquet(path)


def fetch_reviews(ids, chunk=900):
    """Достаёт из SQLite рейтинг, текст и товар по списку review_id (порциями)."""
    if not DB_PATH.exists():
        raise FileNotFoundError("Нет базы данных. Сначала выполните пункт 3.")
    conn = get_connection()
    try:
        parts = []
        for i in range(0, len(ids), chunk):
            part = list(map(int, ids[i:i + chunk]))
            marks = ",".join("?" * len(part))
            parts.append(pd.read_sql_query(
                f"SELECT review_id, rating, text, parent_asin FROM reviews "
                f"WHERE review_id IN ({marks})", conn, params=part))
    finally:
        conn.close()
    return pd.concat(parts, ignore_index=True)


def plot_confusion(name, y_true, y_pred):
    """Рисунок 10: матрица ошибок (количество и доля по строкам = recall)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, fmt, title in ((axes[0], cm, "{:,.0f}", "Количество"),
                                 (axes[1], cm_norm, "{:.0%}", "Доля от истинного класса")):
        ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(3), LABEL_ORDER)
        ax.set_yticks(range(3), LABEL_ORDER)
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Истина")
        ax.set_title(f"{name}: {title}")
        for i in range(3):
            for j in range(3):
                color = "white" if data[i, j] > data.max() / 2 else "black"
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"10_confusion_{name}.png", dpi=150)
    plt.show()
    return cm


def add_tags(df):
    """Эвристические признаки текста для разбора ошибок (колонки-флаги)."""
    ensure_nltk_data()
    analyzer = SentimentIntensityAnalyzer()
    clean = df["text"].map(remove_html)
    compound = clean.map(lambda t: analyzer.polarity_scores(t)["compound"])
    df = df.copy()
    df["отрицание"] = clean.str.contains(NEGATION_RE)
    df["противопоставление"] = clean.str.contains(CONTRAST_RE)
    df["возможный сарказм"] = clean.str.contains(SARCASM_RE)
    df["звёзды против текста"] = (((df["rating"] <= 2) & (compound >= VADER_GAP))
                                 | ((df["rating"] >= 4) & (compound <= -VADER_GAP)))
    df["короткий текст (<=10 слов)"] = clean.str.count(r"\S+") <= 10
    df["clean_text"] = clean
    return df


def tag_table(df):
    """Для каждого признака: сколько отзывов и какая доля из них классифицирована неверно."""
    tags = [c for c in df.columns if df[c].dtype == bool and c != "error"]
    overall = df["error"].mean() * 100
    rows = [{"признак": "(все отзывы)", "n": len(df), "error_rate_pct": round(overall, 1)}]
    for tag in tags:
        sub = df[df[tag]]
        if len(sub):
            rows.append({"признак": tag, "n": len(sub),
                         "error_rate_pct": round(sub["error"].mean() * 100, 1)})
    return pd.DataFrame(rows)


def sample_errors(df, n=N_SAMPLES, per_pair=PER_PAIR):
    """Выборка ошибок для ручного разбора: поровну по парам (истина -> предсказание)."""
    errors = df[df["error"]]
    parts = [g.sample(min(per_pair, len(g)), random_state=42)
             for _, g in errors.groupby(["y_true", "y_pred"])]
    sample = pd.concat(parts) if parts else errors.head(0)
    if len(sample) < n:
        rest = errors.drop(sample.index)
        sample = pd.concat([sample, rest.sample(min(n - len(sample), len(rest)), random_state=42)])
    return sample.head(n).sort_values(["y_true", "y_pred"])


def run_error_analysis(name=None):
    """Матрица ошибок, таблица признаков и выборка ошибок для разбора."""
    name = name or best_model()
    print(f"Модель: {name}")
    pred = load_predictions(name)
    plot_confusion(name, pred["y_true"], pred["y_pred"])

    df = pred.merge(fetch_reviews(pred["review_id"].tolist()), on="review_id")
    df["error"] = df["y_true"] != df["y_pred"]
    print(f"Ошибок: {df['error'].sum():,} из {len(df):,} ({df['error'].mean():.1%})")
    print("\nПары (истина -> предсказание), топ ошибок:")
    print(df[df["error"]].groupby(["y_true", "y_pred"]).size()
            .sort_values(ascending=False).to_string())

    df = add_tags(df)
    table = tag_table(df)
    print("\nДоля ошибок среди отзывов с признаком:")
    print(table.to_string(index=False))

    sample = sample_errors(df)
    out = sample[["review_id", "rating", "y_true", "y_pred", "clean_text"]].copy()
    out["признаки"] = sample[[c for c in df.columns if df[c].dtype == bool
                              and c != "error"]].apply(
        lambda row: ", ".join(row.index[row]), axis=1)
    out["clean_text"] = out["clean_text"].str.slice(0, 300)
    out = out.rename(columns={"clean_text": "text"})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / f"error_tags_{name}.csv", index=False)
    out.to_csv(REPORTS_DIR / f"error_samples_{name}.csv", index=False)
    print(f"\nДля разбора: {REPORTS_DIR / f'error_samples_{name}.csv'} ({len(out)} отзывов)")
    for _, r in out.head(10).iterrows():
        print(f"  [{r['rating']}★ {r['y_true']} -> {r['y_pred']}] {r['text'][:140]}")
    return df, table, out


if __name__ == "__main__":
    try:
        run_error_analysis()
    except FileNotFoundError as e:
        print(e)
