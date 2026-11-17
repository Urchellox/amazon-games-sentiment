"""compare.py — сводная таблица сравнения моделей из reports/model_results.json."""
import json

import matplotlib.pyplot as plt
import pandas as pd

from data_loader import REPORTS_DIR
from eda import FIGURES_DIR
from evaluation import RESULTS_PATH

COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"


def build_table(path=RESULTS_PATH):
    """Читает JSON с результатами и возвращает DataFrame, отсортированный по macro-F1."""
    if not path.exists():
        raise FileNotFoundError("Нет reports/model_results.json. Сначала обучите модели (пункты 8-9).")
    results = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for name, r in results.items():
        rows[name] = {
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "f1_neg": r["per_class"]["negative"]["f1"],
            "f1_neu": r["per_class"]["neutral"]["f1"],
            "f1_pos": r["per_class"]["positive"]["f1"],
            "train_s": r["train_s"],
            "predict_s": r["predict_s"],
            "peak_mem_mb": r["peak_mem_mb"],
            "n_train": r["n_train"],
        }
    return pd.DataFrame(rows).T.sort_values("f1_macro", ascending=False)


def plot_table(table):
    """Рисунок 9: accuracy и macro-F1 по моделям."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ax = table[["accuracy", "f1_macro"]].plot.barh(figsize=(9, 0.6 * len(table) + 2),
                                                   color=["#4c78a8", "#f58518"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("Сравнение моделей")
    ax.set_xlabel("Значение метрики")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2)
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES_DIR / "09_model_comparison.png", dpi=150)
    plt.close(ax.figure)


def run_comparison_table():
    """Строит, печатает и сохраняет сводную таблицу и график."""
    table = build_table()
    table.to_csv(COMPARISON_CSV)
    plot_table(table)
    print(table.to_string())
    print(f"\nТаблица: {COMPARISON_CSV}\nГрафик: {FIGURES_DIR / '09_model_comparison.png'}")
    return table


if __name__ == "__main__":
    try:
        run_comparison_table()
    except FileNotFoundError as e:
        print(e)
