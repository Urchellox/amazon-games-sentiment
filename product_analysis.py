"""product_analysis.py — запись предсказаний в SQLite и тональность по товарам."""
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import REPORTS_DIR
from database import DB_PATH, get_connection, table_counts
from eda import FIGURES_DIR
from error_analysis import best_model, list_models, load_predictions

PRODUCT_SQL = """
SELECT p.parent_asin,
       p.product_title,
       p.main_category,
       COUNT(*) AS n_reviews,
       ROUND(AVG(r.rating), 2) AS avg_rating,
       ROUND(100.0 * SUM(pr.predicted_label = 'positive') / COUNT(*), 1) AS pred_pos_pct,
       ROUND(100.0 * SUM(pr.predicted_label = 'negative') / COUNT(*), 1) AS pred_neg_pct,
       ROUND(100.0 * SUM(r.label = 'positive') / COUNT(*), 1) AS true_pos_pct,
       ROUND(100.0 * SUM(pr.predicted_label = r.label) / COUNT(*), 1) AS accuracy_pct
FROM predictions pr
JOIN reviews r  ON r.review_id = pr.review_id
JOIN products p ON p.parent_asin = r.parent_asin
WHERE pr.model = ?
GROUP BY p.parent_asin
HAVING COUNT(*) >= ?
ORDER BY pred_pos_pct DESC, n_reviews DESC
"""


def save_predictions_to_db(models=None):
    """Записывает предсказания всех моделей в таблицу predictions (повторный запуск обновляет)."""
    if not DB_PATH.exists():
        raise FileNotFoundError("Нет базы данных. Сначала выполните пункт 3.")
    models = models or list_models()
    if not models:
        raise FileNotFoundError("Нет предсказаний моделей. Сначала обучите модели (пункты 8-9).")
    conn = get_connection()
    try:
        for name in models:
            pred = load_predictions(name)
            rows = [(int(rid), name, label, None)
                    for rid, label in zip(pred["review_id"], pred["y_pred"])]
            conn.executemany(
                "INSERT OR REPLACE INTO predictions (review_id, model, predicted_label, score) "
                "VALUES (?, ?, ?, ?)", rows)
            conn.commit()
            print(f"  {name}: записано {len(rows):,} предсказаний")
        print("Строк в таблицах:", table_counts(conn))
    finally:
        conn.close()


def product_sentiment(model, min_reviews=10):
    """Тональность по товарам: только товары, у которых не меньше min_reviews отзывов с предсказанием."""
    conn = get_connection()
    try:
        return pd.read_sql_query(PRODUCT_SQL, conn, params=(model, min_reviews))
    finally:
        conn.close()


def plot_products(df, model, k=10):
    """Рисунок 11: товары с лучшей и худшей предсказанной тональностью."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = df.assign(net=df["pred_pos_pct"] - df["pred_neg_pct"])
    ranked = df.sort_values("net", ascending=False)
    top = pd.concat([ranked.head(k), ranked.tail(k)]).drop_duplicates("parent_asin")
    labels = [f"{t[:38]} ({n})" for t, n in zip(top["product_title"], top["n_reviews"])]
    colors = ["#54a24b" if v >= 0 else "#e45756" for v in top["net"]]
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(top) + 1.5))
    ax.barh(labels, top["net"], color=colors)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% положительных − % отрицательных (предсказание)")
    ax.set_title(f"Тональность по товарам, модель {model} (в скобках число отзывов)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_product_sentiment.png", dpi=150)
    plt.show()


def run_product_analysis(model=None, min_reviews=10):
    """Предсказания -> SQLite, затем отчёт по товарам для выбранной модели."""
    model = model or best_model()
    print("Записываем предсказания в SQLite...")
    save_predictions_to_db()
    df = product_sentiment(model, min_reviews)
    print(f"\nМодель {model}: товаров с >= {min_reviews} отзывами в тесте: {len(df):,}")
    if df.empty:
        print("Таких товаров нет. Уменьшите порог минимального числа отзывов.")
        return df
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS_DIR / f"product_sentiment_{model}.csv", index=False)
    cols = ["product_title", "n_reviews", "avg_rating", "pred_pos_pct", "pred_neg_pct", "true_pos_pct"]
    show = df.assign(product_title=df["product_title"].str.slice(0, 45))[cols]
    print("\nСамые «положительные» товары:")
    print(show.head(10).to_string(index=False))
    print("\nСамые «отрицательные» товары:")
    print(show.sort_values("pred_pos_pct").head(10).to_string(index=False))
    print(f"\nКорреляция предсказанной и истинной доли позитива по товарам: "
          f"{df['pred_pos_pct'].corr(df['true_pos_pct']):.2f}")
    plot_products(df, model)
    return df


if __name__ == "__main__":
    try:
        run_product_analysis()
    except FileNotFoundError as e:
        print(e)
