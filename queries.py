"""queries.py — выполнение SQL-запросов из sql/queries.sql, результаты в Pandas и CSV."""
import re
import time

import pandas as pd

from data_loader import BASE_DIR, REPORTS_DIR
from database import get_connection

SQL_FILE = BASE_DIR / "sql" / "queries.sql"
RESULTS_DIR = REPORTS_DIR / "sql_results"


def load_queries(path=SQL_FILE):
    """Читает файл и возвращает словарь {имя: (заголовок, SQL)}.

    Блок запроса начинается со строки '-- name: ...', затем '-- title: ...'.
    """
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл с запросами: {path}")
    text = path.read_text(encoding="utf-8")
    queries = {}
    for block in re.split(r"^-- name:\s*", text, flags=re.MULTILINE)[1:]:
        lines = block.strip().splitlines()
        name = lines[0].strip()
        title, sql_lines = "", []
        for line in lines[1:]:
            if line.startswith("-- title:"):
                title = line[len("-- title:"):].strip()
            else:
                sql_lines.append(line)
        queries[name] = (title, "\n".join(sql_lines).strip())
    return queries


def run_one(name, queries=None, save=True):
    """Выполняет один запрос, печатает и сохраняет результат (DataFrame)."""
    queries = queries or load_queries()
    title, sql = queries[name]
    conn = get_connection()
    try:
        t0 = time.perf_counter()
        df = pd.read_sql_query(sql, conn)
        elapsed = time.perf_counter() - t0
    finally:
        conn.close()
    print(f"\n[{name}] {title}  ({elapsed:.1f} с, строк: {len(df)})")
    print(df.to_string(index=False))
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return df


def run_all(queries=None):
    """Выполняет все запросы, возвращает словарь {имя: DataFrame}."""
    queries = queries or load_queries()
    return {name: run_one(name, queries) for name in queries}


if __name__ == "__main__":
    try:
        run_all()
    except FileNotFoundError as e:
        print(e)