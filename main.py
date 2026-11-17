"""main.py — консольное меню проекта: тональность отзывов Amazon (Video Games)."""
import pandas as pd

import data_loader
import database
import eda
import analysis
import compare
import embeddings
import models
import predict
import preprocessing
import queries

MENU = {
    "1": "Загрузить сырые данные -> Parquet",
    "2": "Провести EDA (метки, графики)",
    "3": "Создать базу SQLite",
    "4": "Показать статистику базы",
    "5": "Сравнить stemming и lemmatization",
    "6": "Частотный анализ и n-граммы",
    "7": "SQL-запросы",
    "8": "Обучить baseline и BoW/TF-IDF модели",
    "9": "Обучить MiniLM (sentence-transformers) + LR",
    "10": "Сводная таблица сравнения моделей",
    "11": "Предсказать тональность введённого текста",
    "0": "Выход",
}


def print_menu():
    """Печатает меню."""
    print("\n=== Меню ===")
    for key, title in MENU.items():
        print(f"  {key} - {title}")


def read_choice(valid):
    """Запрашивает пункт меню, пока не введён корректный."""
    while True:
        try:
            choice = input("Выберите пункт: ").strip()
            if choice not in valid:
                raise ValueError(f"нет пункта '{choice}'")
            return choice
        except ValueError as e:
            print(f"Ошибка: {e}. Попробуйте ещё раз.")


def confirm(question):
    """Спрашивает подтверждение (y/n)."""
    while True:
        answer = input(f"{question} (y/n): ").strip().lower()
        if answer in ("y", "д"):
            return True
        elif answer in ("n", "н"):
            return False
        print("Введите y или n.")


def action_load_data():
    """Пункт 1: сырые данные -> Parquet."""
    if not confirm("Это займёт несколько минут. Продолжить?"):
        print("Отменено.")
        return
    try:
        data_loader.run()
    except FileNotFoundError as e:
        print(f"Ошибка: {e}\nПоложите скачанные файлы в {data_loader.RAW_DIR}")


def action_eda():
    """Пункт 2: EDA."""
    try:
        eda.run_eda()
        print(f"Графики сохранены в {eda.FIGURES_DIR}")
    except FileNotFoundError:
        print("Не найден reviews.parquet. Сначала выберите пункт 1.")


def action_build_db():
    """Пункт 3: создание базы SQLite."""
    if not confirm("База будет создана заново. Продолжить?"):
        print("Отменено.")
        return
    try:
        database.build_database()
    except FileNotFoundError:
        print("Не найдены Parquet-файлы. Сначала выберите пункт 1.")


def action_show_stats():
    """Пункт 4: статистика базы."""
    if not database.DB_PATH.exists():
        print("Базы данных нет. Сначала выберите пункт 3.")
        return
    conn = database.get_connection()
    try:
        print("Строк в таблицах:", database.table_counts(conn))
        query = """
            SELECT label,
                   COUNT(*) AS n_reviews,
                   ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM reviews), 1) AS share_pct
            FROM reviews
            GROUP BY label
            ORDER BY n_reviews DESC
        """
        print(pd.read_sql_query(query, conn).to_string(index=False))
    finally:
        conn.close()

def action_preprocessing():
    """Пункт 5: сравнение stemming и lemmatization."""
    try:
        preprocessing.run_comparison()
    except FileNotFoundError:
        print("Не найден reviews.parquet. Сначала выберите пункт 1.")


def action_analysis():
    """Пункт 6: частотный анализ и n-граммы."""
    try:
        analysis.run_analysis()
    except FileNotFoundError:
        print("Не найден reviews.parquet. Сначала выберите пункт 1.")

def action_sql():
    """Пункт 7: SQL-запросы."""
    if not database.DB_PATH.exists():
        print("Базы данных нет. Сначала выберите пункт 3.")
        return
    try:
        all_queries = queries.load_queries()
    except FileNotFoundError as e:
        print(e)
        return
    names = list(all_queries)
    print("\nДоступные запросы:")
    for i, name in enumerate(names, start=1):
        print(f"  {i} - {all_queries[name][0]}")
    while True:
        choice = input("Номер запроса (a - все, 0 - назад): ").strip().lower()
        if choice == "0":
            return
        elif choice == "a":
            queries.run_all(all_queries)
            return
        try:
            index = int(choice)
            if not 1 <= index <= len(names):
                raise ValueError
        except ValueError:
            print("Введите номер из списка, 'a' или 0.")
            continue
        queries.run_one(names[index - 1], all_queries)

def action_text_models():
    """Пункт 8: baseline (majority, VADER) и BoW/TF-IDF + LR."""
    if not confirm("Обучение 5 моделей займёт около 10-20 минут. Продолжить?"):
        print("Отменено.")
        return
    try:
        models.run_baselines()
        models.run_text_models()
    except FileNotFoundError as e:
        print(f"Ошибка: {e}\nСначала выполните пункты 1 и 3.")


def action_embeddings():
    """Пункт 9: эмбеддинги MiniLM + логистическая регрессия."""
    if not confirm("Кодирование текстов займёт несколько минут (на CPU дольше). Продолжить?"):
        print("Отменено.")
        return
    try:
        embeddings.run_embeddings()
    except FileNotFoundError as e:
        print(f"Ошибка: {e}\nСначала выполните пункты 1 и 3.")
    except ImportError as e:
        print(e)


def action_compare():
    """Пункт 10: сводная таблица моделей."""
    try:
        compare.run_comparison_table()
    except FileNotFoundError as e:
        print(e)


def action_predict():
    """Пункт 11: ввод текста -> предсказание."""
    try:
        loaded = predict.load_text_model()
    except FileNotFoundError as e:
        print(e)
        return
    print(f"Модель: {loaded[0]}. Пустая строка - выход в меню.")
    while True:
        text = input("Отзыв (по-английски): ").strip()
        if not text:
            return
        label, scores = predict.predict_text(text, loaded)
        print(f"  -> {label}  {scores}")


ACTIONS = {
    "1": action_load_data,
    "2": action_eda,
    "3": action_build_db,
    "4": action_show_stats,
    "5": action_preprocessing,
    "6": action_analysis,
    "7": action_sql,
    "8": action_text_models,
    "9": action_embeddings,
    "10": action_compare,
    "11": action_predict,
}


def main():
    """Главный цикл меню."""
    print("=== Анализ тональности отзывов Amazon (Video Games) ===")
    try:
        while True:
            print_menu()
            choice = read_choice(MENU)
            if choice == "0":
                print("До свидания!")
                break
            ACTIONS[choice]()
    except (KeyboardInterrupt, EOFError):
        print("\nЗавершение работы.")


if __name__ == "__main__":
    main()