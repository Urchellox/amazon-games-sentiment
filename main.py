"""main.py — консольное меню проекта: тональность отзывов Amazon (Video Games)."""
import pandas as pd

import data_loader
import database
import eda
import analysis
import preprocessing

MENU = {
    "1": "Загрузить сырые данные -> Parquet",
    "2": "Провести EDA (метки, графики)",
    "3": "Создать базу SQLite",
    "4": "Показать статистику базы",
    "5": "Сравнить stemming и lemmatization",
    "6": "Частотный анализ и n-граммы",
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

ACTIONS = {
    "1": action_load_data,
    "2": action_eda,
    "3": action_build_db,
    "4": action_show_stats,
    "5": action_preprocessing,
    "6": action_analysis,
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