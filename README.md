# amazon-games-sentiment

Анализ тональности отзывов Amazon (категория Video Games, Amazon Reviews 2023): классы negative / neutral / positive по числу звёзд (1-2 / 3 / 4-5).
Сравниваются три подхода: baseline (majority, VADER), BoW / TF-IDF + Logistic Regression, эмбеддинги MiniLM + Logistic Regression.

## Установка

Нужен Python 3.10+.

```bash
git clone <ссылка на репозиторий>
cd amazon-games-sentiment
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux / Mac:  source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch (нужен для MiniLM) подтянется вместе с `sentence-transformers`, но это версия для CPU. Для видеокарты NVIDIA сначала поставьте PyTorch с CUDA по команде с https://pytorch.org, затем `pip install -r requirements.txt`.
Проверка: `python -c "import torch; print(torch.cuda.is_available())"`.
Без GPU всё работает, только пункт 9 идёт заметно дольше (для быстрой проверки уменьшите `TRAIN_SUBSAMPLE` и `TEST_SUBSAMPLE` в `embeddings.py`).

## Данные

Скачайте два файла Amazon Reviews 2023 (McAuley Lab, UCSD) и положите в `data/raw/` под такими именами:

- `Video_Games.jsonl.gz` — отзывы;
- `meta_Video_Games.jsonl.gz` — метаданные товаров.

Файлы лежат на странице датасета: https://amazon-reviews-2023.github.io/ (раздел Video Games: «review» и «meta»).

## Запуск

```bash
python main.py
```

Пункты меню выполняйте по порядку (каждый следующий использует результаты предыдущих):

| № | Пункт | Что делает | Примерное время |
|---|---|---|---|
| 1 | Загрузить сырые данные | JSONL -> Parquet, замер форматов | несколько минут |
| 2 | EDA | метки, графики в `reports/figures/` | 1 мин |
| 3 | Создать базу SQLite | `data/processed/reviews.db` | несколько минут |
| 4 | Статистика базы | число строк, распределение классов | секунды |
| 5 | Stemming vs lemmatization | `reports/preprocessing_comparison.json` | 1-2 мин |
| 6 | Частотный анализ | n-граммы, характерные слова | 2-3 мин |
| 7 | SQL-запросы | 10 запросов из `sql/queries.sql`, CSV в `reports/sql_results/` | 1 мин |
| 8 | Baseline + BoW/TF-IDF | обучение и метрики, модели в `models/` | 10-20 мин |
| 9 | MiniLM + LR | эмбеддинги, метрики | минуты (GPU) / дольше (CPU) |
| 10 | Сводная таблица | `reports/model_comparison.csv`, график `09_model_comparison.png` | секунды |
| 11 | Предсказать текст | ввод отзыва -> класс и вероятности | секунды |
| 12 | Матрица ошибок и разбор ошибок | confusion matrix (`10_confusion_*.png`), таблица признаков, 30 ошибок для ручного разбора (`error_samples_*.csv`) | 1-2 мин |
| 13 | Предсказания в SQLite и анализ по товарам | заполняет таблицу `predictions`, отчёт по товарам с фильтром по числу отзывов (`product_sentiment_*.csv`, `11_product_sentiment.png`) | 1 мин |

Минимум для получения результатов моделей: 1 -> 3 -> 8 -> 9 -> 10; для анализа ошибок и товаров затем 12 и 13. Пункты 2, 5, 6, 7 нужны для EDA и отчёта.
Ресурсы NLTK (стоп-слова, WordNet, VADER) скачиваются автоматически при первом запуске, нужен интернет.

Отдельные модули можно запускать и напрямую: `python data_loader.py`, `python database.py`, `python models.py text`, `python embeddings.py`, `python compare.py`, `python error_analysis.py`, `python product_analysis.py`.

## Структура

- `data_loader.py` — чтение JSONL чанками, типы, Parquet, сравнение форматов
- `eda.py` — метки и EDA
- `database.py`, `sql/` — схема SQLite, загрузка, запросы, ER-диаграмма
- `queries.py` — выполнение SQL, результаты в Pandas
- `preprocessing.py`, `analysis.py` — предобработка и частотный анализ
- `dataset.py`, `evaluation.py` — выборка train/test (200 000 отзывов, 80/20), метрики, время, память
- `models.py`, `embeddings.py` — модели; `compare.py` — сводная таблица; `predict.py` — предсказание текста
- `error_analysis.py` — confusion matrix и разбор ошибок; `product_analysis.py` — предсказания в SQLite и тональность по товарам
- `main.py` — консольное меню
- `reports/` — результаты (JSON, CSV, графики); `models/` и `data/` в git не попадают