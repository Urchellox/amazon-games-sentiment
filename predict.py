"""predict.py — предсказание тональности введённого текста сохранённой моделью."""
import json

import joblib

from evaluation import RESULTS_PATH
from models import MODELS_DIR, build_stopwords, ensure_nltk_data, make_normalizers, texts_to_docs


def best_model_name():
    """Имя лучшей по macro-F1 сохранённой текстовой модели (BoW/TF-IDF)."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("Нет результатов моделей. Сначала обучите их (пункт 8).")
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    candidates = [(r["f1_macro"], name) for name, r in results.items()
                  if (MODELS_DIR / f"{name}.joblib").exists()
                  and name not in ("majority", "vader", "minilm_lr")]
    if not candidates:
        raise FileNotFoundError("Нет сохранённых моделей в папке models/. Сначала обучите их (пункт 8).")
    return max(candidates)[1]


def load_text_model(name=None):
    """Загружает модель и готовит нормализатор и стоп-слова."""
    name = name or best_model_name()
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Не найдена модель: {path}")
    bundle = joblib.load(path)
    ensure_nltk_data()
    norm = bundle["normalizer"]
    normalize = None if norm == "none" else make_normalizers()[norm]
    return name, bundle, build_stopwords(), normalize


def predict_text(text, loaded):
    """Возвращает (класс, {класс: вероятность}) для одного текста."""
    _, bundle, stop, normalize = loaded
    x = bundle["vectorizer"].transform(texts_to_docs([text], stop, normalize))
    clf = bundle["classifier"]
    probs = clf.predict_proba(x)[0]
    scores = {label: round(float(p), 3) for label, p in zip(clf.classes_, probs)}
    return max(scores, key=scores.get), scores


if __name__ == "__main__":
    try:
        loaded = load_text_model()
        print(f"Модель: {loaded[0]}")
        print(predict_text(input("Отзыв: "), loaded))
    except FileNotFoundError as e:
        print(e)
