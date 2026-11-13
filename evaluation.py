"""evaluation.py — метрики, замер времени и памяти, сохранение результатов."""
import json
import threading
import time

import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from data_loader import PROCESSED_DIR, REPORTS_DIR

LABELS = ["negative", "neutral", "positive"]
RESULTS_PATH = REPORTS_DIR / "model_results.json"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"


class PeakMemory:
    """Контекстный менеджер: прирост пиковой памяти процесса (RSS) за время блока, МБ."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak_mb = 0.0

    def __enter__(self):
        self._process = psutil.Process()
        self._start = self._process.memory_info().rss
        self._peak = self._start
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def _watch(self):
        while not self._stop.is_set():
            self._peak = max(self._peak, self._process.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._peak = max(self._peak, self._process.memory_info().rss)
        self.peak_mb = (self._peak - self._start) / 1e6
        return False


def compute_metrics(y_true, y_pred):
    """Accuracy, macro-F1 и precision/recall/F1 по классам."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=LABELS,
                                         average="macro", zero_division=0)), 4),
        "per_class": {
            label: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                    "f1": round(float(f), 4)}
            for label, p, r, f in zip(LABELS, precision, recall, f1)
        },
    }


def record_result(name, entry, path=RESULTS_PATH):
    """Добавляет/обновляет результат эксперимента в JSON-файле."""
    results = {}
    if path.exists():
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("Файл результатов повреждён, создаём заново.")
    results[name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def save_predictions(name, test, y_pred):
    """Сохраняет предсказания (для confusion matrix, error analysis и SQLite)."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"review_id": test["review_id"].values,
                        "y_true": test["label"].values,
                        "y_pred": list(y_pred)})
    out.to_parquet(PREDICTIONS_DIR / f"{name}.parquet", index=False)


def report_result(name, test, y_pred, train_s, predict_s, peak_mb, n_train, notes=""):
    """Считает метрики, сохраняет результат и предсказания, печатает итог."""
    metrics = compute_metrics(test["label"], y_pred)
    entry = {**metrics,
             "train_s": round(train_s, 2),
             "predict_s": round(predict_s, 2),
             "peak_mem_mb": round(peak_mb),
             "n_train": n_train,
             "n_test": len(test),
             "notes": notes}
    record_result(name, entry)
    save_predictions(name, test, y_pred)
    f1 = metrics["per_class"]
    print(f"[{name}] accuracy={metrics['accuracy']} f1_macro={metrics['f1_macro']} | "
          f"F1: neg={f1['negative']['f1']} neu={f1['neutral']['f1']} pos={f1['positive']['f1']} | "
          f"обучение {train_s:.1f} с, предсказание {predict_s:.1f} с, память +{peak_mb:.0f} МБ")
    return entry