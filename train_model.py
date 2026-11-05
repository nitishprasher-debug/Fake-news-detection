import argparse, json, re
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def normalize_label(value):
    s = str(value).strip().lower()
    if s in {"fake", "false", "0", "unreliable", "fabricated"}: return "FAKE"
    if s in {"real", "true", "1", "reliable", "genuine"}: return "REAL"
    raise ValueError(f"Unknown label {value!r}; use FAKE/REAL or 0/1.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample_news.csv")
    p.add_argument("--text-column", default="text")
    p.add_argument("--label-column", default="label")
    p.add_argument("--test-size", type=float, default=0.2)
    args = p.parse_args()
    path = Path(args.data)
    if not path.is_absolute(): path = ROOT / path
    if not path.exists(): raise SystemExit(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    for col in (args.text_column, args.label_column):
        if col not in df.columns: raise SystemExit(f"Column '{col}' not found. Available: {list(df.columns)}")
    data = df[[args.text_column, args.label_column]].copy()
    data.columns = ["text", "label"]
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data = data[data["text"].str.len() >= 30]
    data["label"] = data["label"].apply(normalize_label)
    data = data.drop_duplicates(subset=["text"])
    if data["label"].nunique() != 2: raise SystemExit("Need examples from both FAKE and REAL classes.")
    if data["label"].value_counts().min() < 2: raise SystemExit("Need at least 2 rows per class.")
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=args.test_size,
        random_state=42, stratify=data["label"]
    )
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), max_features=150000,
                                 stop_words="english", sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5),
                                 max_features=100000, min_df=2, sublinear_tf=True))
    ])
    model = Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced",
                                           C=2.0, solver="liblinear"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, labels=["FAKE","REAL"], average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, pred, labels=["FAKE","REAL"]).tolist()
    metrics = {
        "rows_used": int(len(data)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision_macro": round(float(precision), 4), "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4), "labels": ["FAKE","REAL"], "confusion_matrix": cm,
        "classification_report": classification_report(y_test, pred, zero_division=0),
        "dataset": path.name
    }
    joblib.dump(model, MODEL_DIR / "fake_news_model.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("\nSaved model to models/fake_news_model.joblib")

if __name__ == "__main__":
    main()
