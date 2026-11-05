# TruthLens — Fake News Detection (Machine Learning Mini Project)

A beginner-friendly Flask + scikit-learn project. It trains a TF-IDF + Logistic Regression classifier from a labeled CSV and provides:
- Single-article prediction with probability scores and highlighted signal words
- Batch CSV prediction and downloadable results
- Prediction history stored in SQLite
- Dashboard with model metrics and class distribution
- Model training page with configurable test split
- Dataset format validation and sample data
- Clear warning: this is a text-pattern classifier, **not a fact-checking service**.

## 1. Requirements
- Windows 10/11
- Python 3.10 or newer
- VS Code

## 2. Run in VS Code
1. Extract the ZIP.
2. Open the `FakeNews_ML_Detector` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create and activate a virtual environment:

   ```powershell
   py -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.
5. Install libraries:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Put your labeled dataset CSV in `data/`. See `data/README_DATASET.md`.
7. Train the model:

   ```powershell
   python train_model.py --data data/news.csv --text-column text --label-column label
   ```
   The app accepts labels `REAL`/`FAKE`, `1`/`0`, `true`/`false`, or `reliable`/`unreliable`. By default `FAKE=1`, `REAL=0`.
8. Start the website:

   ```powershell
   python run.py
   ```
9. Open `http://127.0.0.1:5000` in your browser.

You can also train from the **Train Model** page after starting the app.

## 3. Large dataset options
A small sample is included so the project can be tested immediately. For a larger experiment, download a reputable research dataset such as **WELFake** (about 72,000 labeled articles) from its publisher/repository or a recognized dataset host. Check the dataset's license, label definitions, and source before use. Place the CSV in `data/`, then map its text and label columns in the training command.

Alternative: the widely used Kaggle Fake and Real News dataset has separate `Fake.csv` and `True.csv` files. Combine them into one CSV using `scripts/prepare_kaggle_dataset.py`. You must download the data yourself from Kaggle and comply with its terms; this project does not redistribute it.

## 4. Dataset format
Minimum columns:
- `text`: article body (or `title` if body is unavailable)
- `label`: `FAKE` or `REAL`

For WELFake-style files, use `--text-column text --label-column label`. If the column names differ, pass their actual names.

Example:
```powershell
python train_model.py --data data/WELFake_Dataset.csv --text-column text --label-column label
```

## 5. Features
- TF-IDF word and character n-grams
- Logistic Regression with balanced class weights
- Stratified train/test split
- Accuracy, precision, recall, F1 and confusion matrix
- Single and batch prediction
- SQLite prediction history
- Model metadata saved to `models/metrics.json`

## 6. Important limitations
- High test accuracy does not prove real-world truth detection. Datasets can contain publisher, topic, date, and writing-style shortcuts.
- A prediction is a model estimate based on training labels, not verification against evidence or trusted sources.
- Do not use this tool to make consequential decisions about people or publishers.
- For stronger research, use time-based and source-separated evaluation, inspect false positives/negatives, and test on fresh independently labeled articles.

## 7. Project structure
```text
FakeNews_ML_Detector/
├── app/
│   ├── __init__.py
│   ├── templates/
│   └── static/
├── data/
├── models/
├── reports/
├── scripts/
├── train_model.py
├── run.py
└── requirements.txt
```
