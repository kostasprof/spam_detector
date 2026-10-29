# SMS Spam Classifier

A small machine learning project that classifies SMS messages as **spam** or **ham** (legitimate) using TF-IDF features and two classic models: **Multinomial Naive Bayes** and **Logistic Regression**. It also explores how lowering the decision threshold of the Logistic Regression model changes the precision/recall trade-off.

## Features

- Loads and parses the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) dataset
- Dataset analysis helpers: spam/ham counts, spam percentage, average and maximum message length, and median length per class
- Stratified 80/20 train/test split (reproducible with a fixed random seed)
- TF-IDF vectorization of message text
- Trains and compares two models:
  - Multinomial Naive Bayes
  - Logistic Regression
- Reports accuracy, a full classification report, and a labelled confusion breakdown (true/false positives and negatives) for each model
- Threshold experiment: re-classifies test messages using a custom spam-probability cutoff (default `0.3`) to favour catching more spam

## Project Structure

```
project/
├── data/
│   └── SMSSpamCollection      # dataset file (tab-separated: label<TAB>message)
└── src/
    ├── main.py                # entry point: runs the full pipeline
    ├── data_cleaner.py        # loading, analysis, splitting, vectorizing
    └── model.py               # model training and evaluation
```

> `data_cleaner.py` looks for the dataset at `../data/SMSSpamCollection` relative to its own location, so the `data/` folder must sit one level above the folder containing the Python files.

| File | Responsibility |
| --- | --- |
| `main.py` | Orchestrates the pipeline and runs the threshold experiment |
| `data_cleaner.py` | `load_dataset`, `prepare_dataset`, `analyze`, `message_stats`, `find_median`, `split_dataset`, `vectorize_data` |
| `model.py` | `train_bayes_model`, `train_lr_model`, `evaluate_model` |

## Getting Started

### Prerequisites

- Python 3.8+
- [scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)

### Installation

```bash
# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install scikit-learn numpy
```

### Get the dataset

1. Download the SMS Spam Collection from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
2. Extract it and place the `SMSSpamCollection` file in the `data/` folder (see structure above).

Each line of the file should look like:

```
ham	Go until jurong point, crazy.. Available only in bugis n great world
spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005...
```

### Run

From inside the `src/` folder:

```bash
python main.py
```

## How It Works

1. **Load** – each line is split on the first tab into a `(label, message)` pair.
2. **Prepare** – messages and labels are separated into two parallel lists.
3. **Split** – 80% training / 20% testing, stratified by label so both sets keep the same spam/ham ratio (`random_state=42`).
4. **Vectorize** – a `TfidfVectorizer` is fit on the training text only, then applied to the test text, which avoids data leakage.
5. **Train** – Naive Bayes and Logistic Regression are fit on the TF-IDF training vectors.
6. **Evaluate** – both models are scored on the held-out test set.
7. **Threshold experiment** – Logistic Regression's `predict_proba` output is compared against a custom threshold instead of the default 0.5.

### Why experiment with the threshold?

In spam filtering, the two kinds of mistakes have different costs. A **false negative** lets spam through; a **false positive** sends a real message to the spam folder. Lowering the threshold below 0.5 makes the model flag a message as spam more readily, which typically raises spam **recall** at the cost of **precision**. Change `threshold` in `main.py` to explore the trade-off:

```python
threshold = 0.3   # try 0.2, 0.4, 0.5, 0.7 ...
```

## Example Output

Running the script prints, for each model, the accuracy, a classification report, and a confusion-matrix breakdown:

```
===== Multinomial Naive Bayes =====
Accuracy: ...
Classification report: ...
True negatives (ham correctly identified): ...
False positives (ham incorrectly flagged as spam): ...
False negatives (spam missed and classified as ham): ...
True positives (spam correctly identified): ...

===== Logistic Regression =====
...

===== Logistic Regression (threshold=0.3) =====
...
```

Exact numbers depend on your environment and scikit-learn version.

## Dataset

The [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) contains 5,574 English SMS messages labelled as `ham` or `spam`. It is imbalanced, with spam making up roughly 13% of messages, which is why the split is stratified and why accuracy alone is not a sufficient metric. Look at precision, recall, and the confusion matrix too.

Credit: Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the study of SMS spam filtering: new collection and results.* Proceedings of the 11th ACM Symposium on Document Engineering.

## Ideas for Future Work

- Add the dataset analysis helpers (`analyze`, `message_stats`, `find_median`) to the main pipeline output
- Tune hyperparameters (e.g. TF-IDF n-grams, `alpha` for Naive Bayes, `C` for Logistic Regression) with cross-validation
- Add a precision-recall curve to pick a threshold systematically
- Save the trained model and vectorizer (e.g. with `joblib`) and add a script to classify new messages
- Add unit tests for the data-loading and splitting functions

## License

Add your preferred license here (e.g. MIT).
