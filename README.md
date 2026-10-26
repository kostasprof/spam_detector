# SMS Spam Detector

A machine learning project that classifies SMS messages as **spam** or **ham** (legitimate messages). The project compares two classification algorithms using TF-IDF text features: Multinomial Naive Bayes and Logistic Regression.

## Project Goals

- Understand and preprocess a real-world SMS dataset.
- Convert text messages into numerical features using TF-IDF.
- Train and compare two machine learning classification models.
- Evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Investigate ways to improve spam detection.
- Eventually build an interface that allows users to enter a message and receive a prediction.

## Dataset

The project uses the `SMSSpamCollection` dataset, containing 5,574 SMS messages labelled as either `ham` or `spam`.

| Class | Number of messages | Percentage |
|---|---:|---:|
| Ham | 4,827 | 86.6% |
| Spam | 747 | 13.4% |
| **Total** | **5,574** | **100%** |

The dataset is imbalanced, with legitimate messages making up the majority of the data. Therefore, accuracy alone is not sufficient to evaluate the models. Spam recall and the number of missed spam messages are also important.

## Project Structure

```text
spam_detector/
├── data/
│   └── SMSSpamCollection
├── src/
│   ├── data_cleaner.py
│   ├── model.py
│   └── main.py
└── README.md
```

## Data Preprocessing

The preprocessing pipeline currently performs the following steps:

1. **Load the dataset:** Read each message and its corresponding label from the tab-separated dataset.
2. **Separate features and labels:** Store SMS message text separately from its `spam` or `ham` label.
3. **Split the data:** Divide the dataset into training and testing sets using an 80/20 split, with stratification to preserve the class proportions.
4. **Convert text into numerical features:** Use `TfidfVectorizer` to represent messages numerically according to the importance of their terms.

The split uses `random_state=42` to make the results reproducible. The TF-IDF vectorizer is fitted on the training messages only and then used to transform the test messages, preventing information from the test set from influencing feature extraction.

### Dataset Split

| Set | Number of messages |
|---|---:|
| Training | 4,459 |
| Testing | 1,115 |
| Total | 5,574 |

The resulting TF-IDF matrices have 7,716 features, representing the vocabulary learned from the training data.

## Machine Learning Models

### 1. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classification algorithm commonly used for text classification. It estimates the probability of a message belonging to each class based on the features present in that message.

It serves as the initial baseline model for this project.

### 2. Logistic Regression

Logistic Regression is a classification algorithm that estimates the probability of a message belonging to a class. It provides a second model for comparison against the Naive Bayes baseline.

Both models are trained using the same training data and TF-IDF features.

## Evaluation Metrics

The models are evaluated using:

- **Accuracy:** The proportion of all messages classified correctly.
- **Precision:** Of the messages predicted as spam, the proportion that are actually spam.
- **Recall:** Of all actual spam messages, the proportion successfully detected.
- **F1-score:** The harmonic mean of precision and recall.
- **Confusion matrix:** Shows correctly and incorrectly classified messages.

For this project, a false negative is a spam message classified as ham, while a false positive is a legitimate message incorrectly classified as spam.

## Initial Results

The following results were obtained on the held-out test set of 1,115 messages.

| Metric | Multinomial Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | 96.23% | 97.49% |
| Spam precision | 100%* | 100%* |
| Spam recall | 71.81% | 81.21% |
| Spam F1-score | 0.84 | 0.90 |
| True negatives | 966 | 966 |
| False positives | 0 | 0 |
| False negatives | 42 | 28 |
| True positives | 107 | 121 |

*Precision is rounded to the displayed value. Both models produced zero false positives on this particular test set.

### Results Analysis

Logistic Regression performed better than Multinomial Naive Bayes on the initial evaluation:

- Accuracy increased from 96.23% to 97.49%.
- Spam recall increased from approximately 71.8% to 81.2%.
- The number of missed spam messages decreased from 42 to 28.

Logistic Regression detected 14 more spam messages than Naive Bayes while producing the same number of false positives on this test set.

These results suggest that Logistic Regression is the stronger of the two models in the current experiment. However, performance on this test set does not guarantee the same results on unseen, real-world messages.

## Current Status

- [x] Load and inspect the dataset.
- [x] Separate message text and labels.
- [x] Split the dataset into training and testing sets.
- [x] Convert messages into TF-IDF features.
- [x] Train Multinomial Naive Bayes.
- [x] Train Logistic Regression.
- [x] Evaluate and compare both models.
- [ ] Investigate the classification threshold and its effect on spam recall and false positives.
- [ ] Validate any improvements without tuning against the test set.
- [ ] Save the trained model and TF-IDF vectorizer.
- [ ] Build a message-entry prediction interface.
- [ ] Complete final testing and documentation.

## Technologies

- Python
- scikit-learn
- TF-IDF text vectorization
- Multinomial Naive Bayes
- Logistic Regression
- Git and GitHub

## Future Improvements

The next step is to investigate whether the Logistic Regression model can detect more spam messages by adjusting its classification threshold. This will involve examining the trade-off between recall and false positives, using validation data for experimentation and retaining the test set for final evaluation.

Once the model has been evaluated, the project will be extended to support predictions on messages entered by a user.
