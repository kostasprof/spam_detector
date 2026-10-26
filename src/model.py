import data_cleaner
import main
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

'''
Naive Bayes training 
'''
def train_bayes_model(train_vectors,labels):
    mnb = MultinomialNB()
    mnb.fit(train_vectors,labels)
    return mnb

'''
Logistic regression training
'''
def train_lr_model(train_vectors,labels):
    lrm = LogisticRegression()
    lrm.fit(train_vectors,labels)
    return lrm

'''
Model evaluation
'''
def evaluate_model(bm, lrm, text_vectors, test_labels):
    
    # Evaluate Multinomial Naive Bayes
    bm_predictions = bm.predict(text_vectors)
    bm_accuracy = accuracy_score(test_labels, bm_predictions)

    print("\n===== Multinomial Naive Bayes =====")
    print(f"Accuracy: {bm_accuracy:.4f}")
    print(f"Classification report:\n{classification_report(test_labels, bm_predictions)}")

    bm_confusion = confusion_matrix(test_labels, bm_predictions)
    print(f"True negatives (ham correctly identified): {bm_confusion[0, 0]}")
    print(f"False positives (ham incorrectly flagged as spam): {bm_confusion[0, 1]}")
    print(f"False negatives (spam missed and classified as ham): {bm_confusion[1, 0]}")
    print(f"True positives (spam correctly identified): {bm_confusion[1, 1]}")

    # Evaluate Logistic Regression
    lrm_predictions = lrm.predict(text_vectors)
    lrm_accuracy = accuracy_score(test_labels, lrm_predictions)

    print("\n===== Logistic Regression =====")
    print(f"Accuracy: {lrm_accuracy:.4f}")
    print(f"Classification report:\n{classification_report(test_labels, lrm_predictions)}")

    lrm_confusion = confusion_matrix(test_labels, lrm_predictions)
    print(f"True negatives (ham correctly identified): {lrm_confusion[0, 0]}")
    print(f"False positives (ham incorrectly flagged as spam): {lrm_confusion[0, 1]}")
    print(f"False negatives (spam missed and classified as ham): {lrm_confusion[1, 0]}")
    print(f"True positives (spam correctly identified): {lrm_confusion[1, 1]}")
