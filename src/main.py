import data_cleaner as dc
import model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    messages = dc.load_dataset()
    
    texts,labels = dc.prepare_dataset(messages)

    texts_train, texts_test, labels_train, labels_test = dc.split_dataset(texts,labels)
    
    vectorizer,train_vect,test_vect = dc.vectorize_data(texts_train,texts_test)
    
    trained_bm_model = model.train_bayes_model(train_vect,labels_train)
    
    trained_lr_model = model.train_lr_model(train_vect,labels_train)
    
    model.evaluate_model(trained_bm_model,trained_lr_model,test_vect,labels_test)
    
    
    #get the probability of spam for each message
    probabilities = trained_lr_model.predict_proba(test_vect)
    spam_probabilities = probabilities[:, 1]

    #experiment with a lower threshold
    threshold = 0.3

    #classify each message using the new threshold
    threshold_predictions = np.where(
        spam_probabilities >= threshold,
        "spam",
        "ham"
    )

    # Evaluate the predictions
    print(f"\n===== Logistic Regression (threshold={threshold}) =====")
    print(f"Accuracy: {accuracy_score(labels_test, threshold_predictions):.4f}")
    print(classification_report(labels_test, threshold_predictions))
    print(confusion_matrix(labels_test, threshold_predictions, labels=["ham", "spam"]))
    
if __name__ == "__main__":
    main()