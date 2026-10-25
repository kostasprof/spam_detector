import data_cleaner
import main
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
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
def tran_lr_model(train_vectors,labels):
    pass


'''
Model evaluation
'''
def evaluate_model(model,text_vectors,test_labels):
    
    predictions = model.predict(text_vectors)
    accuracy = accuracy_score(test_labels,predictions)
    print(f"Current accuracy:{accuracy}")
    print(f"Classification report:{classification_report(test_labels,predictions)}")
    confusion = confusion_matrix(test_labels,predictions)
    print(f"True negatives (ham correctly identified): {confusion[0, 0]}")
    print(f"False positives (ham incorrectly flagged as spam): {confusion[0, 1]}")
    print(f"False negatives (spam missed and classified as ham): {confusion[1, 0]}")
    print(f"True positives (spam correctly identified): {confusion[1, 1]}")
    
    return True