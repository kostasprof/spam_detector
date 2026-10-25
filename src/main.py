import data_cleaner as dc
import model

def main():
    messages = dc.load_dataset()
    
    texts,labels = dc.prepare_dataset(messages)

    texts_train, texts_test, labels_train, labels_test = dc.split_dataset(texts,labels)
    
    train_vect,test_vect = dc.vectorize_data(texts_train,texts_test)
    
    trained_model = model.train_bayes_model(train_vect,labels_train)
    
    model.evaluate_model(trained_model,test_vect,labels_test)
    
if __name__ == "__main__":
    main()