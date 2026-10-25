from pathlib import Path
from statistics import median
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import model
dataset_path = Path(__file__).resolve().parent.parent / "data" / "SMSSpamCollection"


#load the dataset into a list
def load_dataset():
    
    messages = []
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            label, message = line.rstrip("\n").split("\t", 1)
            messages.append((label, message))
            
    return messages

'''
Seperate labels with message into 2 lists
'''
def prepare_dataset(messages):
    texts = [message[1] for message in messages]
    labels = [message[0] for message in messages]
    return texts,labels
        
'''
Count spam/ham messages and the % of spam
'''
def analyze(messages):
    
    spam_counter = 0
    ham_counter = 0 
    for message in messages:
        if message[0] == "spam":
            spam_counter += 1
        else:
            ham_counter += 1
    spam_percent = (spam_counter*100)/(ham_counter+spam_counter)
    return spam_counter,ham_counter,spam_percent

'''
Find the longest message for each category and the average length of messages 
for each category
'''
def message_stats(messages,spam_count,ham_count):
    total_spam_words = 0
    spam_msg_size = 0
    max_spam_length = 0
    
    total_ham_words = 0
    ham_msg_size = 0
    max_ham_length = 0
    
    for message in messages:
        
        if message[0] == "spam":
            total_spam_words += len(message[1].split())
            spam_msg_size += len(message[1])
            if len(message[1]) > max_spam_length: max_spam_length = len(message[1])
            
        elif message[0] == "ham":
            total_ham_words += len(message[1].split())
            ham_msg_size += len(message[1])
            if len(message[1]) > max_ham_length: max_ham_length = len(message[1])
                    
    average_words_spam = total_spam_words/spam_count
    average_spam_size = spam_msg_size/spam_count
    
    average_words_ham = total_ham_words/ham_count
    average_ham_size = ham_msg_size/ham_count
    
    spam_stats = [average_words_spam,average_spam_size,max_spam_length]
    ham_stats = [average_words_ham,average_ham_size,max_ham_length]
    
    return spam_stats,ham_stats

'''
Find the median message length in characters for spam or ham messages
'''
def find_median(messages):
    ham_lengths = []
    spam_lengths = []
    for message in messages:
        if message[0] == "spam":
            spam_lengths.append(len(message[1]))
        elif message[0] == "ham":
            ham_lengths.append(len(message[1]))
            
    ham_median = median(ham_lengths)
    spam_median = median(spam_lengths) 
    return spam_median,ham_median

'''
Split the data set to a training and testing dataset
'''
def split_dataset(texts,labels):
    #Make test and train data sets
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts,
        labels,
        train_size = 0.8,
        random_state = 42,
        stratify = labels
    )
    print(f"Training texts total:{len(texts_train)}")
    print(f"Testing texts total:{len(texts_test)}")
    print(f"Training labels total:{len(labels_train)}")
    print(f"Testing labels total:{len(labels_test)}")
    
    return texts_train, texts_test, labels_train, labels_test

'''
Convert text into TF-IDF features
'''
def vectorize_data(train_text,test_text):
    vectorizer = TfidfVectorizer()
    x_train_vect = vectorizer.fit_transform(train_text)
    x_test_vect = vectorizer.transform(test_text)
    
    print(f"Train shape:{x_train_vect.shape}")
    print(f"Test shape:{x_test_vect.shape}")
    return x_train_vect,x_test_vect

