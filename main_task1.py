import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_data(file): #functie ce citeste datele din fisier

    df = pd.read_csv(file, delimiter =',')
    labels = df['label'].values # citim labelurile
    tweets = df['tweet'].values # tweeturile ( Textele implicit )

    shuffle = StratifiedShuffleSplit(n_splits=1, test_size=0.2) # creem un obiect de tip StratifiedShuffleSplit

    for train_i, test_j in shuffle.split(tweets, labels): # impartim tweeturile si labelurile in date de train si date de test
        tweets_train, tweets_test = tweets[train_i], tweets[test_j]
        labels_train, labels_test = labels[train_i], labels[test_j]

    return tweets_train, tweets_test, labels_train, labels_test


tweets_train, tweets_test, labels_train, labels_test = get_data('train.csv') #apelam functia

count_vectorizer = CountVectorizer(lowercase=True, analyzer='word', stop_words='english') #creem obiectul de tip countvectorizer

count_vectorizer.fit(tweets_train)  #antrenam count_vectorizerul pe baza datelor de train cu functia fit
X_train = count_vectorizer.transform(tweets_train) #Transformam tweeturile de train si de test ( toate ) in vectori cu nr de aparitii ale cuvintelor
X_test = count_vectorizer.transform(tweets_test)

model = MultinomialNB(alpha=1) #creem model de naive bayes
model.fit(X_train, labels_train) #antrenam modelul de naive bayes

predictions = model.predict(X_test) #prezicem rezultatele pe datele de test

print(accuracy_score(labels_test, predictions)) #afisam acuratetea modelului nostru
print(classification_report(labels_test, predictions)) #afisam precision recall f1 score
print(confusion_matrix(labels_test, predictions)) #afisam si matricea de confuzie


# o mica chestie facuta de mine unde bagam un text din input pentru a vedea daca este hate speech conform modelului antrenat
# nu functioneaza tocmai mereu bine

message = input("Introduceti textul pentru a detecta daca este hate speech:")
hate_message = np.array([message])
hate_message_test = count_vectorizer.transform(hate_message) #transformam in vector numeric

predict = model.predict(hate_message_test) #prezicem

if predict[0] == 1:
    print("Mesajul introdus este hate speech.")
else:
    print("Mesajul introdus nu este hate speech.")