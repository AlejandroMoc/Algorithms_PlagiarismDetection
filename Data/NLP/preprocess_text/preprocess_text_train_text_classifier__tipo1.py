from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from typing import List

def text_classifier_train(training_documents: List[str], training_labels: List[int]):
    text_vectorizer = CountVectorizer()
    X_train = text_vectorizer.fit_transform(training_documents)
    text_classifier = MultinomialNB()
    text_classifier.fit(X_train, training_labels)
    return text_classifier, text_vectorizer