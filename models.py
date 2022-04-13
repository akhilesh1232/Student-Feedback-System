import pandas as pd
import numpy as np
import pickle
# import nltk
# nltk.download()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

class Models:

    def __init__(self):
        self.name = ''
        path = 'dataset/trainingdata.csv'
        df = pd.read_csv(path)
        df = df.dropna()
        self.x = df['sentences']
        self.y = df['sentiments']

    def svm_classifier(self):
        self.name = 'SVM classifier'
        classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))])
        classifier = classifier.fit(self.x, self.y)
        pickle.dump(classifier,open(self.name + '.pkl', "wb"))
        return classifier

    def accuracy(self, model):
        predicted = model.predict(self.x)
        accuracy = np.mean(predicted == self.y)
        print(f"{self.name} has accuracy of {accuracy * 100} % ")

class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


if __name__ == '__main__':
    model = Models()
    model.accuracy(model.svm_classifier())

