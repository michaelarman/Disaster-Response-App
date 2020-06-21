from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob
import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """A transformer to use for the pipeline that adds a
     True or False if the message starts with a verb"""
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(nltk.word_tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
   

class NumCharacters(BaseEstimator, TransformerMixin):
    """A transformer to use for the pipeline that adds a 
    feature for the number of characters in the message"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(pd.Series(X).apply(lambda x: len(x)).values)
    
       
class WordCount(BaseEstimator, TransformerMixin):
    """A transformer to use for the pipeline that adds a 
    feature for the number of words in the message"""  
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(pd.Series(X).apply(lambda x: len(str(x).split(" "))).values)
    
class SentimentScore(BaseEstimator, TransformerMixin):
    """A transformer to use for the pipeline that adds a
    feature for the sentiment score of a message. The possible
    scores range from [-1,1] where -1 is a negative sentiment and 1 
    is a positive sentiment"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return pd.DataFrame(pd.Series(X).apply(lambda x: TextBlob(x).sentiment[0]).values)