from transformers import StartingVerbExtractor, NumCharacters, WordCount, SentimentScore
import sys
import nltk
import pickle
import pandas as pd
from sqlalchemy import create_engine

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from utils import tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report



def load_data(database_filepath):
    '''
    For loading our data.
    INPUT:
    database_filepath - the path of the db file we want to read
    OUTPUT:
    X - a column of the text data
    Y - a matrix of dependent variables (the categories of the messages)
    category_names - the column names of the categories in Y
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(con=engine, table_name='DisasterResponse')
    X = df['message']
    Y = df.drop(columns=['id','original','message','genre'])
    category_names = Y.columns
    return X, Y, category_names


def build_model():
    '''
    Create our pipeline where we create features from our transformers file along with
    the TFIDF of the messages.
    These features include:
    1. True or False if a message starts with a Verb
    2. Number of characters in a message
    3. Sentiment score of a message
    4. Number of words in the message
    This is then fed into a Random Forest classifier and then a gridsearch 
    cross validation is done to return the model with the best parameters
    OUTPUT:
    cv - which is the model with the best parameters from our parameter grid.
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            # I've tried configurations of features to find the best accuracy but you're welcome to try
            ('starting_verb', StartingVerbExtractor()),
            #('num_chars', NumCharacters()),
            #('sentiment', SentimentScore()),
            ('wordcount', WordCount())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = { 
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__stop_words': (None, 'english'),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline

    


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model by giving a classification report for every category.
    The classification report builds a text report showing the main classification metrics
    such as f1 score, precision and recall.
    INPUT:
    model - the model from the above function
    X_test - the test set of the features
    Y_test - the test set of the dependent variables
    category_names - the names of the dependent variables
    OUTPUT:
    Prints the classification report for every category of the model
    '''
    
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=Y_test.columns))


def save_model(model, model_filepath):
    '''
    Save the model so we can use it in the webapp
    INPUT:
    model - our multiclass model
    model_filepath - where we want to save the model
    OUTPUT:
    A pickle file of our model in the desired filepath.
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()