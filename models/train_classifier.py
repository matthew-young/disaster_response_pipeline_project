import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pickle
import sqlite3
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """load the data from a specified database and get the independent
    and dependent variables and the category names for modeling
    
    Args:
    database_filepath: str. the filepath to access the database and 
    download the data
    
    |Returns:
    X: the independent variable, the messages themselves
    Y: the dependent variable, all the different categories
    category_names: no surprise, the name of the dependent variable 
    categories 
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names = df.columns.drop(['id','message','original','genre']).tolist()

    return X, Y, category_names

def tokenize(text):
    """tokenize the messages for easier analysis
    
    Args: 
    text: str. the messages we're looking to break up into their words
    
    Returns:
    clean_tokens: the tokenized, lemmatized, lowercased and de-spaced 
    (stripped) words
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """ building the pipeline for data modeling
    
    Args:
    none
    
    Returns:
    pipeline: our pipeline of CountVectorizer, TfidfTransformer and
    MultiOutplutClassifier
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluating the model by fitting the data to the model, predicting 
    the results and conducting a gridsearch to optimize
    
    Args:
    model: our pipeline from def 'build_model'
    X_test: array. our test values for X
    Y_test: array. our test values for Y
    category_names: the names of the Y categories

    Returns:
    none
    """
    #fit the data to the model
    model.fit(X_test, Y_test)
    #get predicted values
    y_pred = model.predict(X_test)
    #set parameters for gridsearch
    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__max_df': (0.5, 0.75, 1.0),
    'tfidf__use_idf': (True, False),
    'clf__estimator__n_neighbors': (4,5,6),
    'clf__estimator__leaf_size':(30, 35, 40)
    }
    #execute gridsearch
    cv = GridSearchCV(model, param_grid=parameters)
    
    #print results
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test[i], y_pred[i]))


def save_model(model, model_filepath):
    """save the model to a pickle file
    
    Args:
    model: our"""
    save_classifier = open(model_filepath, 'wb')
    pickle.dump(model, save_classifier)
    save_classifier.close()



def main():
    """main method, given from Udacity"""
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