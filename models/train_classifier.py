import sys
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pickle 

def load_data(database_filepath):
    """
    Loads data from sqlite database, splits, outputs dataframe 
    
    Args:
        database_filepath: the name of the desired table 
    
    Returns: 
        X: the features to predict the labels
        Y: the message labels
        category_names: the various names of message categories
    """   
    
    engine = create_engine('sqlite:///cleaned_disaster.db')
    df = pd.read_sql_table(database_filepath, engine)
    df = df.dropna()
    X = df.message
    Y = df.drop(['id', 'original', 'genre', 'message'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Splits text into individual words, removes the non-informative ones, and takes words to their root
    
    Args:
        text: the text used to train/test NLP model

    Returns: 
        clean_tokens: processed text for the NLP model
    """   
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for word in words:
        clean_tok = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Constructs NLP pipeline, performs grid search to find best parameters
    
    Args:
        None

    Returns:
        A pipeline to build a classifying model 
    """   

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf' , TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.85, 1.0),
        'tfidf__use_idf': (True, False)
        #,'estimator__clf__min_samples_split': [2, 4] 
        }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Makes prediction on the test set, checks performance
    
    Args:
        model: the NLP model
        X_test: the features needed to test the model
        Y_test: the labels used to test the model
        category_names: the various names of message category labels
    
    Returns: 
        A report of the model's performance on test data
    """   

    y_pred = model.predict(X_test)

    for i, column in enumerate(Y_test):
        #print(str(column).upper())
        return classification_report(Y_test[column], y_pred[:, i])
        
        
def save_model(model, model_filepath):
    """
    Saves the trained model as a pickle file
    
    Args:
        model: the trained NLP model
        model_filepath: the desired file path for the model

    Returns: 
        None
    """   
    
    pickle.dump(model, open(model_filepath, 'wb'))
    #loaded_model = pickle.load(open(model_filepath, 'rb'))

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
