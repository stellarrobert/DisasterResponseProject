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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pickle 

def load_data(database_filepath):
    # Arguments: 
    #    database_filepath is the name of the desired table 
    # Loads data from sqlite database, splits, outputs dataframe 
    
    engine = create_engine('sqlite:///cleaned_disaster.db')
    df = pd.read_sql_table(database_filepath, engine)
    df = df.dropna()
    X = df.message
    Y = df.drop(['id', 'original', 'genre', 'message'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    # splits text into individual words, removes the non-informative ones, and takes words to their root
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for word in words:
        clean_tok = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    # constructs NLP pipeline
    # trains on training dataset
    
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf' , TfidfTransformer()),
                        ('clf', RandomForestClassifier())
                        ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # makes prediction on the test set, checks performance
    
    y_pred = model.predict(X_test)

    for i, column in enumerate(Y_test):
        #print(str(column).upper())
        return classification_report(Y_test[column], y_pred[:, i])
        
        
def save_model(model, model_filepath):
    # saves the trained model as a pickle file
    
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