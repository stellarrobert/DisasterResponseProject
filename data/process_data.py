import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk import pos_tag, ne_chunk

from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine




def load_data(messages_filepath, categories_filepath):
    """
    Loads the appropriate csv files as dataframes, joins them
    
    Args:
        messages_filepath: a filepath for the messages (tweets) 
        categories_filepath: a filepath for the categories into which the messages are classified

    Returns: 
        A single dataframe used as an input in clean_data()
    """    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(categories, messages, how='inner', on='id')
    return df


def clean_data(df):
    """
    Splits column containing all possible categories into separate features
    Binary classification for each category feature
    Eliminates duplicates, outputs clean dataframe
    
    Args:
        df: the text df output by load_data()

    Returns: 
        A dataframe properly formatted for an NLP model
    """   
   
    categories_split = df["categories"].str.split(";", expand = True) 
    row = categories_split.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    categories_split.columns = category_colnames
    
    for column in categories_split:
        categories_split[column] = categories_split[column].apply(lambda x: x[-1])
        categories_split[column] = categories_split[column].astype('int')
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories_split], axis =1)
    df = df.drop_duplicates()
    return df
    
def save_data(df, database_filename):
    """
    Saves cleaned dataframe in an sqlite database
    
    Args:
        df: the cleaned/formatted dataframe returned by clean_data()
        database_filename: the desired table name in sqlite database

    Returns: 
        No objects returned
    """   
    
    engine = create_engine('sqlite:///cleaned_disaster.db')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
