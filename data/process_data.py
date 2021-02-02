import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load data from the given csvs
    
    Args:
    messages_filepath: str. filepath to access the csv of all the 
    messages to analyse
    categories_filepath: str. ilepath to access the csv of all the 
    categories these messages could be a part of
    
    Returns:
    df: a dataframe of the merged tables drawn from the csvs
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on=['id'])
        
    return df



def clean_data(df):
    """clean the data by dropping duplicates, wrong values and splitting
    the categories into their own columns

    Args:
    df: dataframe. This is the merged dataframe created in def 'load_data'

    Returns:
    df: a dataframe with individual columns for each category and no duplicates
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    categories.head()
    categories.shape

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[slice(0, -2, 1)])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split("-").str.get(1)
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(['categories'],axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
   
    # drop duplicates
    df = df.drop_duplicates()

    # drop incorrect values in 'related' field
    df = df.drop(df[df.related == 2].index)

    return df


def save_data(df, database_filename):
    """save the dataframe to the appropriate database

    Args:
    df: dataframe. Our tidy new dataframe from def 'clean_data'
    database_filename: str. the name of the database we want to create

    Returns:
    none
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')
  


def main():
    """main method, given from Udacity"""
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