import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id", how='inner')
    return df 


def clean_data(df):
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0].str.split('-', expand = True)
    category_colnames = list(row[0])
    #print(category_colnames)
    categories.columns = category_colnames
    #print(categories.columns)
    missing_catogories = []
    for column in categories:
        categories[column] = categories[column].str.get(-1)
        categories[column] = categories[column].astype(int)
        if len(categories[column].unique()) == 1:
            missing_catogories.append(column)
        #print(categories[column])
       
    categories = categories.clip(0, 1)
    categories = categories.drop(missing_catogories, axis=1)
    
    df.drop(['categories'], axis = 1, inplace = True)
    df =  pd.concat([df, categories], axis=1, join="inner").drop_duplicates()
   # df.drop(['child_alone'], axis = 1, inplace = True)
    return df 


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


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
