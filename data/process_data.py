import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - string, specify the filepath where the disaster meesages locate.
    categories_filepath - string, specify the filepath where the disaster meesages locate.
    
    OUTPUT
    df - dataframe which have disaster messages and corresponding categories merged.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id", how='inner')
    return df 


def clean_data(df):
    '''
    INPUT
    df - dataframe which includes disaster messages and corresponding categories.
    
    OUTPUT
    df - dataframe (a) that is splitted into category columns
         (b) has category values converted into binary
         (c) with duplicates removed
    '''

    # the column-splitting action
    categories = df['categories'].str.split(';', expand = True)

    # retrieve the name of the catogories
    row = categories.iloc[0].str.split('-', expand = True)
    category_colnames = list(row[0])
    categories.columns = category_colnames

    # sanity check whether any category corresponds to no data
    missing_catogories = []
    for column in categories:
        categories[column] = categories[column].str.get(-1)
        categories[column] = categories[column].astype(int)
        if len(categories[column].unique()) == 1:
            missing_catogories.append(column)


    # binary coversion: give the upper and lower limites of the integer entries   
    categories = categories.clip(0, 1)

    # drop whichever categories are missing
    categories = categories.drop(missing_catogories, axis=1)

    # replace the old categories with the new binary one
    df.drop(['categories'], axis = 1)

    # merge the processed binary categories into the dataframe
    df =  pd.concat([df, categories], axis=1)

    # drop duplicates redundant for machine-training purposes
    df = df.drop_duplicates()

    return df 


def save_data(df, database_filepath):
    '''
    INPUT
    df - processed clean dataframe
    database_filename - string, assign a location where the database wishs to be stored.
    
    OUTPUT
    N/A - a fixed name 'DisasterResponse' is assigned and any newer version would overwrite the old one.  
    '''
    engine = create_engine('sqlite:///'+database_filepath)
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
