import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, fbeta_score
from sklearn import preprocessing
import pickle

import warnings
warnings.filterwarnings('always') 
import numpy as np
def load_data(database_filepath):
    '''
    INPUT
    database_filepath - string, specify the filepath where the SQL database DisasterResponse.db locates.
    
    OUTPUT
    X - the sampled disaster messages
    Y - the sampled labels for the disaster messages
    category_names - the diaster categories implemented in machine-learning-model classification
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    return X, Y, category_names

def tokenize(text):
    '''
    INPUT
    text - string, text message including punctuation marks
    
    OUTPUT
    clean_tokens - the tokenized and lematized text samples prepared for training
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, 'url_place_holder_string')
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='n').strip()
        clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    '''
    INPUT
    N/A - a prototype model is defined via a machine-learning pipeline.

    
    OUTPUT
    model - the machine model to be trained based on the sampled X_train, Y_train 
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC()))
    ])

    parameters = {"clf__estimator__C": [1.0, 10.0, 100.0, 1000.0]}
    model = GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy',n_jobs=-1)
    return model
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model - the sklearn machine-learning model just trained
    X_test - the sampled messages splitted from X for testing purposes
    Y_test - the sampled labels splitted from Y for testing purposes
    
    OUTPUT
    report - the text report showing main classification metrics: precision, recall, F1-score on each category  
    '''
    Y_pred = model.predict(X_test)
    report= classification_report(Y_pred,Y_test, target_names=category_names)
    print(report)
    return report

def save_model(model, model_filepath):
    '''
    INPUT
    model - the sklearn machine-learning model just trained and tested
    model_filepath - string, specify the filepath where the model classifier.pkl should be saved.
    
    OUTPUT
    N/A - save the model in the designated file path.
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
        
        
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
