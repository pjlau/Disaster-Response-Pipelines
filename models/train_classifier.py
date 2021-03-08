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
#import warnings
#warnings.simplefilter(action="ignore", category=FutureWarning)
import warnings
warnings.filterwarnings('always') 
import numpy as np
def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    return X, Y, category_names

def tokenize(text):
    #text = re.sub(r'[^a-zA-Z0-9]', ' ',text)
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, 'url_place_holder_string')
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='n').strip()
        #I passed in the output from the previous noun lemmatization step. This way of chaining procedures is very common.
        clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        #('clf', MultiOutputClassifier(RandomForestClassifier()))
        ('clf', MultiOutputClassifier(LinearSVC()))
    ])
    #parameters = [{'tfidf__norm': ['l1','l2'],'clf__estimator__criterion': ["gini", "entropy"],'clf__estimator__max_depth': [30,40],'clf__estimator__n_estimators':[10,20,40]}]
    #parameters = {'clf__estimator__n_estimators':[1,10,100,1000],'clf__estimator__min_samples_split': [2,3,4,5]}
    #parameters = [{'tfidf__norm': ['l1','l2'],'clf__estimator__criterion': ["gini", "entropy"]}, {"clf": [LinearSVC()],
    # "clf__estimator__C": [1.0, 10.0, 100.0, 1000.0]}]
    parameters = {"clf__estimator__C": [1.0, 10.0, 100.0, 1000.0]}
   # parameters = {"clf__estimator__C": [0.1,1.0, 10.0, 100.0, 1000.0],"clf__estimator__max_iter":[50,100],"clf__estimator__intercept_scaling":[0.1,0.5,1.0,1.5,2.0]}
    #parameters = {'tfidf__norm': ['l1','l2'],'clf__estimator__learning_rate':[0.05,0.10,0.15,0.20],'clf__estimator__n_estimators':[10,25,50]}
   #return pipeline
    model = GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy',n_jobs=-1)
    return model
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    report= classification_report(Y_pred,Y_test, target_names=category_names)
    print(report)
    return report

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #Y = Y.astype(int)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
        
        #X_train = preprocessing.scale(X_train) 
        #X_test = preprocessing.scale(X_test) 
        
        print('Building model...')
        model = build_model()
        
        #labels = np.unique(Y_train)
        #print(labels)
        #print(Y_train.shape) #(20972,36)
        #sumY_train = Y_train.sum(axis=0)
        #print(X_train.shape)
        #print(X_test.shape)
        #print(Y_train.shape)
        #print(Y_test.shape)
        #X_train.drop(['child_alone'], axis = 1, inplace = True)
        #X_test.drop(['child_alone'], axis = 1, inplace = True)
        #Y_train.drop(['child_alone'], axis = 1, inplace = True)
        #Y_test.drop(['child_alone'], axis = 1, inplace = True)
        #sumY_train = Y_train.sum(axis=0)
        #print(sumY_train) 
        #Y_train.loc[0:10, 'child_alone']=1
        #sumY_train = Y_train.sum(axis=0)
        #print(sumY_train) 
        
        #print(sumY_train) 
        #for i in range(Y_train.shape[1]):
        #    if sumY_train[i]==0:
        #        Y_train.loc[0, i]=1
        
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
