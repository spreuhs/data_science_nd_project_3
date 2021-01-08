import sys

# import libraries
import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# download stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('english')

def load_data(database_filepath):
    '''
    load files from database and split into features and labels
    
    Args:
        database_filepath: path to database file
        
    Return:
        X: dataframe containing features
        Y: dataframe containing labels
        category_names: list containing category names
    '''
    
    # create SQL engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    # load data from database
    df = pd.read_sql_table('my_table', con=engine)
    
    # Split into feature and label
    category_names = ['related', 'request',
           'offer', 'aid_related', 'medical_help', 'medical_products',
           'search_and_rescue', 'security', 'military', 'child_alone', 'water',
           'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',
           'death', 'other_aid', 'infrastructure_related', 'transport',
           'buildings', 'electricity', 'tools', 'hospitals', 'shops',
           'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
           'storm', 'fire', 'earthquake', 'cold', 'other_weather',
           'direct_report']
    X = df['message']
    Y = df[category_names]
       
    return X, Y, category_names


def tokenize(text):
    '''
    normalize, clean and tokenize text
    
    Args:
        text: input text for tokenization
    
    Return:
        lemmed: cleaned tokens
    '''
    
    # normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize
    tokens = word_tokenize(text)
    
    # remove stopwords
    tokens = [w for w in tokens if w not in stop_words]
    
    # lemmatize
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    
    return lemmed
    
def build_model():
    '''
    builds and returns the classifier
    
    Args:
        None
        
    Return:
        cv: classifier
    '''
    
    # create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize, smooth_idf=False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # print(pipeline.get_params())
    
    # define parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__max_depth': [1, 10, 20],
        'clf__estimator__max_features': ['auto', 'sqrt'],
    }   
    
    # compute grid search
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1,verbose=3, cv=3)

    return cv
      


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluates model and creates console output
    
    Args:
        model: trained model
        X_test: test split of X
        Y_test: test split of Y
        category_names: list containing category names
    
    Return:
        None
    '''
    
    # predict test values
    y_pred = model.predict(X_test)
    
    # calculate f1-score, precision and recall
    for i in range(Y_test.shape[1]):
        result = classification_report(Y_test.iloc[:,i], y_pred[:,i])
        print("Report on", category_names[i], ":")
        print(result)

def save_model(model, model_filepath):
    '''
    saves the model als pkl
    
    Args:
        model: model to be saved
        model_filepath: location and filename
        
    Return:
        None        
    '''
    
    # save model to pickle
    pickle.dump(model, open(model_filepath, 'wb'))
    
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