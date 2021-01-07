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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# download stopwords
nltk.download('stopwords')


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
    df = pd.read_sql_table(database_filepath, con=engine)
    
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
    comment
    '''
    
    # normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize
    tokens = word_tokenize(text)
    
    # remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # lemmatize
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    
    return lemmed
    
def build_model():
    '''
    comment
    '''
    
    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer(smooth_idf=False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # print(pipeline.get_params())
    
    # define parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__max_depth': [1, 5, 10, 25],
        'clf__estimator__max_features': [*np.arange(0.1, 1.1, 0.1)]
    }   
    
    # compute grid search
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv
      


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    comment
    '''
    
    # predict test values
    y_pred = model.predict(X_test)
    
    # calculate f1-score, precision and recall
    fscore = []
    precision = []
    recall = []
    
    print('results by category:')
    print('')
    
    for i in range(y_true.shape[1]):
        prec, rec, f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')
        fscore.append(f1)
        precision.append(prec)
        recall.append(rec)
        
        print('category: ', category_names[i])
        print('f_score: ', f1)
        print('precision: ', prec)
        print('recall: ', rec)
        print('')
    
    print('final results:')
    print('the mean f1 score is: {}'.format(pd.Series(fscore).mean()))
    print('the mean precision is: {}'.format(pd.Series(precision).mean()))
    print('the mean recall is: {}'.format(pd.Series(recall).mean()))

def save_model(model, model_filepath):
    '''
    comment
    '''
    
    # save model to pickle
    pkl.dump(model, open(model_filepath, 'wb'))


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