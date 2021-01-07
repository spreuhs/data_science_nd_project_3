import sys

# import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load and merge datasets
    
    Args:
        messages_filepath: path to message .csv file
        categories_filepath: path to categories .csv file
    
    Return:
        df: merged dataframe
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df_load = categories.merge(messages, on='id')
	
    return df_load

def clean_data(df):
    '''
    cleans the dataframe by:
        - cleaning and one-hot encoding category column
        - dropping duplicates
        
    Args:
        df: raw dataframe
    
    Return:
        df: cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0,:]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [item[0] for item in row.str.split('-')]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [item[1] for item in categories[column].str.split('-')]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
        
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
	
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    saves the data into an sqlite database
    
    Args:
        df: dataframe to save
        database_filename: name of the database
        
    Return:
        None
    '''
    
    # create SQL engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    # write to SQL database
    df.to_sql(database_filename, engine, index=False)

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