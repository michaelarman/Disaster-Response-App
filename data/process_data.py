import sys
import pandas as pd
import string
import nltk
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    loads the two datasets and merges both by the primary key,
    i.e. the id variable
    INPUT:
    messages_filepath - filepath for disaster_messages.csv
    categories_filepath - filepath for disaster_categories.csv
    OUTPUT:
    df - the merged dataframe
    .'''
    df1 = pd.read_csv(messages_filepath)
    df2 = pd.read_csv(categories_filepath)
    df = df1.merge(df2, on='id')
    return df


def clean_data(df):
    '''
    takes the df and cleans the data by first one-hot encoding the categories
    and then removing the redundant columns and removing the duplicates. Then
    normalizing the text by removing the stopwords and punctuation and making all lower case
    INPUT:
    df - the df that was loaded from load_data
    OUTPUT:
    df - cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split('-').str[0].values
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df = df.drop(columns=['categories'])
    df.reset_index()
    df = pd.concat([df,categories], axis=1)
    df = df.drop_duplicates()
    # remove columns containing all 0 such as the child_alone column
    df = df.loc[:, (df != 0).any(axis=0)]
    # make message column lowercase
    df['message'] = df['message'].str.lower()
    # remove punctuation and not stop words since we can configure that in the tfidf
    df['message'] = df['message'].str.translate(str.maketrans('', '', string.punctuation))
    
    return df


def save_data(df, database_filename):
    '''
    Saves the data into a sqlite db file
    INPUT:
    df - the cleaned dataframe
    database_filename - where to store the database we create
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)  


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