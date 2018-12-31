import pandas as pd
import math
import re
import nltk

NUMBERS_OF_ROW = 20


def clean_input_data(df):

    print('checking for Null entries : %s' % df.isnull().sum())
    print('Old size: %d' % len(df))
    df = df.dropna()
    print('New size: %d' % len(df))

    # change all the strings to lower case
    df.loc[:, 'bread'] = df.loc[:, 'bread'].str.lower()

    # tokenize input strings into series of key words
    df.loc[:, 'bread'] = df.apply(lambda row: re.split("[ \*~&)(,+ ]+", row['bread']), axis=1)

    # TODO remove duplicates

    # TODO remove empty

    return df


def get_labelled_data():

    print('Reading input data file...')
    raw_input_df = pd.read_csv("input.csv")

    # creating raw input data frame
    raw_input_list = raw_input_df['bread1'].values.tolist() + raw_input_df['bread2'].values.tolist()
    raw_input_df = pd.DataFrame(raw_input_list, columns=['bread'])

    # cleaning the raw input data
    print("cleaning raw input data...")
    input_df = clean_input_data(raw_input_df)

    print("creating labels...")
    input_df.loc[:, 'labels'] = 0
    for index, row in input_df.iterrows():
        print("Processing index : %d" % index)
        for s in row['bread']:
            if "mobile" in s:
                input_df.loc[index, 'labels'] = 1

    print(input_df.head())
    input_df.to_csv("training_input.csv")


get_labelled_data()
