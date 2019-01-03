import nltk
import pandas as pd
import re

NUMBERS_OF_ROW = 20


def clean_input_data(df):
    print('Checking for Null entries : %s' % df.isnull().sum())
    print('Old size: %d' % len(df))
    df = df.dropna()
    print('New size: %d' % len(df))

    # change all the strings to lower case
    print("Changing to lower case..")
    df.loc[:, 'bread'] = (df.loc[:, 'bread']).str.lower()

    # tokenize input strings into series of key words
    print("Tokenizing...")

    df.loc[:, 'bread'] = df.apply(lambda row: re.sub('\W+', ' ', row['bread']), axis=1)
    df.loc[:, 'bread'] = df.apply(lambda row: row['bread'].split(), axis=1)

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
    print("Cleaning raw input data...")
    input_df = clean_input_data(raw_input_df)

    print("Creating labels...")
    input_df.loc[:, 'label'] = 0

    for index, row in input_df.iterrows():
        if index == 10:
            break
        print("Processing index : %d" % index)
        if "clothing" in row['bread']:
            input_df.loc[index, 'label'] = 1

    input_df.to_csv("training_input.csv")

