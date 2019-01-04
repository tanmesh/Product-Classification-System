import re

import pandas as pd

NUMBERS_OF_ROW = 20


def clean_input_data(df):
    print('Checking for Null entries : %s' % df.isnull().sum())
    print('Old size: %d' % len(df))
    df = df.dropna()
    print('New size: %d' % len(df))

    print("Changing all the strings to lower case..")
    df.loc[:, 'bread'] = (df.loc[:, 'bread']).str.lower()

    print("Tokenizing input strings into series of key words...")

    df.loc[:, 'bread'] = df.apply(lambda row: re.sub('\W+', ' ', row['bread']), axis=1)
    df.loc[:, 'bread'] = df.apply(lambda row: row['bread'].split(), axis=1)

    # TODO remove duplicates

    # TODO remove empty

    return df


def get_labelled_data():
    print('Reading input data file...')
    raw_input_df = pd.read_csv("input.csv")

    print("Creating raw input data frame...")
    raw_input_list = raw_input_df['bread1'].values.tolist() + raw_input_df['bread2'].values.tolist()
    raw_input_df = pd.DataFrame(raw_input_list, columns=['bread'])

    print("Cleaning raw input data...")
    input_df = clean_input_data(raw_input_df)

    print("Creating labels...")
    input_df.loc[:, 'label'] = 0

    for index, row in input_df.iterrows():
        if index == 100000:
            break
        print("Processing index : %d" % index)
        if "clothing" in row['bread']:
            input_df.loc[index, 'label'] = 1

    input_df.to_csv("training_input.csv")

    return input_df
