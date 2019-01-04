import numpy as np
from gensim.models import Word2Vec
from sklearn import svm
from sklearn.model_selection import train_test_split

from classifer.generate_training_data import get_labelled_data


def train_w2v_model(input_data):
    print("Model starts running....")
    data = input_data.loc[:, 'bread']
    model = Word2Vec(data, min_count=1)
    model.train(data, total_examples=len(data), epochs=1)
    model.save("word2vec.model")
    print("Model finished!")
    return model


def do_word_embedding(input_data):
    # model = train_w2v_model(input_data)
    model = Word2Vec.load("word2vec.model")

    X_final = np.zeros((len(input_data), 50, 100))
    for index, row in input_data.iterrows():
        if index == 100000:
            break
        if len(row['bread']) == 0:
            continue
        try:
            X_tmp = model[row['bread']]
            col = len(row['bread'])
            X_final[index, 0:col] = X_tmp
        except Exception as e:
            print(e)
            print(row['bread'])
    return X_final


# THIS HAS EVERYTHING
def product_classifier():
    print("Getting labelled data from csv file...")
    input_df = get_labelled_data()
    print("Labels generated successfully!")

    print("Doing Word Embedding on the raw input data...")
    inputs = do_word_embedding(input_df)
    print("Data cleaned successfully!")

    print("Mapping data for SVM classifier...")
    inputs_array = np.asarray(inputs[:, :, -1])
    labels_array = np.asarray(input_df.loc[:, 'label'])

    print("Splitting the data...")
    train_input, test_input, train_labels, test_labels = train_test_split(inputs_array, labels_array)

    print("Running the SVM...")
    classifier = svm.SVC(gamma='auto')
    classifier.fit(train_input, train_labels)

    print("Checking the accuracy...")
    acc = classifier.score(test_input, test_labels)
    print("Accuracy: " + str(acc))


product_classifier()
